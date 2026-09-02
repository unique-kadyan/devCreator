"""Scene rendering and final assembly.

Two decisions worth stating. First, scenes render to individual mp4s and are then
stream-copy concatenated: re-encoding the whole episode after rendering would double the
most expensive step for no quality gain, and per-scene files make a crashed job resumable
at scene granularity. Second, transitions are baked into the *incoming* scene rather than
applied to the concatenated master, because an xfade across the join would force a full
re-encode of both sides.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..core.db import tx
from ..core.errors import RenderError
from ..core.logging import get_logger
from ..characters.species import relative_scale
from ..media.animation.grade import light_from_plate
from ..media.animation.parallel import CharacterSpec, SceneSpec, render_scene_parallel

log = get_logger("assemble")

# Shot -> where the camera looks. A close-up that stays centred on the world plate is not a
# close-up of anybody, so the focus follows the speaking character when there is one.
SPEAKER_FOCUS_SHOTS = {"close_up", "extreme_close_up", "over_shoulder", "medium", "insert"}

TRANSITION_FADE_S = {"cut": 0.0, "dissolve": 0.6, "fade_black": 0.8, "whip_pan": 0.25,
                     "iris": 0.5}


@dataclass
class RenderedScene:
    scene_id: int
    index: int
    path: Path
    duration_s: float
    stats: dict


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _focus_for(scene: dict, staging: dict, speaker: str | None) -> tuple[float, float]:
    if speaker and speaker in staging and scene["shot"] in SPEAKER_FOCUS_SHOTS:
        st = staging[speaker]
        # Aim at the head, not the feet: `y` in staging is where the feet sit.
        return float(st.get("x", 0.5)), max(0.18, float(st.get("y", 0.93)) - 0.34)
    if staging:
        xs = [float(v.get("x", 0.5)) for v in staging.values()]
        return sum(xs) / len(xs), 0.5
    return 0.5, 0.5


def build_scene_spec(scene: dict, audio, cast_dirs: dict[str, Path], plate: Path,
                     world: tuple[int, int], frame: tuple[int, int], fps: int,
                     final: bool = True, species: dict[str, str] | None = None) -> SceneSpec:
    staging = scene["staging"] or {}
    with Image.open(plate) as im:
        grade_rgb, grade_strength = light_from_plate(im.copy())

    speaker = None
    for cue in audio.timing.cues:
        if cue.character_id:
            speaker = cue.character_id
            break

    chars: list[CharacterSpec] = []
    for cid, st in staging.items():
        rig = cast_dirs.get(cid)
        if rig is None:
            log.warning("staging_for_unknown_character", scene=scene["idx"], character=cid)
            continue
        # The writer's `scale` is a DRAMATIC choice (how close, how looming). Species
        # scale is a PHYSICAL fact. Multiplying them keeps both: a mouse pushed forward for
        # emphasis is still smaller than the elephant behind it.
        physical = relative_scale((species or {}).get(cid, "fox"))
        chars.append(CharacterSpec(
            rig_path=str(rig / "rig.json"),
            x=float(st.get("x", 0.5)), y=float(st.get("y", 0.93)),
            scale=float(st.get("scale", 1.0)) * physical,
            facing=st.get("facing", "right"),
            gesture=st.get("gesture", "idle"),
            speech=audio.timing.speech_spans(cid),
            blink_seed=(scene["idx"] * 101 + abs(hash(cid)) % 977),
            envelope=audio.timing.envelope_for(cid, fps), envelope_fps=fps,
            grade_rgb=grade_rgb, grade_strength=grade_strength))

    baked = [plate.parent / n for n in ("far.png", "mid.png", "near.png")]
    return SceneSpec(
        plate_path=str(plate),
        plate_layers=[str(p) for p in baked] if all(p.exists() for p in baked) else None,
        characters=chars, duration=audio.duration_s,
        world=world, frame=frame, fps=fps,
        camera_move=scene["camera_move"] or "static",
        shot_from=scene["camera_from"] or scene["shot"] or "wide",
        shot_to=scene["camera_to"],
        focus=_focus_for(scene, staging, speaker),
        easing="in_out", final=final)


def render_scenes(db: Path, scenes: list[dict], audio_by_scene: dict[int, object],
                  cast_dirs: dict[str, Path], plates: dict[str, Path], out_dir: Path,
                  world: tuple[int, int], frame: tuple[int, int], fps: int,
                  workers: int | None = None, crf: int = 20,
                  species: dict[str, str] | None = None) -> list[RenderedScene]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedScene] = []
    for sc in scenes:
        audio = audio_by_scene[sc["id"]]
        dest = out_dir / f"scene_{sc['idx']:03d}.mp4"
        # Resume: a scene whose file already exists and whose recorded hash still matches
        # is not re-rendered. At ~18 min per episode this is the difference between a
        # crash costing seconds and costing the whole run.
        if dest.exists() and sc.get("render_sha256") and _sha(dest) == sc["render_sha256"]:
            log.info("scene_render_reused", scene=sc["idx"])
            rendered.append(RenderedScene(sc["id"], sc["idx"], dest, audio.duration_s,
                                          {"reused": True}))
            continue
        plate = plates.get(sc["location_id"])
        if plate is None:
            raise RenderError(f"scene {sc['idx']} has no plate for location "
                              f"{sc['location_id']!r}")
        spec = build_scene_spec(sc, audio, cast_dirs, plate, world, frame, fps,
                                species=species)
        stats = render_scene_parallel(spec, dest, workers=workers, crf=crf)
        log.info("scene_rendered", scene=sc["idx"], **{k: stats[k] for k in
                 ("frames", "seconds", "ms_per_frame", "realtime_factor")})
        with tx(db) as con:
            con.execute("UPDATE scenes SET render_path = ?, render_sha256 = ?, "
                        "status = 'rendered' WHERE id = ?",
                        (str(dest), _sha(dest), sc["id"]))
        rendered.append(RenderedScene(sc["id"], sc["idx"], dest, audio.duration_s, stats))
    return rendered


def concat(scene_paths: list[Path], out_path: Path) -> Path:
    """Stream-copy concat. Every input came from the same encoder settings, so the
    timestamps line up and no re-encode is needed."""
    if not scene_paths:
        raise RenderError("nothing to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    listing = out_path.with_suffix(".concat.txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in scene_paths))
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out_path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"concat failed: {r.stderr[-600:]}")
    listing.unlink(missing_ok=True)
    return out_path


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"ffprobe failed on {path}: {r.stderr[-300:]}")
    return float(json.loads(r.stdout)["format"]["duration"])


def probe_stream(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
         str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"ffprobe failed on {path}: {r.stderr[-300:]}")
    data = json.loads(r.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), {})
    return {
        "duration": float(data["format"]["duration"]),
        "bytes": int(data["format"]["size"]),
        "width": int(v.get("width", 0)), "height": int(v.get("height", 0)),
        "fps": eval_fraction(v.get("r_frame_rate", "0/1")),
        "vcodec": v.get("codec_name", ""), "acodec": a.get("codec_name", ""),
        "sample_rate": int(a.get("sample_rate", 0) or 0),
        "channels": int(a.get("channels", 0) or 0),
        "video_start": float(v.get("start_time", 0) or 0),
        "audio_start": float(a.get("start_time", 0) or 0),
    }


def eval_fraction(text: str) -> float:
    try:
        num, den = text.split("/")
        return float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0
