"""Final assembly: concat the scenes, build the audio bus, master, mux, record.

The audio bus is built against the *rendered* scene durations, not the planned ones, so a
scene whose speech ran long cannot drift its music and sfx out of alignment. Everything is
placed on one absolute timeline and mixed in a single ffmpeg graph - mixing incrementally
would apply loudness normalisation more than once, which is how a master ends up crushed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..core.db import tx
from ..core.errors import RenderError
from ..core.logging import get_logger
from ..media.audio.mixer import Placed, measure_loudness, mix, mux
from ..scenes.persist import load_scenes, load_story
from .video import concat, probe_stream

log = get_logger("mixdown")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble_episode(ctx, job: dict) -> dict:
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    audio = ctx.load_audio(job["id"], scenes, story["cast"])
    by_scene = {a.scene_id: a for a in audio.scenes}

    scene_files = []
    for s in scenes:
        p = Path(s["render_path"] or "")
        if not p.exists():
            raise RenderError(f"scene {s['idx']} has no rendered file at {p}")
        scene_files.append(p)

    silent = paths.work / "video_silent.mp4"
    concat(scene_files, silent)

    voice: list[Placed] = []
    music: list[Placed] = []
    sfx: list[Placed] = []
    clock = 0.0
    music_lufs = float(ctx.cfg.get("audio.music_lufs", -30.0))
    sfx_gain = float(ctx.cfg.get("audio.sfx_lufs", -22.0)) - float(
        ctx.cfg.get("audio.voice_lufs", -16.0))

    for s in scenes:
        sa = by_scene[s["id"]]
        for cue in sa.timing.cues:
            voice.append(Placed(path=cue.utterance.path, start_s=clock + cue.start_s))

        bed = _music_for(ctx, s, sa.duration_s, paths.work)
        if bed is not None:
            music.append(Placed(path=bed, start_s=clock,
                                gain_db=music_lufs - float(
                                    ctx.cfg.get("audio.voice_lufs", -16.0))))
        for i, tag in enumerate(s["sfx_tags"][:4]):
            clip = ctx.sfx.resolve(tag)
            if clip is None:
                continue
            # Space effects across the first half of the scene so they punctuate the action
            # instead of stacking on the first frame.
            sfx.append(Placed(path=clip,
                              start_s=clock + 0.4 + i * max(0.8, sa.duration_s / 8),
                              gain_db=sfx_gain))
        clock += sa.duration_s

    total = clock
    mixdown = paths.work / "mixdown.wav"
    mix(voice, music, sfx, duration_s=total, out_path=mixdown,
        target_lufs=float(ctx.cfg.get("audio.target_lufs", -14.0)),
        true_peak_db=float(ctx.cfg.get("audio.true_peak_db", -1.0)),
        duck_db=float(ctx.cfg.get("audio.duck_db", -12.0)))

    final = paths.out / "episode.mp4"
    mux(silent, mixdown, final)
    info = probe_stream(final)
    loud = measure_loudness(final)
    lufs, peak = loud["integrated_lufs"], loud["true_peak_dbfs"]

    with tx(ctx.db) as con:
        con.execute("""
            INSERT INTO videos (job_id, story_id, path, sha256, duration_s, width, height,
                                fps, bytes, lufs)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
                path=excluded.path, sha256=excluded.sha256,
                duration_s=excluded.duration_s, bytes=excluded.bytes, lufs=excluded.lufs
        """, (job["id"], job["story_id"], str(final), _sha(final), info["duration"],
              info["width"], info["height"], info["fps"], info["bytes"], lufs))
    log.info("episode_assembled", duration_s=round(info["duration"], 2),
             lufs=lufs, true_peak_db=peak, mb=round(info["bytes"] / 1e6, 1))
    return {"path": str(final), "duration_s": round(info["duration"], 2),
            "lufs": lufs, "true_peak_db": peak,
            "mb": round(info["bytes"] / 1e6, 1)}


def _music_for(ctx, scene: dict, duration_s: float, work: Path) -> Path | None:
    """A bed per scene, chosen by emotion. Returns None when the library is empty - which
    is a legitimate state, not an error: the episode simply has no score."""
    try:
        return ctx.music.bed_for(scene.get("emotion") or scene.get("music_cue"),
                                 duration_s,
                                 work / f"music_{scene['idx']:03d}.wav",
                                 seed=scene["idx"])
    except Exception as e:                                     # noqa: BLE001
        log.warning("music_bed_failed", scene=scene["idx"], error=str(e)[:140])
        return None
