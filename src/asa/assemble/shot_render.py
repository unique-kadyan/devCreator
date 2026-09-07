"""Turning a scene into shots on disk: the images the art stage buys, the clips it cuts.

This is the join between three pieces that deliberately know nothing about each other -
`assemble/shotlist.py` decides where the cuts fall, `media/images` buys the pictures, and
`media/video` renders the clips - and it is the only place that knows a shot's image lives
at a particular path.

That path is derived, not recorded. The art stage and the animate stage run in separate
processes (and after a crash, separate days), so a shot image is addressed by a hash of the
same `image_key` both stages compute from the same dialogue. A missing file then means
exactly one thing - the art stage has not run for this scene - and says so, rather than
silently rendering a different picture.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..core.errors import RenderError
from ..core.logging import get_logger
from ..media.animation.face import locate_face
from ..media.images.scene_image import negative_for, shot_prompt, styled_for
from ..media.video.base import ShotJob
from ..media.video.motion import shot_motion_prompt
from .shotlist import Shot, emotion_bucket, image_keys, plan_shots, time_shots
from .video import concat

log = get_logger("shots")


def shot_image_name(scene_idx: int, image_key: str) -> str:
    """Deterministic filename for one camera setup. Derived from the key, so both stages
    agree without a table to join on."""
    digest = hashlib.sha256(image_key.encode()).hexdigest()[:8]
    return f"scene_{scene_idx:03d}_{digest}.png"


def generate_shot_images(ctx, scene: dict, cast_by_id: dict, out_dir: Path, *,
                         max_images: int = 4, region_hint: str | None = None,
                         look_hint: str | None = None,
                         art_style: str | None = None) -> dict:
    """Buy every distinct camera setup this scene needs. Returns {image_key: path}.

    `art_style` names the preset (`channel.art_style`) and it is passed to BOTH halves of
    the prompt on purpose: the subject clause and the negative belong to the same preset,
    and mixing them - a cartoon subject under the photoreal negative, which ends with
    "cartoon, illustration, drawing" - is a prompt arguing with itself.
    """
    negative = negative_for(art_style)
    shots = plan_shots(scene, max_images=max_images)
    first_of: dict[str, Shot] = {}
    for s in shots:
        first_of.setdefault(s.image_key, s)

    made = reused = 0
    paths: dict[str, Path] = {}
    for key in image_keys(shots):
        shot = first_of[key]
        kwargs = {"region_hint": region_hint} if region_hint else {}
        # The frame shape comes from the size the pipeline will actually ask for, so a
        # portrait render does not ask the model for widescreen. `ctx.images.size` is the
        # single source of that - the same tuple `scene()` passes to the provider below.
        prompt = shot_prompt(scene, cast_by_id, speaker=shot.speaker,
                             framing=shot.framing,
                             emotion_bucket=emotion_bucket(shot.emotion),
                             speaking=shot.speaking,
                             style=styled_for(ctx.images.size, look_hint, art_style),
                             **kwargs)
        plate = ctx.images.scene(scene["idx"], prompt, out_dir, negative,
                                 name=shot_image_name(scene["idx"], key))
        paths[key] = Path(plate.path)
        made += int(not plate.cached)
        reused += int(plate.cached)
    return {"paths": paths, "shots": len(shots), "generated": made, "cached": reused}


def build_shot_jobs(scene: dict, timing, out_dir: Path, *, fps: int,
                    size: tuple[int, int], max_images: int = 4,
                    min_shot_s: float = 0.75, max_shot_s: float = 6.5,
                    cast_by_id: dict | None = None,
                    art_style: str | None = None,
                    ) -> list[tuple[Shot, ShotJob]]:
    """Plan the scene's shots against its measured audio and pair each with its render job.

    `cast_by_id` is the story's cast, and it is here for the motion prompt rather than for
    the picture: a hosted video model is told who is on screen using the SAME appearance
    sentence the image prompt used, because it has no more memory between calls than the
    image model did. Optional, so the local renderer - which reads none of this - still
    works from a scene alone.
    """
    shots = time_shots(plan_shots(scene, max_images=max_images), timing, fps,
                       min_shot_s=min_shot_s, max_shot_s=max_shot_s)
    jobs: list[tuple[Shot, ShotJob]] = []
    for shot in shots:
        image = out_dir / shot_image_name(scene["idx"], shot.image_key)
        if not image.exists():
            raise RenderError(
                f"scene {scene['idx']} shot {shot.index} has no image at {image}; "
                f"re-run the art stage")
        face = locate_face(image, shot.framing) if shot.animates_face else None
        voice, v_off, v_dur = _voice_window(timing, shot)
        jobs.append((shot, ShotJob(
            image_path=image, frames=shot.frames, fps=fps, size=size,
            camera_move=shot.camera_move,
            face=face.as_tuple() if face else None,
            face_confidence=face.confidence if face else 0.0,
            envelope=_slice_envelope(timing, shot, fps),
            voice_path=voice, voice_offset_s=v_off, voice_duration_s=v_dur,
            motion_prompt=shot_motion_prompt(scene, shot, cast_by_id,
                                             art_style=art_style),
            speaker=shot.speaker,
            seed=scene["idx"] * 31 + shot.index)))
    return jobs


def _slice_envelope(timing, shot: Shot, fps: int) -> list[float]:
    """The speaker's own speech level over this shot, one value per frame of the clip.

    Per SPEAKER, not per scene: an envelope built from the whole mix would open the mouth
    of whoever is on screen whenever anyone is talking, which is the mistake that makes
    two-hander scenes look dubbed.
    """
    if shot.speaker is None:
        return []
    env = timing.envelope_for(shot.speaker, fps)
    out = env[shot.start_frame:shot.end_frame]
    if len(out) < shot.frames:
        out = list(out) + [0.0] * (shot.frames - len(out))
    return list(out[:shot.frames])


def _voice_window(timing, shot: Shot) -> tuple[Path | None, float, float]:
    """The speaker's wav, and the slice of it this shot plays over.

    (path, offset into that file, duration). A shot is not always a whole line: a long
    speech is planned as several setups, and each covers a different stretch of the same
    utterance. Handing a hosted lip-sync model the whole file and trimming its output
    afterwards would sync every part of the speech to the line's opening words - the mouth
    would be moving, plausibly, and saying the wrong thing.
    """
    for i in shot.lines:
        if 0 <= i < len(timing.cues) and timing.cues[i].character_id == shot.speaker:
            cue = timing.cues[i]
            offset = max(0.0, shot.start_s - cue.start_s)
            available = max(0.0, cue.utterance.duration_s - offset)
            return cue.utterance.path, offset, min(shot.duration_s, available)
    return None, 0.0, 0.0


def render_scene_shots(chain, jobs: list[tuple[Shot, ShotJob]], scene_idx: int,
                       out_dir: Path, dest: Path) -> dict:
    """Render every shot of a scene and concatenate them into the scene's clip.

    The scene clip stays the unit the rest of the pipeline understands - `scenes.render_path`,
    the part planner and the audio bus are all unchanged - so cutting a scene into shots is
    invisible above this line.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    stats: list[dict] = []
    for shot, job in jobs:
        clip = out_dir / f"scene_{scene_idx:03d}_shot_{shot.index:02d}.mp4"
        result = chain.render(job, clip)
        clips.append(clip)
        stats.append(result.stats)
        log.info("shot_rendered", scene=scene_idx, shot=shot.index,
                 speaker=shot.speaker or "narration", framing=shot.framing,
                 move=shot.camera_move, frames=shot.frames,
                 provider=result.provider, seconds=result.stats.get("seconds"))
    if len(clips) == 1:
        clips[0].replace(dest)
    else:
        concat(clips, dest)
    return {"shots": len(clips),
            "seconds": round(sum(s.get("seconds", 0) for s in stats), 2),
            "frames": sum(s.get("frames", 0) for s in stats)}
