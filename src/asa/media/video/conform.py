"""Force a clip returned by a hosted model into the shape the timeline promised.

A hosted video model answers with whatever it feels like: 1280x720 at 25fps for 5.0s when
the shot is 1920x1080 at 24fps for 4.67s. The rest of the pipeline cannot absorb that.
Scene clips are concatenated with a stream copy and the mixdown is laid against the scene
duration measured from the audio, so a clip that is half a frame long walks every later
scene out of sync - and nothing downstream notices, because each individual file is valid.

So conforming is not a nicety, it is the contract in `base.ShotJob`: exactly `frames`
frames, at `fps`, at `size`, encoded identically to the local renderer.

Two ways to hit an exact length, and the difference matters:

  preserve_timing=True   trim or clone-pad. Used when the clip is lip-synced to a specific
                         wav - stretching it slides the mouth off the voice, which is the
                         one defect the hosted provider exists to fix.
  preserve_timing=False  retime with setpts. Used for image-to-video clips, where nothing
                         in the picture is synchronised to anything and a model that only
                         emits 5-second clips would otherwise dictate the edit.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ...core.errors import RenderError
from ...core.logging import get_logger

log = get_logger("video_conform")

# How far a clip may drift from the requested duration before it is retimed rather than
# padded. A model that returns 4.9s for a 5.0s ask is fine to pad; one that returns 5.0s
# for a 2.1s ask is not a rounding difference and cloning 70 frames would freeze the shot.
RETIME_TOLERANCE = 0.04


def probe(path: Path) -> dict:
    """Duration, size and frame rate of a clip, or a RenderError naming the file."""
    if shutil.which("ffprobe") is None:
        raise RenderError("ffprobe not found on PATH")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
         "-show_entries", "format=duration", "-count_packets", "-of", "json", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RenderError(f"ffprobe could not read {path}: {out.stderr.strip()[:200]}")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RenderError(f"{path} has no video stream")
    s = streams[0]
    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    try:
        rate = float(num) / float(den or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        rate = 0.0
    return {"width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
            "fps": rate, "packets": int(s.get("nb_read_packets") or 0),
            "duration_s": float((data.get("format") or {}).get("duration") or 0.0)}


def conform_clip(src: Path, dest: Path, *, frames: int, fps: int,
                 size: tuple[int, int], preserve_timing: bool = True,
                 crf: int = 20, preset: str = "veryfast") -> dict:
    """Re-encode `src` to exactly `frames` frames at `fps` and `size`.

    Aspect is handled by covering and cropping rather than letterboxing: a hosted model
    asked for 16:9 sometimes returns 4:3, and black bars appearing for one shot in the
    middle of a scene reads as a broken render, whereas a slightly tighter crop does not.
    """
    if shutil.which("ffmpeg") is None:
        raise RenderError("ffmpeg not found on PATH")
    if frames < 1:
        raise RenderError(f"cannot conform {src} to {frames} frames")
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    target_s = frames / float(fps)
    w, h = size

    chain = []
    stretched = False
    if not preserve_timing and info["duration_s"] > 0.01:
        ratio = target_s / info["duration_s"]
        if abs(ratio - 1.0) > RETIME_TOLERANCE:
            chain.append(f"setpts={ratio:.6f}*PTS")
            stretched = True
    chain += [
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
        f"fps={fps}",
        # Clone the last frame rather than ending short. `-frames:v` below then cuts back to
        # the exact count, so this only ever fills a shortfall.
        "tpad=stop_mode=clone:stop_duration=2",
    ]

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vf", ",".join(chain), "-frames:v", str(frames),
           "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-g", str(fps * 2), "-movflags", "+faststart",
           "-r", str(fps), str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RenderError(f"conforming {src} failed: {r.stderr.strip()[:300]}")

    got = probe(dest)
    if got["packets"] and got["packets"] != frames:
        # Loud, because this is the failure that desynchronises an episode silently.
        log.warning("conform_frame_mismatch", path=str(dest), wanted=frames,
                    got=got["packets"])
    log.info("clip_conformed", src_size=[info["width"], info["height"]],
             src_fps=round(info["fps"], 2), src_s=round(info["duration_s"], 2),
             frames=frames, fps=fps, retimed=stretched)
    return {"frames": frames, "fps": fps, "retimed": stretched,
            "source": {"width": info["width"], "height": info["height"],
                       "fps": round(info["fps"], 3),
                       "duration_s": round(info["duration_s"], 3)},
            "path": str(dest), "bytes": dest.stat().st_size}
