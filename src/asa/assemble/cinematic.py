"""Render a scene as a moving camera over one generated frame.

The puppet path animates a rig: layered PNGs, lip-sync, blinks, staged characters. This
path has none of that, because the frame arrives from the image model already containing
the characters. What is left to animate is the CAMERA - a slow push, pull or drift across
an image rendered larger than the output - which is what carries a still photograph for
six seconds without reading as a slideshow.

The move is always slow and always in one direction. Anything faster looks like a zoom
artefact rather than a camera, and reversing direction inside a scene draws attention to
the fact that nothing in the picture is moving.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..core.errors import RenderError
from ..core.logging import get_logger

log = get_logger("cinematic")

# How far the camera travels over a scene, as a fraction of the frame. 8% reads as a
# deliberate push; past ~15% the softness of upscaling becomes visible on a still.
DEFAULT_TRAVEL = 0.08

# Which way the camera moves for each scripted camera_move. The story already chooses
# these, so the shot language survives even though nothing in the frame moves.
_MOVES = {
    "static":         (0.0, 0.0, 0.02),      # a barely-there drift stops it looking frozen
    "push_in":        (0.0, 0.0, DEFAULT_TRAVEL),
    "pull_out":       (0.0, 0.0, -DEFAULT_TRAVEL),
    "pan_left":       (-DEFAULT_TRAVEL, 0.0, 0.02),
    "pan_right":      (DEFAULT_TRAVEL, 0.0, 0.02),
    "tilt_up":        (0.0, -DEFAULT_TRAVEL, 0.02),
    "tilt_down":      (0.0, DEFAULT_TRAVEL, 0.02),
    "handheld_drift": (DEFAULT_TRAVEL * 0.5, DEFAULT_TRAVEL * 0.3, 0.03),
}


def move_for(camera_move: str) -> tuple[float, float, float]:
    """(x travel, y travel, zoom travel) as fractions of the frame."""
    return _MOVES.get((camera_move or "static").lower(), _MOVES["static"])


def _zoompan_filter(width: int, height: int, fps: int, duration_s: float,
                    camera_move: str) -> str:
    """ffmpeg zoompan expression for a linear camera move over `duration_s`.

    zoompan works in whole output frames, so the scene is upscaled first: zooming a
    1024-wide source directly to a 1920-wide output resamples every frame from too little
    data and the result crawls. Scaling up once, then moving inside that, keeps the image
    stable.
    """
    frames = max(1, int(round(duration_s * fps)))
    dx, dy, dz = move_for(camera_move)
    # Start zoomed in enough that the whole move stays inside the source.
    base = 1.0 + abs(dz) + max(abs(dx), abs(dy))
    z_expr = f"{base:.4f}{'+' if dz >= 0 else '-'}{abs(dz):.4f}*on/{frames}"
    # x/y are the top-left of the crop window, in source pixels.
    x_expr = (f"iw/2-(iw/zoom/2)+({dx:.4f}*iw*(on/{frames}-0.5))")
    y_expr = (f"ih/2-(ih/zoom/2)+({dy:.4f}*ih*(on/{frames}-0.5))")
    upscale = f"scale={width * 2}:{height * 2}:flags=lanczos"
    return (f"{upscale},zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            f":d={frames}:s={width}x{height}:fps={fps},format=yuv420p")


def render_still(image: Path, dest: Path, *, duration_s: float, fps: int,
                 size: tuple[int, int], camera_move: str = "static",
                 crf: int = 20) -> dict:
    """Render one scene image as a video clip with a moving camera."""
    if not image.exists():
        raise RenderError(f"no scene image at {image}")
    width, height = size
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = _zoompan_filter(width, height, fps, duration_s, camera_move)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
         "-t", f"{duration_s:.3f}", "-vf", vf, "-r", str(fps),
         "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
         "-pix_fmt", "yuv420p", str(dest)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists():
        raise RenderError(f"ffmpeg failed rendering {image.name}: "
                          f"{proc.stderr[-300:]}")
    return {"frames": int(round(duration_s * fps)), "camera_move": camera_move,
            "seconds": round(duration_s, 2)}
