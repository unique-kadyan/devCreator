"""Render frames to an mp4 by piping raw RGB straight into ffmpeg.

No intermediate PNG files: writing 700+ PNGs and re-reading them costs more than the
compositing itself on a spinning-rust-era laptop.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .compositor import SceneRenderer


def render_scene(renderer: SceneRenderer, out_path: Path, crf: int = 20,
                 preset: str = "veryfast", progress_every: int = 48) -> dict:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(round(renderer.duration * renderer.fps)))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{renderer.frame_w}x{renderer.frame_h}",
        "-r", str(renderer.fps), "-i", "pipe:0",
        "-an",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-g", str(renderer.fps * 2),
        "-movflags", "+faststart",
        str(out_path),
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(n_frames):
            frame = renderer.render_frame(i).convert("RGB")
            proc.stdin.write(frame.tobytes())
            if progress_every and i and i % progress_every == 0:
                el = time.time() - t0
                print(f"    frame {i:4d}/{n_frames}  {el:5.1f}s  "
                      f"({i/el:4.1f} fps, {el/i*1000:5.1f} ms/frame)", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")
    el = time.time() - t0
    return {"frames": n_frames, "seconds": round(el, 2),
            "ms_per_frame": round(el / n_frames * 1000, 1),
            "realtime_factor": round(renderer.duration / el, 2),
            "path": str(out_path), "bytes": out_path.stat().st_size}
