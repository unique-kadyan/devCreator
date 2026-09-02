"""Parallel frame rendering.

Frames are independent, so the compositor scales across cores. Each worker builds its own
SceneRenderer once (backgrounds are tens of MB - pickling them per frame would cost more
than the compositing). Frames come back in order via imap and stream into ffmpeg.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ...characters.rig import Rig
from .camera import Camera
from .compositor import (BackgroundLayer, CharacterInstance, SceneRenderer,
                          SpeechSpan)
from .parallax import PARALLAX_FACTORS, multiplane


@dataclass
class CharacterSpec:
    """Picklable description of a character placement."""
    rig_path: str
    x: float = 0.5
    y: float = 0.93
    scale: float = 1.0
    facing: str = "right"
    gesture: str = "idle"
    speech: list[tuple[float, float]] = field(default_factory=list)
    blink_seed: int = 0
    envelope: list[float] = field(default_factory=list)
    envelope_fps: int = 24
    grade_rgb: tuple[int, int, int] = (255, 255, 255)
    grade_strength: float = 0.0


@dataclass
class SceneSpec:
    """Everything a worker needs to rebuild the renderer. Paths, not images."""
    plate_path: str
    characters: list[CharacterSpec]
    duration: float
    # Pre-baked far/mid/near planes. Each worker otherwise recomputes three LANCZOS
    # resizes of a ~2.7k plate at startup - six workers paying that separately is several
    # seconds per scene for a result the art stage already wrote to disk.
    plate_layers: list[str] | None = None
    world: tuple[int, int] = (2688, 1512)
    frame: tuple[int, int] = (1920, 1080)
    fps: int = 24
    camera_move: str = "static"
    shot_from: str = "wide"
    shot_to: str | None = None
    focus: tuple[float, float] = (0.5, 0.5)
    easing: str = "in_out"
    final: bool = True


_RENDERER: SceneRenderer | None = None


def _init_worker(spec: SceneSpec) -> None:
    global _RENDERER
    # each worker is single-threaded; parallelism comes from processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if spec.plate_layers and all(Path(p).exists() for p in spec.plate_layers):
        bg = [BackgroundLayer(Image.open(p).convert("RGBA"), parallax=f)
              for p, f in zip(spec.plate_layers, PARALLAX_FACTORS)]
    else:
        bg = multiplane(Image.open(spec.plate_path), spec.world)
    chars = []
    for cs in spec.characters:
        rig_path = Path(cs.rig_path)
        chars.append(CharacterInstance(
            rig=Rig.load(rig_path), base_dir=rig_path.parent,
            x=cs.x, y=cs.y, scale=cs.scale, facing=cs.facing, gesture=cs.gesture,
            speech=[SpeechSpan(a, b) for a, b in cs.speech], blink_seed=cs.blink_seed,
            envelope=cs.envelope, envelope_fps=cs.envelope_fps,
            grade_rgb=cs.grade_rgb, grade_strength=cs.grade_strength,
        ))
    cam = Camera(spec.world, spec.frame, move=spec.camera_move, shot_from=spec.shot_from,
                 shot_to=spec.shot_to, focus=spec.focus, easing=spec.easing)
    _RENDERER = SceneRenderer(spec.world, spec.frame, bg, chars, cam,
                              spec.duration, spec.fps, final=spec.final)


def _render_one(i: int) -> bytes:
    assert _RENDERER is not None
    return _RENDERER.render_frame(i).convert("RGB").tobytes()


def render_scene_parallel(spec: SceneSpec, out_path: Path, workers: int | None = None,
                          crf: int = 20, preset: str = "veryfast",
                          chunksize: int = 4) -> dict:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(round(spec.duration * spec.fps)))
    workers = workers or max(1, (os.cpu_count() or 4) - 2)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{spec.frame[0]}x{spec.frame[1]}", "-r", str(spec.fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-g", str(spec.fps * 2), "-movflags", "+faststart",
        str(out_path),
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    ctx = mp.get_context("fork")
    try:
        with ctx.Pool(workers, initializer=_init_worker, initargs=(spec,)) as pool:
            for buf in pool.imap(_render_one, range(n), chunksize=chunksize):
                proc.stdin.write(buf)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")
    el = time.time() - t0
    return {"frames": n, "workers": workers, "seconds": round(el, 2),
            "ms_per_frame": round(el / n * 1000, 1),
            "realtime_factor": round(spec.duration / el, 2),
            "path": str(out_path), "bytes": out_path.stat().st_size}
