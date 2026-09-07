"""The local renderer as a shot-video provider: no network, no credits, no GPU.

It is last in the chain and always available, for the same reason the procedural image
provider is: a chain that can run out is a chain that fails a render at 3am with nothing to
fall back on.
"""
from __future__ import annotations

from pathlib import Path

from ...core.logging import get_logger
from ..animation.performance import ShotSpec, render_shot
from .base import RenderedShot, ShotJob

log = get_logger("video_local")


class LocalPerformance:
    name = "local"

    def __init__(self, workers: int | None = None, crf: int = 20,
                 lipsync: bool = True, blink: bool = True,
                 min_face_confidence: float = 0.5):
        self.workers = workers
        self.crf = crf
        self.lipsync = lipsync
        self.blink = blink
        self.min_face_confidence = min_face_confidence

    @property
    def available(self) -> bool:
        return True

    def render(self, job: ShotJob, dest: Path) -> RenderedShot:
        spec = ShotSpec(
            image_path=str(job.image_path), frames=job.frames, fps=job.fps,
            size=job.size, camera_move=job.camera_move, face=job.face,
            envelope=list(job.envelope) if self.lipsync else [],
            animate_face=job.face is not None, blink=self.blink, seed=job.seed,
            face_confidence=job.face_confidence,
            min_face_confidence=self.min_face_confidence)
        stats = render_shot(spec, dest, workers=self.workers, crf=self.crf)
        return RenderedShot(path=dest, provider=self.name, model_id="2.5d-performance",
                            stats=stats)
