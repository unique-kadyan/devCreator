"""What it takes to turn one shot into one video clip, whoever renders it.

The shot layer (`assemble/shotlist.py`) decides WHAT each clip shows and how long it runs.
This protocol is the seam under it, so the same cut episode can be rendered by the local
2.5D animator today and by an image-to-video model later without re-cutting anything.

A provider is handed a still, the speech that plays over it, and the timing - and returns
an mp4 of exactly `frames` frames at `fps`. That frame count is a hard contract, not a
hint: clips are stream-copy concatenated and the audio is laid against the scene duration,
so a provider that returns a clip half a frame long walks the whole episode out of sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ShotJob:
    """One shot, everything any renderer might need to draw it."""

    image_path: Path
    frames: int
    fps: int
    size: tuple[int, int]
    camera_move: str = "static"
    # Normalised head box, when the framing has an animatable face.
    face: tuple[float, float, float, float] | None = None
    # How far that box can be trusted, 0..1. It is a prompt-derived guess unless a detector
    # wrote a sidecar, and a renderer that treats a guess as a measurement moves scenery.
    face_confidence: float = 0.35
    # Per-frame speech level of the SPEAKER of this shot, one value per frame of the clip.
    envelope: list[float] = field(default_factory=list)
    # The voice itself, for providers that do their own lip-sync from audio, and the window
    # of it this shot covers. The window is not decoration: a long speech is covered by
    # several shots, and handing each of them the whole line would lip-sync every one of
    # them to the line's opening words.
    voice_path: Path | None = None
    voice_offset_s: float = 0.0
    voice_duration_s: float = 0.0
    # Plain-language description of the action, for providers that take a motion prompt.
    motion_prompt: str = ""
    speaker: str | None = None
    seed: int = 0

    @property
    def duration_s(self) -> float:
        return self.frames / max(1, self.fps)

    @property
    def speaking(self) -> bool:
        return bool(self.envelope) and self.face is not None


@dataclass
class RenderedShot:
    path: Path
    provider: str
    model_id: str = ""
    stats: dict = field(default_factory=dict)


class ShotVideoProvider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def accepts(self, job: ShotJob) -> bool:
        """Whether this provider wants THIS shot, as opposed to being usable at all.

        `available` is about the account - a key, a reachable host. `accepts` is about the
        shot, and it exists because the interesting providers are metered. A hosted model
        earns its cost on a talking close-up and wastes it on a two-second insert of a
        letterbox, so it declines the insert and the chain falls through to the local
        renderer. Declining is routine, not an error.
        """
        return True

    def render(self, job: ShotJob, dest: Path) -> RenderedShot: ...
