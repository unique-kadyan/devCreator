"""TTS provider protocol. Everything downstream consumes Utterance, not a provider."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Utterance:
    """One synthesised line. `path` is authoritative for timing - never trust an estimate."""
    text: str
    path: Path
    duration_s: float
    sample_rate: int
    character_id: str | None      # None = narrator
    voice_id: str
    provider: str
    text_sha256: str
    envelope: list[float] = field(default_factory=list)   # per-frame RMS, 0..1
    envelope_fps: int = 24

    @staticmethod
    def hash_text(text: str, voice_id: str, speed: float, pitch: float) -> str:
        key = f"{voice_id}|{speed:.3f}|{pitch:.3f}|{text.strip()}"
        return hashlib.sha256(key.encode()).hexdigest()


class TTSProvider(Protocol):
    name: str
    sample_rate: int

    def voices(self) -> list[str]: ...

    def synthesize(self, text: str, voice_id: str, out_path: Path,
                   speed: float = 1.0, pitch_semitones: float = 0.0) -> Utterance: ...
