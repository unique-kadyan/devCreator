"""Voice casting and deterministic per-character timbre offsets.

Kokoro ships a fixed set of preset voices. Pitch/rate offsets applied after synthesis widen
the usable cast well beyond that, and because they are stored on the character they are as
permanent as fur colour.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceCasting:
    voice_id: str
    speed: float = 1.0
    pitch_semitones: float = 0.0

    def key(self) -> str:
        return f"{self.voice_id}@{self.speed:.2f}/{self.pitch_semitones:+.2f}"


# Emotion is not a Kokoro parameter. It is expressed as pacing, which is what a narrator
# actually varies. Values multiply the character's base speed.
EMOTION_RATE = {
    "neutral": 1.00, "curious": 0.97, "happy": 1.06, "sad": 0.88,
    "scared": 1.10, "angry": 1.08, "surprised": 1.12, "determined": 1.02,
    "wry": 0.95, "excited": 1.14,
}


def rate_for(emotion: str | None) -> float:
    return EMOTION_RATE.get((emotion or "neutral").lower(), 1.0)


def shift_pitch(src: Path, dst: Path, semitones: float, sample_rate: int) -> None:
    """Pitch-shift without changing duration, via ffmpeg asetrate + atempo.

    atempo only accepts 0.5-2.0 per instance, so large shifts chain filters. Kept modest
    (+/- 4 semitones) because heavy shifting on 24 kHz speech sounds synthetic.
    """
    if abs(semitones) < 0.01:
        dst.write_bytes(src.read_bytes())
        return
    ratio = 2.0 ** (semitones / 12.0)
    tempo_chain, remaining = [], ratio
    while remaining > 2.0:
        tempo_chain.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        tempo_chain.append(0.5)
        remaining /= 0.5
    tempo_chain.append(remaining)
    filters = f"asetrate={int(sample_rate * ratio)}," + ",".join(
        f"atempo={t:.6f}" for t in tempo_chain) + f",aresample={sample_rate}"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af", filters, str(dst)],
        check=True)
