"""Amplitude envelope for lip sync (docs/02 §8.1, Tier A).

Windowed RMS at the video frame rate, gated and normalised. Cheap, deterministic, and at
24fps on a stylised character it reads as convincing speech.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf


def rms_envelope(path, fps: int = 24, gate_db: float = -42.0,
                 attack: float = 0.55, release: float = 0.30) -> list[float]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    hop = max(1, int(round(sr / fps)))
    n = int(np.ceil(len(wav) / hop))
    pad = np.pad(wav, (0, n * hop - len(wav)))
    frames = pad.reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)

    db = 20 * np.log10(rms + 1e-12)
    db = np.where(db < gate_db, gate_db, db)
    lo, hi = gate_db, max(gate_db + 6.0, float(db.max()))
    lvl = (db - lo) / (hi - lo)

    # asymmetric smoothing: mouths open fast and close slower, like real jaws
    out = np.zeros_like(lvl)
    prev = 0.0
    for i, v in enumerate(lvl):
        coef = attack if v > prev else release
        prev = prev + (v - prev) * coef
        out[i] = prev
    return [float(min(1.0, max(0.0, v))) for v in out]


def envelope_to_viseme(level: float, prev: str = "rest") -> str:
    """Map openness to a mouth shape. Thresholds are tuned to read at 24fps, not to be
    phonetically correct - true phoneme mapping is the forced-alignment path."""
    if level < 0.12:
        return "rest"
    if level < 0.28:
        return "M" if prev in ("rest", "M") else "I"
    if level < 0.45:
        return "I"
    if level < 0.62:
        return "E"
    if level < 0.78:
        return "U" if prev in ("O", "U") else "O"
    return "A"
