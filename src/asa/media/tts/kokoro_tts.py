"""Kokoro-82M TTS. Apache-2.0, runs locally on CPU, no API and no quota.

Measured on this machine (i7-8550U, no GPU): ~1.2x realtime, ~24s pipeline init.
The pipeline is therefore a process-level singleton - re-initialising per line would
dominate the cost of a whole episode.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import Utterance
from .voices import shift_pitch

SAMPLE_RATE = 24_000
_PIPELINES: dict[str, object] = {}


def _pipeline(lang_code: str = "a"):
    if lang_code not in _PIPELINES:
        warnings.filterwarnings("ignore")
        from kokoro import KPipeline
        _PIPELINES[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _PIPELINES[lang_code]


class KokoroTTS:
    name = "kokoro_local"
    sample_rate = SAMPLE_RATE

    def __init__(self, lang_code: str = "a"):
        self.lang_code = lang_code

    def voices(self) -> list[str]:
        """Kokoro voice ids are <lang><gender>_<name>; see the model card for the full set."""
        return [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_alloy",
            "af_aoede", "af_jessica", "af_kore", "af_nova", "af_river",
            "am_michael", "am_adam", "am_echo", "am_eric", "am_fenrir",
            "am_liam", "am_onyx", "am_puck", "am_santa",
            "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
            "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
        ]

    def synthesize(self, text: str, voice_id: str, out_path: Path,
                   speed: float = 1.0, pitch_semitones: float = 0.0,
                   character_id: str | None = None, cache_dir: Path | None = None) -> Utterance:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty text")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Content-addressed cache: editing one line re-synthesises one clip, not the episode.
        digest = Utterance.hash_text(text, voice_id, speed, pitch_semitones)
        cached = (cache_dir / f"{digest}.wav") if cache_dir else None
        if cached and cached.exists():
            wav, _ = sf.read(cached, dtype="float32")
            out_path.write_bytes(cached.read_bytes())
            return Utterance(
                text=text, path=out_path, duration_s=len(wav) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE, character_id=character_id, voice_id=voice_id,
                provider=self.name + "+cache", text_sha256=digest)

        chunks: list[np.ndarray] = []
        for _gs, _ps, audio in _pipeline(self.lang_code)(text, voice=voice_id, speed=speed):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        if not chunks:
            raise RuntimeError(f"kokoro produced no audio for: {text[:60]!r}")
        wav = np.concatenate(chunks)

        if abs(pitch_semitones) >= 0.01:
            raw = out_path.with_suffix(".raw.wav")
            sf.write(raw, wav, SAMPLE_RATE)
            shift_pitch(raw, out_path, pitch_semitones, SAMPLE_RATE)
            raw.unlink(missing_ok=True)
            wav, _ = sf.read(out_path, dtype="float32")
        else:
            sf.write(out_path, wav, SAMPLE_RATE)

        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(out_path.read_bytes())

        return Utterance(
            text=text, path=out_path, duration_s=len(wav) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE, character_id=character_id, voice_id=voice_id,
            provider=self.name, text_sha256=digest,
        )
