"""Sarvam AI text-to-speech: natural Hindi from a hosted model.

Kokoro-82M is 82 million parameters and Hindi is one of its weaker languages - it is
intelligible but audibly synthetic, which is why this exists. Sarvam's bulbul is trained on
Indian languages specifically and carries the prosody a small multilingual model cannot.

The trade is deliberate and the chain reflects it: this is a network call per line, so it
is slower and spends an allowance, and Kokoro stays underneath as the provider that always
works. A hosted voice that runs out mid-episode must degrade to a local one, not stop the
render.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.logging import get_logger
from .base import Utterance

log = get_logger("sarvam_tts")

ENDPOINT = "https://api.sarvam.ai/text-to-speech"
SAMPLE_RATE = 22_050

# bulbul:v2 speakers. Grouped by gender so casting can honour a character's presentation
# the same way the Kokoro pool does.
FEMALE = ["anushka", "manisha", "vidya", "arya"]
MALE = ["abhilash", "karun", "hitesh"]
SPEAKERS = FEMALE + MALE

LANG_CODES = {"hi": "hi-IN", "en": "en-IN", "bn": "bn-IN", "ta": "ta-IN",
              "te": "te-IN", "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN",
              "ml": "ml-IN", "pa": "pa-IN", "od": "od-IN"}


class SarvamTTS:
    name = "sarvam"
    sample_rate = SAMPLE_RATE

    def __init__(self, api_key: str, language: str = "hi",
                 model: str = "bulbul:v2", timeout: float = 60.0):
        self.api_key = (api_key or "").strip()
        self.language = LANG_CODES.get((language or "hi").lower(), "hi-IN")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def voices(self) -> list[str]:
        return list(SPEAKERS)

    def _post(self, text: str, speaker: str, speed: float, pitch: float) -> bytes:
        payload = {
            "text": text,
            "target_language_code": self.language,
            "speaker": speaker,
            "model": self.model,
            # The pipeline expresses pacing as a multiplier and pitch in semitones, which
            # is what Kokoro takes. Sarvam's `pace` is the same idea; `pitch` is a small
            # unitless nudge, so semitones are scaled down rather than passed through -
            # sending 3.0 here does not mean three semitones, it means a cartoon.
            "pace": round(min(1.5, max(0.5, speed)), 2),
            "pitch": round(min(1.0, max(-1.0, pitch / 6.0)), 2),
            "speech_sample_rate": SAMPLE_RATE,
            "enable_preprocessing": True,
        }
        try:
            r = httpx.post(ENDPOINT, json=payload, timeout=self.timeout,
                           headers={"api-subscription-key": self.api_key,
                                    "Content-Type": "application/json"})
        except httpx.TimeoutException as e:
            raise ProviderError(f"sarvam timed out after {self.timeout}s",
                                provider=self.name) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"sarvam transport: {e}", provider=self.name) from e

        if r.status_code in (401, 403):
            raise AuthError(f"sarvam rejected the key ({r.status_code})")
        if r.status_code == 429:
            raise RateLimited("sarvam rate limited", provider=self.name)
        if r.status_code in (402, 413):
            raise QuotaExhausted("sarvam allowance spent", provider=self.name)
        if r.status_code != 200:
            raise ProviderError(f"sarvam HTTP {r.status_code}: {r.text[:160]}",
                                provider=self.name)

        body = r.json()
        audios = body.get("audios") or []
        if not audios:
            raise ProviderError(f"sarvam returned no audio: {str(body)[:160]}",
                                provider=self.name)
        return base64.b64decode(audios[0])

    def synthesize(self, text: str, voice_id: str, out_path: Path,
                   speed: float = 1.0, pitch_semitones: float = 0.0,
                   character_id: str | None = None,
                   cache_dir: Path | None = None) -> Utterance:
        text = " ".join(text.split())
        if not text:
            raise ValueError("empty text")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        speaker = voice_id if voice_id in SPEAKERS else SPEAKERS[0]

        # Content-addressed cache, same contract as Kokoro: editing one line re-synthesises
        # one clip. It matters more here - every miss is a paid network round trip.
        digest = Utterance.hash_text(text, speaker, speed, pitch_semitones)
        cached = (cache_dir / f"{digest}.wav") if cache_dir else None
        if cached and cached.exists():
            wav, sr = sf.read(cached, dtype="float32")
            out_path.write_bytes(cached.read_bytes())
            return Utterance(text=text, path=out_path, duration_s=len(wav) / sr,
                             sample_rate=sr, character_id=character_id,
                             voice_id=speaker, provider=self.name + "+cache",
                             text_sha256=digest)

        raw = self._post(text, speaker, speed, pitch_semitones)
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:                       # mono downstream; the mixer assumes it
            wav = wav.mean(axis=1)
        sf.write(out_path, wav, sr)
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(out_path.read_bytes())

        log.info("sarvam_ok", speaker=speaker, chars=len(text),
                 duration_s=round(len(wav) / sr, 2))
        # `envelope` is deliberately left empty, exactly as the Kokoro provider does: the
        # RMS envelope is computed once downstream in context.load_audio, and a provider
        # that fills it in would be duplicating work the caller redoes anyway.
        return Utterance(text=text, path=out_path, duration_s=len(wav) / sr,
                         sample_rate=sr, character_id=character_id, voice_id=speaker,
                         provider=self.name, text_sha256=digest)
