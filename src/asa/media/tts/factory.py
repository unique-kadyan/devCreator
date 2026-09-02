"""Assemble the TTS chain from config.

`providers.tts.chain` was declared in config from the beginning and read by nothing:
`ctx.tts` constructed Kokoro directly, so the piper fallback listed beside it had never
once been reachable. This module makes the setting real, which matters more now that a
HOSTED voice leads the chain - a network provider that runs out of allowance mid-episode
must fall through to the local one rather than stop the render.
"""
from __future__ import annotations

from pathlib import Path

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.logging import get_logger
from .base import Utterance

log = get_logger("tts_factory")


class TTSChain:
    """Try each provider in order; the first that synthesises wins.

    Deliberately NOT a quota tracker. Whether a provider has allowance left is its own
    business - the chain only cares that a line came back as audio, because a missing line
    is a silent gap in a finished episode.
    """

    name = "tts_chain"

    def __init__(self, providers: list):
        self.providers = [p for p in providers if p is not None]
        if not self.providers:
            raise RuntimeError("no TTS provider is usable")
        self.sample_rate = self.providers[0].sample_rate

    def voices(self) -> list[str]:
        return self.providers[0].voices()

    def synthesize(self, text: str, voice_id: str, out_path: Path, **kw) -> Utterance:
        failures: list[str] = []
        for i, p in enumerate(self.providers):
            if not getattr(p, "available", True):
                failures.append(f"{p.name}: unavailable")
                continue
            try:
                return p.synthesize(text, voice_id, out_path, **kw)
            except (AuthError, QuotaExhausted, RateLimited, ProviderError) as e:
                failures.append(f"{p.name}: {str(e)[:80]}")
                log.warning("tts_provider_failed", provider=p.name, error=str(e)[:140])
            except Exception as e:                                   # noqa: BLE001
                failures.append(f"{p.name}: {type(e).__name__}")
                log.warning("tts_provider_error", provider=p.name, error=str(e)[:140])
        raise ProviderError("every TTS provider failed: " + " | ".join(failures),
                            provider="tts")


def build_tts_chain(cfg, language: str = "en"):
    from ...characters.factory import KOKORO_LANG_CODES
    order = cfg.get("providers.tts.chain", ["kokoro_local"]) or ["kokoro_local"]
    providers, active, skipped = [], [], []

    for name in order:
        node = cfg.get(f"providers.tts.{name}", {}) or {}
        try:
            if name == "sarvam":
                from .sarvam_tts import SarvamTTS
                key = cfg.secret(node.get("api_key_env", "SARVAM_API_KEY"),
                                 required=False)
                p = SarvamTTS(key, language=language,
                              model=node.get("model", "bulbul:v2"))
                if not p.available:
                    skipped.append(f"{name}(no key)")
                    continue
            elif name == "kokoro_local":
                from .kokoro_tts import KokoroTTS
                p = KokoroTTS(lang_code=KOKORO_LANG_CODES.get(language, "a"))
            else:
                skipped.append(f"{name}(unknown)")
                continue
        except Exception as e:                                       # noqa: BLE001
            skipped.append(f"{name}({type(e).__name__})")
            continue
        providers.append(p)
        active.append(name)

    log.info("tts_chain_built", active=active, skipped=skipped, language=language)
    return TTSChain(providers)
