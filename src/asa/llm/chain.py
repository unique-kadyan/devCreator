"""Provider chain with a terminal local fallback.

Order comes from config: openrouter_free -> huggingface -> local_llamacpp. Within OpenRouter
the ModelRouter already walks every free model, so reaching the next provider means
OpenRouter as a whole is unusable, not that one model was busy.
"""
from __future__ import annotations

from ..core.errors import AllProvidersExhausted, AuthError, ProviderError, QuotaExhausted
from ..core.logging import get_logger
from .base import Completion

log = get_logger("llm_chain")


class LLMChain:
    def __init__(self, providers: list):
        self.providers = [p for p in providers if p is not None]
        if not self.providers:
            raise ValueError("LLMChain needs at least one provider")

    def complete(self, system: str, user: str, role: str = "story",
                 max_tokens: int = 4096, temperature: float = 0.9,
                 structured: bool = False) -> Completion:
        failures: list[str] = []
        for provider in self.providers:
            try:
                return provider.complete(system, user, role=role, max_tokens=max_tokens,
                                         temperature=temperature, structured=structured)
            except (QuotaExhausted, AllProvidersExhausted) as e:
                failures.append(f"{provider.name}: {e}")
                log.warning("provider_exhausted", provider=provider.name, error=str(e)[:140])
                continue
            except AuthError as e:
                failures.append(f"{provider.name}: auth - {e}")
                log.warning("provider_auth_failed", provider=provider.name)
                continue
            except ProviderError as e:
                failures.append(f"{provider.name}: {e}")
                log.warning("provider_failed", provider=provider.name, error=str(e)[:140])
                continue
        raise AllProvidersExhausted("every LLM provider failed: " + " | ".join(failures))

    def note_schema_failure(self, completion: Completion, role: str) -> None:
        for p in self.providers:
            if getattr(p, "name", None) == completion.provider and hasattr(
                    p, "note_schema_failure"):
                p.note_schema_failure(completion.model_id, role)
                return
