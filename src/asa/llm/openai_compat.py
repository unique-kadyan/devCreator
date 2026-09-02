"""Generic provider for any OpenAI-compatible /chat/completions endpoint.

Used to turn every legitimate free tier into buffer capacity behind one interface. Each
instance gets its own ModelRouter row-space (keyed by provider name), so health is learned
per service, not pooled.

Only services with a real, no-card free tier belong here. A time-limited trial or an expiring
credit grant is not "free" and must not be added.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import httpx

from ..core.errors import (AllProvidersExhausted, AuthError, ProviderError,
                           QuotaExhausted, RateLimited)
from ..core.logging import get_logger
from ..core.quota import Limits, QuotaTracker
from .base import Completion
from .router import ModelRouter

log = get_logger("openai_compat")


class OpenAICompatProvider:
    """Speaks OpenAI chat-completions to Groq, Gemini, Cloudflare, HF router, etc."""

    def __init__(self, name: str, base_url: str, api_key: str, models: list[str],
                 router: ModelRouter | None = None, quota: QuotaTracker | None = None,
                 rpm: int | None = None, rpd: int | None = None,
                 timeout: float = 180.0, extra_headers: dict | None = None,
                 auth_scheme: str = "Bearer"):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.models = [m for m in models if m]
        self.router = router
        self.quota = quota
        self.limits = Limits(rpm=rpm, rpd=rpd)
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"{auth_scheme} {self.api_key}".strip()
        self._headers.update(extra_headers or {})

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.models)

    def _candidates(self, role: str, structured: bool) -> list[str]:
        if self.router:
            ranked = self.router.rank(role, structured)
            ordered = [m for m in ranked if m in self.models]
            return ordered + [m for m in self.models if m not in ordered]
        return list(self.models)

    def complete(self, system: str, user: str, role: str = "story",
                 max_tokens: int = 4096, temperature: float = 0.9,
                 structured: bool = False) -> Completion:
        if not self.available:
            raise AuthError(f"{self.name}: no API key or no models configured")
        if self.quota:
            self.quota.check(self.name, self.limits)

        tried: list[str] = []
        for model_id in self._candidates(role, structured):
            t0 = time.time()
            try:
                c = self._call(model_id, system, user, max_tokens, temperature)
            except RateLimited:
                if self.router:
                    self.router.record_rate_limit(model_id, role)
                tried.append(f"{model_id}=429")
                continue
            except (AuthError, QuotaExhausted):
                raise
            except ProviderError as e:
                if self.router:
                    self.router.record_error(model_id, role, str(e))
                tried.append(f"{model_id}={str(e)[:40]}")
                continue
            if not c.text.strip():
                if self.router:
                    self.router.record_empty(model_id, role)
                tried.append(f"{model_id}=empty")
                continue
            if self.router:
                self.router.record_success(model_id, role, time.time() - t0,
                                           c.completion_tokens)
            if self.quota:
                self.quota.consume(self.name)
            log.info("model_ok", provider=self.name, model=model_id,
                     after_failures=len(tried))
            return c
        raise AllProvidersExhausted(f"{self.name}: all models failed: {'; '.join(tried[:6])}")

    def note_schema_failure(self, model_id: str, role: str) -> None:
        if self.router:
            self.router.record_schema_failure(model_id, role)

    def _call(self, model_id: str, system: str, user: str,
              max_tokens: int, temperature: float) -> Completion:
        payload = {"model": model_id,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "max_tokens": max_tokens, "temperature": temperature}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{self.base_url}/chat/completions",
                                headers=self._headers, json=payload)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{model_id} timed out", provider=self.name) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{model_id} transport: {e}", provider=self.name) from e

        if r.status_code == 200:
            j = r.json()
            choice = (j.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            usage = j.get("usage") or {}
            return Completion(text=text, model_id=j.get("model", model_id),
                              provider=self.name,
                              prompt_tokens=usage.get("prompt_tokens", 0),
                              completion_tokens=usage.get("completion_tokens", 0))
        if r.status_code in (401, 403):
            raise AuthError(f"{self.name} rejected the key ({r.status_code})")
        if r.status_code == 429:
            raise RateLimited(f"{model_id} rate limited", provider=self.name)
        if r.status_code in (402, 413):
            raise QuotaExhausted(f"{self.name} quota reached", provider=self.name,
                                 resets_at=datetime.now(timezone.utc) + timedelta(hours=6))
        raise ProviderError(f"{model_id} HTTP {r.status_code}: {r.text[:140]}",
                            provider=self.name)
