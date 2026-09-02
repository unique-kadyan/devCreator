"""OpenRouter provider, free tier only, driven by the adaptive ModelRouter.

A 429 from a `:free` model means the shared upstream pool is busy - not that you are over
your own quota - so the correct response is to advance to the next model immediately, mark
this one cold for a short window, and keep going. The call only fails after every free model
in the live catalogue has been tried.
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

log = get_logger("openrouter")
BASE = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    name = "openrouter_free"

    def __init__(self, api_key: str, router: ModelRouter,
                 quota: QuotaTracker | None = None, rpm: int = 20, rpd: int = 50,
                 app_name: str = "asa", site_url: str = "", timeout: float = 180.0,
                 max_models_per_call: int = 12):
        if not api_key:
            raise AuthError("OPENROUTER_API_KEY is empty")
        if not api_key.startswith("sk-or-"):
            raise AuthError("OPENROUTER_API_KEY does not look like an OpenRouter key "
                            "(expected 'sk-or-v1-...')")
        self.api_key = api_key
        self.router = router
        self.quota = quota
        self.limits = Limits(rpm=rpm, rpd=rpd)
        self.timeout = timeout
        self.max_models_per_call = max_models_per_call
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url or "http://localhost",
            "X-Title": app_name,
        }

    def complete(self, system: str, user: str, role: str = "story",
                 max_tokens: int = 4096, temperature: float = 0.9,
                 structured: bool = False, min_context: int = 0) -> Completion:
        if self.quota:
            self.quota.check(self.name, self.limits)

        candidates = self.router.rank(role, structured, min_context)[:self.max_models_per_call]
        if not candidates:
            raise AllProvidersExhausted("no free models available from OpenRouter")

        tried: list[str] = []
        for model_id in candidates:
            t0 = time.time()
            try:
                c = self._call(model_id, system, user, max_tokens, temperature)
            except RateLimited:
                self.router.record_rate_limit(model_id, role)
                tried.append(f"{model_id}=429")
                log.info("model_busy_advancing", model=model_id, tried=len(tried))
                continue
            except AuthError:
                raise
            except QuotaExhausted:
                raise
            except ProviderError as e:
                self.router.record_error(model_id, role, str(e))
                tried.append(f"{model_id}={str(e)[:40]}")
                log.info("model_failed_advancing", model=model_id, error=str(e)[:80])
                continue
            if not c.text.strip():
                self.router.record_empty(model_id, role)
                tried.append(f"{model_id}=empty")
                continue
            if c.meta.get("finish_reason") == "length":
                # It answered, but it ran out of room. parse_model salvages what it can;
                # the router should still learn that this model does not finish.
                self.router.record_schema_failure(model_id, role)
                log.info("model_truncated", model=model_id, tokens=c.completion_tokens)
            self.router.record_success(model_id, role, time.time() - t0,
                                       c.completion_tokens)
            if self.quota:
                self.quota.consume(self.name)
            log.info("model_ok", model=model_id, after_failures=len(tried),
                     latency_s=round(time.time() - t0, 1))
            return c

        raise AllProvidersExhausted(
            f"all {len(candidates)} free OpenRouter models failed: {'; '.join(tried[:8])}")

    def note_schema_failure(self, model_id: str, role: str) -> None:
        """Called by the generator when output parsed as HTTP 200 but failed validation."""
        self.router.record_schema_failure(model_id, role)

    def _call(self, model_id: str, system: str, user: str,
              max_tokens: int, temperature: float) -> Completion:
        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(f"{BASE}/chat/completions", headers=self._headers,
                                json=payload)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{model_id} timed out after {self.timeout}s",
                                provider=self.name) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{model_id} transport error: {e}", provider=self.name) from e

        if r.status_code == 200:
            j = r.json()
            if "error" in j and not j.get("choices"):
                raise ProviderError(f"{model_id}: {str(j['error'])[:120]}", provider=self.name)
            choice = (j.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            usage = j.get("usage") or {}
            return Completion(text=text, model_id=j.get("model", model_id),
                              provider=self.name,
                              prompt_tokens=usage.get("prompt_tokens", 0),
                              completion_tokens=usage.get("completion_tokens", 0),
                              meta={"finish_reason": choice.get("finish_reason")})
        if r.status_code == 401:
            raise AuthError(f"OpenRouter rejected the key (401): {r.text[:160]}")
        if r.status_code == 403:
            # 403 from OpenRouter is almost always about THIS MODEL - upstream moderation,
            # a data-policy the account has not enabled, a gated preview. Treating it as an
            # account failure aborts the whole chain and fails the job over one model
            # having a bad day, which is precisely what the router exists to prevent. Only
            # an explicit statement that the key itself is bad is an auth error.
            body = r.text[:400].lower()
            if any(t in body for t in ("invalid api key", "no auth credentials",
                                       "key disabled", "key has been disabled",
                                       "user not found", "account disabled")):
                raise AuthError(f"OpenRouter rejected the key (403): {r.text[:160]}")
            raise ProviderError(f"{model_id} refused (403): {r.text[:140]}",
                                provider=self.name)
        if r.status_code == 429:
            raise RateLimited(f"{model_id} rate limited upstream", provider=self.name)
        if r.status_code == 402:
            raise QuotaExhausted("OpenRouter credit / daily free cap reached",
                                 provider=self.name,
                                 resets_at=datetime.now(timezone.utc) + timedelta(hours=6))
        raise ProviderError(f"{model_id} HTTP {r.status_code}: {r.text[:140]}",
                            provider=self.name)
