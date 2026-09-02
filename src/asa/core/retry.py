"""Retry with exponential backoff and jitter, driven by the error taxonomy.

The rule: RateLimited backs off and retries the same provider; QuotaExhausted and AuthError
never retry (the caller advances the provider chain instead); everything else retries only if
it declared itself retryable.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from .logging import get_logger

T = TypeVar("T")
log = get_logger("retry")


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        raw = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


NETWORK = RetryPolicy(attempts=5, base_delay_s=2.0, max_delay_s=120.0)
LLM = RetryPolicy(attempts=3, base_delay_s=5.0, max_delay_s=60.0)
RENDER = RetryPolicy(attempts=2, base_delay_s=10.0, max_delay_s=60.0)


def with_retry(fn: Callable[[], T], policy: RetryPolicy = NETWORK,
               label: str = "", sleep: Callable[[float], None] = time.sleep) -> T:
    last: Exception | None = None
    for attempt in range(policy.attempts):
        try:
            return fn()
        except (QuotaExhausted, AuthError):
            raise                                   # never retry - advance the chain
        except RateLimited as e:
            last = e
            if attempt == policy.attempts - 1:
                break
            wait = max(e.retry_after_s, policy.delay_for(attempt))
            log.warning("rate_limited", label=label, attempt=attempt + 1, wait_s=round(wait, 1))
            sleep(wait)
        except ProviderError as e:
            last = e
            if not e.retryable or attempt == policy.attempts - 1:
                break
            wait = policy.delay_for(attempt)
            log.warning("provider_error", label=label, attempt=attempt + 1,
                        wait_s=round(wait, 1), error=str(e)[:120])
            sleep(wait)
        except Exception as e:                      # noqa: BLE001 - unknown, retry once class
            last = e
            if attempt == policy.attempts - 1:
                break
            sleep(policy.delay_for(attempt))
    assert last is not None
    raise last
