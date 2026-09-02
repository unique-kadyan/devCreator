"""Exception hierarchy. Stages catch these to decide retry vs fall through vs fail.

The distinction that matters: QuotaExhausted and RateLimited both mean "not now", but
RateLimited means retry shortly on the SAME provider, while QuotaExhausted means move to
the next provider in the chain and do not come back until the window resets.
"""
from __future__ import annotations

from datetime import datetime


class ASAError(Exception):
    """Base for everything this project raises deliberately."""
    kind = "unknown"


class ConfigError(ASAError):
    kind = "config"


class ProviderError(ASAError):
    """A provider failed in a way that may or may not be transient."""
    kind = "provider"

    def __init__(self, message: str, provider: str = "", retryable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class RateLimited(ProviderError):
    """HTTP 429 from a shared pool. Retry soon, same provider, short backoff."""
    kind = "rate_limit"

    def __init__(self, message: str, provider: str = "", retry_after_s: float = 5.0):
        super().__init__(message, provider, retryable=True)
        self.retry_after_s = retry_after_s


class QuotaExhausted(ProviderError):
    """Your allowance is spent. Advance the chain; do not retry until `resets_at`."""
    kind = "quota"

    def __init__(self, message: str, provider: str = "", resets_at: datetime | None = None):
        super().__init__(message, provider, retryable=False)
        self.resets_at = resets_at


class AllProvidersExhausted(ASAError):
    """Every provider in a chain failed, including the terminal fallback."""
    kind = "quota"


class ValidationError(ASAError):
    """Model output did not satisfy its schema after the repair attempt."""
    kind = "validation"


class RenderError(ASAError):
    kind = "render"


class PolicyViolation(ASAError):
    """QC refused to let this through. Never retried automatically."""
    kind = "policy"


class AuthError(ASAError):
    """Credentials missing, invalid or revoked. Never retried blindly."""
    kind = "auth"
