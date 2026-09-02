"""HTTP status classification for OpenAI-compatible providers.

The bug: 413 was folded in with 402 and raised QuotaExhausted with a six-hour cooldown.
Groq returns 413 for a TOKENS-PER-MINUTE ceiling - "Limit 8000, Requested 60122, please
reduce your message size and try again" - on an account with 14,400 requests/day still
available. One oversized scenes prompt therefore removed the fastest provider in the chain
for the rest of the day.
"""
import httpx
import pytest

from asa.core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from asa.llm.openai_compat import OpenAICompatProvider

_REAL_CLIENT = httpx.Client


@pytest.fixture
def responding(monkeypatch):
    """Make the provider's internally-created httpx.Client return a fixed status."""
    def factory(status: int, body: str = "{}"):
        def handler(request):
            return httpx.Response(status, text=body)

        def fake_client(*a, **kw):
            kw.pop("transport", None)
            return _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)

        monkeypatch.setattr(httpx, "Client", fake_client)
        return OpenAICompatProvider(name="test", base_url="https://example.invalid/v1",
                                    api_key="k", models=["m1", "m2"])
    return factory


@pytest.mark.parametrize("status,exc", [
    (413, RateLimited),        # per-minute token ceiling - retry, do not disable
    (402, QuotaExhausted),     # allowance actually spent
    (429, RateLimited),
    (401, AuthError),
    (403, AuthError),
    (500, ProviderError),
])
def test_status_maps_to_the_right_error(responding, status, exc):
    p = responding(status)
    with pytest.raises(exc):
        p._call("m1", "sys", "user", 100, 0.5)


def test_413_is_not_treated_as_exhaustion(responding):
    # RateLimited subclasses ProviderError, so the model loop advances to the next model.
    # QuotaExhausted would have parked the whole provider for six hours instead.
    p = responding(413, '{"error":{"message":"Limit 8000, Requested 60122"}}')
    with pytest.raises(RateLimited) as e:
        p._call("m1", "sys", "user", 100, 0.5)
    assert not isinstance(e.value, QuotaExhausted)
    assert isinstance(e.value, ProviderError), "must be catchable by the model loop"


def test_402_still_parks_the_provider(responding):
    p = responding(402)
    with pytest.raises(QuotaExhausted) as e:
        p._call("m1", "sys", "user", 100, 0.5)
    assert e.value.resets_at is not None, "a spent allowance needs a reset time"
