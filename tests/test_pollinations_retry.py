"""One transient 5xx must not cost an episode.

Measured on job 10: 62 of 63 stills generated cleanly and a single `pollinations HTTP 500`
on scene 33 dropped that one frame to the procedural placeholder. QC then failed the whole
video for it - correctly - after art, audio, rendering and assembly had all finished. The
cheapest possible fix is asking again.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.core.errors import ProviderError, RateLimited      # noqa: E402
from asa.media.images import pollinations as P              # noqa: E402


class Resp:
    def __init__(self, code, content=b""):
        self.status_code, self.content, self.headers = code, content, {}


def png_bytes() -> bytes:
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def provider(monkeypatch, codes):
    """A Pollinations provider whose HTTP layer returns `codes` in order."""
    seen = {"n": 0}

    def fake_get(url, **kw):
        i = seen["n"]
        seen["n"] += 1
        code = codes[min(i, len(codes) - 1)]
        return Resp(code, png_bytes() if code < 400 else b"")

    monkeypatch.setattr(P.httpx, "get", fake_get)
    monkeypatch.setattr(P, "IMAGE_RETRY",
                        P.RetryPolicy(attempts=3, base_delay_s=0, max_delay_s=0, jitter=0))
    return P.PollinationsImages(enabled=True), seen


def test_a_transient_500_is_retried_and_succeeds(monkeypatch, tmp_path):
    p, seen = provider(monkeypatch, [500, 200])
    got = p.generate("a fox", tmp_path / "o.png", (64, 36))
    assert got.provider == "pollinations"
    assert seen["n"] == 2, "the 500 should have been asked again, not surrendered to"
    assert (tmp_path / "o.png").exists()


def test_a_persistent_500_still_gives_up_so_the_chain_can_move_on(monkeypatch, tmp_path):
    p, seen = provider(monkeypatch, [500])
    with pytest.raises(ProviderError):
        p.generate("a fox", tmp_path / "o.png", (64, 36))
    assert seen["n"] == 3, "three attempts, then hand back to the chain"


def test_a_400_is_not_retried_because_asking_again_cannot_help(monkeypatch, tmp_path):
    """A bad prompt or bad params fails identically every time. Retrying it just spends the
    shot's wall clock to be told the same thing three times."""
    p, seen = provider(monkeypatch, [400])
    with pytest.raises(ProviderError):
        p.generate("a fox", tmp_path / "o.png", (64, 36))
    assert seen["n"] == 1


def test_a_429_is_a_rate_limit_not_a_failure(monkeypatch, tmp_path):
    """It must surface as RateLimited so the chain waits rather than latching the provider
    off for the run - see the image quota latch, which deliberately ignores rate limits."""
    p, _ = provider(monkeypatch, [429])
    with pytest.raises(RateLimited):
        p.generate("a fox", tmp_path / "o.png", (64, 36))


def test_a_disabled_provider_still_refuses_before_any_network(monkeypatch, tmp_path):
    """The licence gate must sit in front of the retry, not behind it."""
    p, seen = provider(monkeypatch, [200])
    p.enabled = False
    with pytest.raises(ProviderError) as e:
        p.generate("a fox", tmp_path / "o.png", (64, 36))
    assert "disabled" in str(e.value)
    assert seen["n"] == 0
