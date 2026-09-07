"""An image provider that is out of credit must be asked once, not once per image.

Measured on a real run before this existed: HuggingFace answers 402 once its free stills are
gone, and every remaining image asked it again anyway. Each ask walks HF's whole configured
model list before giving up - about 45 seconds - so roughly twenty images consumed the art
stage's entire 900s budget and the stage FAILED with most of the episode ungenerated, while
a working provider sat next in the chain the whole time.

The bug is not that the fallback was missing. The fallback worked. The bug is that the cost
of rediscovering a permanent condition was paid per image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.core.errors import ProviderError, QuotaExhausted, RateLimited  # noqa: E402
from asa.media.images.base import GeneratedImage                        # noqa: E402
from asa.media.images import factory as image_factory                   # noqa: E402
from asa.media.images.factory import ImageChain                         # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """The chain waits out a 429 for real seconds. Correct in production, pointless in a
    test - without this the rate-limit cases spend 45s asleep proving nothing."""
    monkeypatch.setattr(image_factory.time, "sleep", lambda _s: None)


class Fake:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.calls = name, fail, 0
        self.available = True

    def generate(self, prompt, dest, size, negative="", seed=None):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"\x89PNG\r\n")
        return GeneratedImage(path=Path(dest), provider=self.name, model_id="m",
                              prompt_sha="sha")


def chain(providers, tmp_path):
    """An ImageChain with the DB and cache stubbed out - only routing is under test."""
    c = ImageChain.__new__(ImageChain)
    c.providers = providers
    c.db = tmp_path / "x.db"
    c.size = (64, 36)
    c._spent = {}
    return c


def test_a_spent_provider_is_asked_once_and_then_skipped(tmp_path):
    """The whole point. Twenty images must not cost twenty 402s."""
    broke = Fake("huggingface", QuotaExhausted("HF credits exhausted", provider="hf"))
    good = Fake("pollinations")
    c = chain([broke, good], tmp_path)
    for i in range(20):
        got = c._generate_with("p", tmp_path / f"{i}.png", None, "")
        assert got.provider == "pollinations"
    assert broke.calls == 1, f"asked the broke provider {broke.calls} times, wanted 1"
    assert good.calls == 20


def test_a_rate_limit_is_not_latched(tmp_path):
    """A rate limit clears in seconds and the provider is still the best one available.

    Latching it would give away the good provider for the whole episode - the opposite
    mistake, and the more expensive one, because it degrades output that would have been
    fine.
    """
    limited = Fake("huggingface", RateLimited("slow down", provider="hf"))
    good = Fake("pollinations")
    c = chain([limited, good], tmp_path)
    for i in range(3):
        c._generate_with("p", tmp_path / f"{i}.png", None, "")
    # Asked afresh for EVERY image - never struck off the list the way a 402 is.
    assert limited.calls == 3 * image_factory.RATE_LIMIT_TRIES
    assert "huggingface" not in c._spent, "a rate limit must not disable a provider"


def test_a_rate_limit_is_waited_out_before_falling_down_the_chain(tmp_path):
    """The provider below is a worse picture or a far slower one, so a 429 is worth sleeping
    on rather than stepping over.

    Measured on job 11: Replicate served one still, answered 429 on the next, and the chain
    stepped down to CPU diffusion at 10-17 MINUTES an image - having been ten seconds away
    from the picture it actually wanted. It recovers on the first retry every time.
    """
    calls = {"n": 0}

    class Flaky(Fake):
        def generate(self, prompt, dest, size, negative="", seed=None):
            calls["n"] += 1
            if calls["n"] == 1:                      # 429 once, then fine
                raise RateLimited("slow down", provider="replicate")
            self.fail = None
            return Fake.generate(self, prompt, dest, size, negative, seed)

    flaky = Flaky("replicate")
    slow = Fake("local")
    c = chain([flaky, slow], tmp_path)
    got = c._generate_with("p", tmp_path / "a.png", None, "")
    assert got.provider == "replicate", "waited out the 429 and got the good picture"
    assert slow.calls == 0, "must not have fallen through to the slow provider"


def test_a_rate_limit_still_gives_up_eventually(tmp_path):
    """Bounded. A provider answering 429 forever is saying something real, and the chain
    still has to reach the free floor rather than sleep out the whole stage."""
    limited = Fake("replicate", RateLimited("always", provider="replicate"))
    good = Fake("local")
    c = chain([limited, good], tmp_path)
    got = c._generate_with("p", tmp_path / "a.png", None, "")
    assert got.provider == "local"
    assert limited.calls == image_factory.RATE_LIMIT_TRIES


def test_the_latch_is_per_provider_not_global(tmp_path):
    """One exhausted account says nothing about a different service's allowance."""
    broke = Fake("huggingface", QuotaExhausted("spent", provider="hf"))
    good = Fake("pollinations")
    c = chain([broke, good], tmp_path)
    c._generate_with("p", tmp_path / "a.png", None, "")
    assert "huggingface" in c._spent
    assert "pollinations" not in c._spent
    assert c._live() == [good]


def test_everything_spent_still_raises_rather_than_going_quiet(tmp_path):
    """Skipping a spent provider must not turn into silently returning nothing."""
    broke = Fake("huggingface", QuotaExhausted("spent", provider="hf"))
    c = chain([broke], tmp_path)
    with pytest.raises(ProviderError):
        c._generate_with("p", tmp_path / "a.png", None, "")
    with pytest.raises(ProviderError) as e:
        c._generate_with("p", tmp_path / "b.png", None, "")
    assert "spent earlier in this run" in str(e.value), \
        "the second failure should say why it did not even try"
    assert broke.calls == 1
