"""The metered image path: licence refusals, aspect mapping, and the spend stop.

These are the three things that can quietly cost money or quietly publish something that
was never licensed, so they are tested rather than trusted.
"""
from __future__ import annotations

import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from asa.core.errors import ProviderError, QuotaExhausted
from asa.core.ledger import COMMERCIAL_OK
from asa.media.images.replicate_images import (MODELS, PUBLISHABLE, ReplicateImages,
                                               nearest_aspect)


# ------------------------------------------------------------------ licence

def test_a_non_commercial_model_is_refused_even_though_it_is_on_the_allowlist():
    """flux-dev is LISTED so the refusal can name its licence. Listing is not permission."""
    with pytest.raises(ProviderError) as e:
        ReplicateImages("tok", model="black-forest-labs/flux-dev", enabled=True)
    assert "FLUX-DEV-NONCOMMERCIAL" in str(e.value)


def test_an_unlisted_model_is_refused_rather_than_published_under_a_guess():
    with pytest.raises(ProviderError) as e:
        ReplicateImages("tok", model="someone/brand-new-model", enabled=True)
    assert "allowlist" in str(e.value)


def test_a_disabled_provider_does_not_validate_its_model():
    """A chain listing this provider while it is off must not fail to build."""
    ReplicateImages("tok", model="someone/brand-new-model", enabled=False)


def test_every_publishable_model_has_a_code_the_ledger_will_release():
    """The refusal above is worthless if the code it lets through is one `audit()` blocks."""
    for slug, code in MODELS.items():
        if code in PUBLISHABLE:
            assert code in COMMERCIAL_OK, f"{slug} is publishable but {code} is not cleared"


def test_the_declared_licence_reaches_the_generated_image(monkeypatch, tmp_path):
    img = _stub_generate(monkeypatch, tmp_path)
    assert img.meta["licence"] == "APACHE-2.0"
    assert img.license_code == "APACHE-2.0"


# ------------------------------------------------------------------- aspect

def test_a_portrait_size_maps_to_a_portrait_ratio():
    assert nearest_aspect((768, 1344)) == "9:16"
    assert nearest_aspect((1080, 1920)) == "9:16"


def test_a_landscape_size_maps_to_a_landscape_ratio():
    assert nearest_aspect((1920, 1080)) == "16:9"
    assert nearest_aspect((1024, 576)) == "16:9"


def test_a_square_size_maps_to_square():
    assert nearest_aspect((1024, 1024)) == "1:1"


def test_the_ratio_is_chosen_proportionally_not_by_linear_distance():
    """A linear metric puts 21:9 nearer 16:9 than 9:16 is to 9:21, and so drifts.

    2.4 (ultrawide) must pick 21:9 and not 16:9; the mirrored 0.417 must pick 9:21.
    """
    assert nearest_aspect((2400, 1000)) == "21:9"
    assert nearest_aspect((1000, 2400)) == "9:21"


# -------------------------------------------------------------------- spend

def test_the_budget_stops_generation_rather_than_looping_against_a_metered_api(
        monkeypatch, tmp_path):
    p = _stub_provider(monkeypatch, max_images=1)
    p.generate("a fox", tmp_path / "a.png", (768, 1344))
    assert p.available is False
    with pytest.raises(QuotaExhausted):
        p.generate("a fox", tmp_path / "b.png", (768, 1344))


def test_a_402_latches_so_the_rest_of_the_episode_does_not_pay_to_rediscover_it(
        monkeypatch, tmp_path):
    p = ReplicateImages("tok", enabled=True)

    class R:
        status_code = 402
        text = "insufficient credit"
        headers: dict = {}

    with pytest.raises(QuotaExhausted):
        p._raise_for(R())
    assert p._no_credit is True
    assert p.available is False


# ------------------------------------------------------------------ payload

def test_the_negative_prompt_is_not_sent_because_flux_has_no_such_field(
        monkeypatch, tmp_path):
    sent: dict = {}
    _stub_generate(monkeypatch, tmp_path, capture=sent, negative="blurry, extra limbs")
    assert "negative_prompt" not in sent["input"]
    assert "blurry" not in str(sent["input"])


def test_the_negative_still_changes_the_cache_key_even_though_it_is_not_sent(
        monkeypatch, tmp_path):
    a = _stub_generate(monkeypatch, tmp_path, negative="")
    b = _stub_generate(monkeypatch, tmp_path, negative="blurry")
    assert a.prompt_sha != b.prompt_sha


def test_the_returned_image_is_resized_to_the_size_the_compositor_asked_for(
        monkeypatch, tmp_path):
    """The model answers with the enum's own pixels, which is the right shape, not the
    right size. The frame contract downstream is in pixels."""
    img = _stub_generate(monkeypatch, tmp_path, returns=(512, 896), size=(768, 1344))
    with Image.open(img.path) as im:
        assert im.size == (768, 1344)


# ------------------------------------------------------------------ helpers

def _stub_provider(monkeypatch, *, capture=None, returns=(768, 1344), **kw):
    """A provider whose HTTP layer is replaced by a Replicate that always succeeds."""
    _install_stub_http(monkeypatch, capture=capture, returns=returns)
    return ReplicateImages("tok", enabled=True, poll_s=0.0, **kw)


def _stub_generate(monkeypatch, tmp_path, *, capture=None, negative="",
                   returns=(768, 1344), size=(768, 1344), **kw):
    """Run generate() against a stubbed Replicate that succeeds on the first poll."""
    p = _stub_provider(monkeypatch, capture=capture, returns=returns, **kw)
    return p.generate("an anthropomorphic fox", Path(tmp_path) / "out.png", size,
                      negative=negative)


def _install_stub_http(monkeypatch, *, capture=None, returns=(768, 1344)):
    buf = io.BytesIO()
    Image.new("RGB", returns, (30, 60, 90)).save(buf, format="PNG")
    blob = buf.getvalue()

    class Resp:
        def __init__(self, payload=None, content=b""):
            self.status_code = 200
            self._payload = payload or {}
            self.content = content
            self.text = ""
            self.headers: dict = {}

        def json(self):
            return self._payload

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            if capture is not None:
                capture.update(json)
            return Resp({"id": "p1", "status": "succeeded",
                         "output": ["https://example.invalid/out.png"]})

        def get(self, url, headers=None, follow_redirects=False):
            return Resp(content=blob)

    monkeypatch.setattr(httpx, "Client", Client)
