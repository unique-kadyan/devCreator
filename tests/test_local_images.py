"""The local diffusion provider: licence enforcement, chain placement, no model download."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from asa.core.errors import ProviderError
from asa.media.images.base import GeneratedImage
from asa.media.images.local import MODELS, LocalDiffusionImages


def test_a_non_commercial_checkpoint_is_refused_at_construction():
    """SDXL-Turbo is Stability Non-Commercial. A monetised channel must not be able to
    point at it by editing one config line."""
    with pytest.raises(ProviderError) as e:
        LocalDiffusionImages(model="stabilityai/sdxl-turbo", enabled=True)
    assert "allowlist" in str(e.value)


def test_an_unlisted_model_is_refused_even_if_harmless():
    with pytest.raises(ProviderError):
        LocalDiffusionImages(model="someone/unreviewed-model", enabled=True)


def test_disabled_provider_does_not_validate_or_load():
    """Off is off: a bad slug in a disabled block must not break startup."""
    p = LocalDiffusionImages(model="stabilityai/sdxl-turbo", enabled=False)
    assert p.available is False


def test_every_allowlisted_model_declares_a_licence_the_ledger_knows():
    from asa.core.ledger import COMMERCIAL_OK
    for model, code in MODELS.items():
        assert code in COMMERCIAL_OK, f"{model} declares {code}, which cannot be published"


def test_declared_licence_beats_the_provider_table():
    """The table maps `local` to UNKNOWN; the checkpoint's own code must win, or every
    locally generated still would be unpublishable."""
    img = GeneratedImage(path=Path("x.png"), provider="local", model_id="segmind/SSD-1B",
                         prompt_sha="abc", meta={"licence": "APACHE-2.0"})
    assert img.license_code == "APACHE-2.0"


def test_a_provider_that_declares_nothing_still_fails_closed():
    img = GeneratedImage(path=Path("x.png"), provider="local", model_id="?",
                         prompt_sha="abc")
    assert img.license_code == "UNKNOWN"


def test_generate_uses_the_pipeline_and_records_its_licence(tmp_path, monkeypatch):
    """Exercises generate() against a stub pipeline - the real one is a 2.5 GB download."""
    calls = {}

    class StubPipe:
        def __call__(self, **kw):
            calls.update(kw)
            img = Image.new("RGB", (kw["width"], kw["height"]), (10, 20, 30))
            return types.SimpleNamespace(images=[img])

    p = LocalDiffusionImages(model="segmind/SSD-1B", enabled=True, steps=4)
    monkeypatch.setattr(p, "_pipeline", lambda: StubPipe())
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(Generator=lambda d: types.SimpleNamespace(
                            manual_seed=lambda s: None)))

    out = tmp_path / "a.png"
    res = p.generate("a fox", out, (1024, 576), negative="text", seed=1)
    assert out.exists()
    assert res.license_code == "APACHE-2.0"
    assert res.provider == "local"
    assert calls["num_inference_steps"] == 4
    assert (calls["width"], calls["height"]) == (1024, 576)


def test_odd_sizes_are_snapped_to_the_diffusion_grid(tmp_path, monkeypatch):
    """A non-multiple-of-8 size fails deep inside the UNet; catch it here instead."""
    seen = {}

    class StubPipe:
        def __call__(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(
                images=[Image.new("RGB", (kw["width"], kw["height"]))])

    p = LocalDiffusionImages(model="segmind/SSD-1B", enabled=True)
    monkeypatch.setattr(p, "_pipeline", lambda: StubPipe())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=None))
    res = p.generate("x", tmp_path / "b.png", (1023, 575))
    assert seen["width"] % 8 == 0 and seen["height"] % 8 == 0
    # ...but the file still comes back the size the caller asked for.
    with Image.open(res.path) as im:
        assert im.size == (1023, 575)


# ----------------------------------------------------------- long prompts (CLIP 77 tokens)

def test_a_prompt_longer_than_the_clip_window_is_chunked_not_truncated():
    """Measured on the live scene-2 prompt: 169 tokens against CLIP's 77. The 92 that fell
    off were the location clause, the region hint and all of CINEMATIC_STYLE - the picture
    came back as a studio portrait with no set and no grade."""
    import types

    import torch
    from asa.media.images.local import _chunked_embeds

    class Tok:
        model_max_length = 77
        bos_token_id, eos_token_id = 1, 2

        def __call__(self, text, truncation=True, add_special_tokens=True):
            n = len(text.split())
            return types.SimpleNamespace(input_ids=list(range(10, 10 + n)))

    class Enc:
        device = "cpu"

        def __call__(self, ids, output_hidden_states=False):
            b, n = ids.shape
            hs = [torch.zeros(b, n, 768), torch.zeros(b, n, 768)]
            return types.SimpleNamespace(hidden_states=hs, __getitem__=lambda s, i: None)

    pipe = types.SimpleNamespace(tokenizer=Tok(), tokenizer_2=None,
                                 text_encoder=Enc(), text_encoder_2=None)
    long_prompt = " ".join(f"w{i}" for i in range(169))
    out = _chunked_embeds(pipe, long_prompt, "text, watermark")
    assert out is not None
    # 169 tokens needs three 75-token chunks, each emitted at the full 77 window.
    assert out["prompt_embeds"].shape[1] == 231
    # Negative must match the positive or classifier-free guidance cannot pair them.
    assert out["negative_prompt_embeds"].shape == out["prompt_embeds"].shape


def test_an_unrecognised_pipeline_falls_back_rather_than_failing_the_art_stage():
    import types

    from asa.media.images.local import _chunked_embeds
    assert _chunked_embeds(types.SimpleNamespace(), "x", "y") is None
