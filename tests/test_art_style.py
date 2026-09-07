"""The art-style preset: cartoon or photoreal, switched as one thing.

The bug this file exists to prevent is not a crash. It is a prompt that argues with
itself - a cartoon subject sent under the photoreal negative, which ends with "cartoon,
illustration, drawing, anime, 3d render, cgi". Nothing errors; the model simply settles
the argument differently on every generation, and the episode comes back part cartoon and
part photograph. That is the same art-style drift `_PUPPET_STYLE_WORDS` guards against,
arriving from the other side, and it is invisible until someone watches the render.
"""
from pathlib import Path

import pytest

from asa.assemble.shot_render import generate_shot_images, shot_image_name
from asa.media.images.scene_image import (CINEMATIC_NEGATIVE, CINEMATIC_STYLE, PRESETS,
                                          negative_for, preset, styled_for)
from asa.media.video.motion import medium_clause

SCENE = {"idx": 1, "shot": "close_up", "emotion": "neutral", "camera_move": "static",
         "staging": {"a": {}}, "action": "The fox counts the till",
         "visual_prompt": "a small bakery at night",
         "dialogue": [{"idx": 0, "character_id": "a", "line": "Forty rupees short.",
                       "emotion": "worried"},
                      {"idx": 1, "character_id": "a", "line": "And two people had a key.",
                       "emotion": "worried"}]}
CAST = {"a": {"character_id": "a", "name": "Ruhi", "species": "fox",
              "appearance": "russet fur", "clothing": "a green apron"}}


# ------------------------------------------------------------------ the invariant

@pytest.mark.parametrize("name", sorted(PRESETS))
def test_no_preset_forbids_the_medium_it_asks_for(name):
    """The whole point. `cartoon` must not be in the cartoon negative, and the photoreal
    subject's own words must not be in the photoreal negative."""
    p = preset(name)
    negative = {w.strip() for w in p.negative.split(",")}
    # Every single word of the subject clause that also appears as a standalone negative
    # term is a clause cancelling itself out.
    subject_words = {w.strip() for w in p.subject.split(",")}
    assert not (subject_words & negative), f"{name} asks for and forbids the same thing"


def test_cartoon_is_not_rendered_under_the_photoreal_negative():
    cartoon = styled_for((1080, 1920), None, "cartoon").lower()
    assert "cartoon" in cartoon or "animated" in cartoon
    assert "cartoon" not in negative_for("cartoon").lower()
    # ...and the photoreal negative, which is where this went wrong, still bans it.
    assert "cartoon" in CINEMATIC_NEGATIVE


def test_photoreal_is_not_rendered_under_the_cartoon_negative():
    assert "photorealistic" in negative_for("cartoon")
    assert "photorealistic" not in negative_for("photoreal")


# ------------------------------------------------------------------ lip-sync needs a mouth

def test_the_cartoon_subject_asks_for_a_mouth_a_model_can_find():
    """A hosted lip-sync model animates the mouth in the still it is given. The image
    model's default for "cartoon fox" is a dot, so the muzzle has to be requested."""
    style = styled_for((1080, 1920), None, "cartoon").lower()
    assert "muzzle" in style and "mouth" in style
    assert "eyes" in style


def test_the_cartoon_subject_still_asks_for_a_human_body():
    """Same measured failure as photoreal: the model's default anthro is a four-legged
    mascot, and a positive clause alone did not beat it."""
    style = styled_for((1080, 1920), None, "cartoon").lower()
    assert "human" in style
    assert "quadruped" in negative_for("cartoon")


def test_no_studio_is_named_in_any_preset():
    """prompts/_blocks/safety_rules.md forbids evoking an existing studio, and that rule is
    not suspended because the request is going to an image model rather than to the writer.
    Naming one is also the fastest way to this look, which is why this is a test."""
    from asa.qc.checks import STUDIO_TERMS
    for name, p in PRESETS.items():
        assert not STUDIO_TERMS.search(p.subject), name
        assert not STUDIO_TERMS.search(p.look), name


# ------------------------------------------------------------------ back-compat

def test_the_default_is_still_the_photoreal_channel():
    """Every caller that passes no preset must get exactly what it got before presets
    existed. Silently regrading a running channel is not a default's job."""
    assert styled_for((1920, 1080)).startswith(CINEMATIC_STYLE)
    assert negative_for(None) == CINEMATIC_NEGATIVE
    assert negative_for("") == CINEMATIC_NEGATIVE


def test_a_typo_in_config_says_so_rather_than_rendering_the_wrong_channel():
    with pytest.raises(ValueError) as e:
        styled_for((16, 9), None, "cartoonish")
    assert "cartoon" in str(e.value) and "photoreal" in str(e.value)


def test_look_hint_regrades_a_preset_without_changing_its_medium():
    """The two dials are independent: preset picks the medium, look_hint picks the grade."""
    graded = styled_for((1080, 1920), "lit by a single candle", "cartoon")
    assert "lit by a single candle" in graded
    assert "3D animated feature film still" in graded
    assert "high-key" not in graded


# ------------------------------------------------------------------ the join

class _Plate:
    def __init__(self, path):
        self.path, self.cached = path, False


class _Images:
    """Records the (prompt, negative) pair each generation was actually sent."""
    size = (1080, 1920)

    def __init__(self):
        self.calls = []

    def scene(self, idx, prompt, out_dir, negative, name=None):
        self.calls.append((prompt, negative))
        path = Path(out_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return _Plate(path)


class _Ctx:
    def __init__(self):
        self.images = _Images()


def test_the_subject_and_the_negative_come_from_the_same_preset(tmp_path):
    ctx = _Ctx()
    generate_shot_images(ctx, SCENE, CAST, tmp_path, max_images=2, art_style="cartoon")
    assert ctx.images.calls
    for prompt, negative in ctx.images.calls:
        assert "3D animated feature film still" in prompt
        assert "cartoon" not in negative.lower()
        assert "photorealistic" in negative


def test_passing_no_style_generates_exactly_what_it_used_to(tmp_path):
    ctx = _Ctx()
    generate_shot_images(ctx, SCENE, CAST, tmp_path, max_images=2)
    for prompt, negative in ctx.images.calls:
        assert CINEMATIC_STYLE.split(",")[0] in prompt
        assert negative == CINEMATIC_NEGATIVE


def test_the_image_filename_does_not_depend_on_the_style(tmp_path):
    """The animate stage recomputes this name from the same key and would find nothing if
    the art stage had folded the preset into it. A style change must be a re-render, not a
    silent miss that reports "re-run the art stage" on a job that just ran it."""
    ctx = _Ctx()
    a = generate_shot_images(ctx, SCENE, CAST, tmp_path / "a", max_images=2,
                             art_style="cartoon")
    b = generate_shot_images(_Ctx(), SCENE, CAST, tmp_path / "b", max_images=2,
                             art_style="photoreal")
    assert sorted(p.name for p in a["paths"].values()) == \
           sorted(p.name for p in b["paths"].values())
    assert all(n == shot_image_name(1, k) for k, n in
               ((k, v.name) for k, v in a["paths"].items()))


# ------------------------------------------------------------------ the video anchor

def test_the_motion_prompt_anchors_the_medium_for_a_hosted_model():
    assert "cartoon" in medium_clause("cartoon")
    assert "photoreal" in medium_clause("photoreal")


def test_a_missing_or_unknown_style_adds_no_anchor_rather_than_failing():
    """Unlike the image path, a missing guard rail here must not fail a render that is
    otherwise fine - the still already carries most of the look."""
    assert medium_clause(None) == ""
    assert medium_clause("cartoonish") == ""
