"""The check that separates a figure from a hallucination.

Worth its own tests because the thing it guards against is not "no detection" - that is
harmless, the prior takes over and the shot gets a camera move. It is a CONFIDENT detection
of something that is not a character, which is how a finished episode ended up animating a
laptop and a bookshelf in time with the voice.

The measured case: on a still of a pottery shelf the pose model returned a full skeleton with
every landmark at visibility 1.00, whose ankles sat above its hips. So `visibility` cannot be
the filter and anatomy has to be.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.media.animation.detect import (                     # noqa: E402
    L_HIP, L_SHOULDER, MIN_TORSO, NOSE, R_HIP, R_SHOULDER, Presence, _plausible)


class LM:
    """One landmark. `visibility` is deliberately settable and deliberately ignored."""

    def __init__(self, x=0.5, y=0.5, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def skeleton(nose_y=0.2, sh_y=0.4, hip_y=0.8, nose_x=0.5):
    """33 landmarks with the four the check reads placed where asked."""
    pts = [LM() for _ in range(33)]
    pts[NOSE] = LM(nose_x, nose_y)
    pts[L_SHOULDER] = LM(nose_x - 0.1, sh_y)
    pts[R_SHOULDER] = LM(nose_x + 0.1, sh_y)
    pts[L_HIP] = LM(nose_x - 0.08, hip_y)
    pts[R_HIP] = LM(nose_x + 0.08, hip_y)
    return pts


def test_a_figure_shaped_like_a_person_is_accepted():
    found = _plausible(skeleton())
    assert isinstance(found, Presence)
    assert found.head_x == 0.5 and found.head_y == 0.2


def test_ankles_above_hips_is_the_pottery_shelf_and_is_refused():
    """The measured hallucination: correctly ordered nowhere, confident everywhere.

    Here it is expressed the way the check sees it - hips ABOVE the shoulders - because that
    is the ordering violation the real garbage skeleton had.
    """
    assert _plausible(skeleton(nose_y=0.2, sh_y=0.5, hip_y=0.3)) is None


def test_a_head_below_the_shoulders_is_refused():
    assert _plausible(skeleton(nose_y=0.6, sh_y=0.4, hip_y=0.8)) is None


def test_a_torso_too_short_to_be_a_torso_is_refused():
    """Correctly ordered and still noise: three landmarks a hair apart order themselves
    correctly about as often as not, so ordering alone is not enough."""
    tiny = MIN_TORSO / 2
    assert _plausible(skeleton(nose_y=0.40, sh_y=0.45, hip_y=0.45 + tiny)) is None
    assert _plausible(skeleton(nose_y=0.40, sh_y=0.45, hip_y=0.45 + MIN_TORSO * 2)) is not None


def test_full_visibility_does_not_buy_a_bad_skeleton_anything():
    """`visibility` was 1.00 on every landmark of the garbage detection. If it ever starts
    counting here, that failure comes straight back."""
    bad = skeleton(nose_y=0.2, sh_y=0.5, hip_y=0.3)
    for p in bad:
        p.visibility = 1.0
    assert _plausible(bad) is None


def test_the_module_does_not_offer_a_head_box():
    """Deliberate, and the docstring explains why: the landmark SCALE is wrong on these
    images, so anything shaped like a head box would be used to place a warp band and would
    put it on a chest. Presence only, until that is measured properly."""
    import asa.media.animation.detect as d

    assert not hasattr(d, "head_box")
    assert not hasattr(d, "write_sidecar")
    assert set(Presence.__dataclass_fields__) == {"head_x", "head_y", "torso"}
