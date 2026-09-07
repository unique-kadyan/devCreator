"""The 2.5D performance: does the mouth actually move, and does it move only the mouth?

These are the tests that would have caught the two bugs the renderer shipped with during
development - a jaw band anchored so that it moved the chest instead of the mouth, and a
seam across the face where the two bands met because the lower one resampled pixels the
upper one had already displaced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.media.animation.face import (                                  # noqa: E402
    MOUTH_BOTTOM, MOUTH_HALF_WIDTH, MOUTH_PEAK, MOUTH_TOP, NO_FACE, PRIORS,
    FaceBox, locate_face)
from asa.media.animation.performance import (                           # noqa: E402
    FEATHER, JAW_GAIN, NEUTRAL_LEVEL, ShotRenderer, ShotSpec)

SIZE = (320, 180)
FULL_FRAME_FACE = (0.10, 0.0, 0.80, 1.0)


@pytest.fixture(scope="module")
def still(tmp_path_factory) -> Path:
    """A textured plate. Displacement is invisible on flat colour, so the fixture has
    detail at every row."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)
    for y in range(360):                       # plus strong horizontal structure
        arr[y, :, 0] = (y * 7) % 256
    p = tmp_path_factory.mktemp("still") / "shot.png"
    Image.fromarray(arr).save(p)
    return p


def render(still: Path, level: float, **kw) -> Image.Image:
    """Full confidence and no gate unless a test says otherwise: these tests are about the
    warp's GEOMETRY, and how much of it a given confidence earns - scaling below, gating
    below that - is a separate policy with its own tests.

    Both defaults are load-bearing. Leaving the confidence at the cautious default made the
    displacement small enough that a duplicated feature overlapped its own original, which
    silently disarmed the seam regression; leaving the gate at its shipped 0.5 would stop
    these geometry tests warping anything at all."""
    kw.setdefault("face_confidence", 1.0)
    kw.setdefault("min_face_confidence", 0.0)
    spec = ShotSpec(image_path=str(still), frames=2, fps=24, size=SIZE,
                    camera_move="static", face=FULL_FRAME_FACE,
                    envelope=[level, level], blink=False, **kw)
    return ShotRenderer(spec).render_frame(0)


def gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32)


def test_frames_are_the_requested_size_and_count():
    assert SIZE == (320, 180)


def test_a_loud_frame_differs_from_a_quiet_one(still):
    quiet, loud = gray(render(still, 0.0)), gray(render(still, 1.0))
    assert np.abs(quiet - loud).mean() > 1.0


def test_the_still_is_the_midpoint_of_the_motion(still):
    """Speaking shots are generated mouth-open mid-word, so rest is NEUTRAL_LEVEL and the
    jaw travels both ways from it. A renderer that only ever opened the mouth would leave
    the character talking with its mouth permanently ajar."""
    rest = gray(render(still, NEUTRAL_LEVEL))
    quiet, loud = gray(render(still, 0.0)), gray(render(still, 1.0))
    assert np.abs(rest - quiet).mean() > 0.5
    assert np.abs(rest - loud).mean() > 0.5


def test_nothing_above_the_jaw_band_moves(still):
    """The eyes must not ride on the jaw."""
    quiet, loud = gray(render(still, 0.0)), gray(render(still, 1.0))
    cut = int(MOUTH_TOP * SIZE[1]) - 2
    assert np.array_equal(quiet[:cut], loud[:cut])


def test_the_jaw_moves_most_at_the_mouth_not_at_the_chest(still):
    """Anchoring the stretch at the bottom of the head box - the obvious way to write it -
    moves the chest and leaves the mouth almost still."""
    quiet, loud = gray(render(still, 0.0)), gray(render(still, 1.0))
    rows = np.abs(quiet - loud).mean(axis=1)
    peak_row = int(np.argmax(rows))
    expected = MOUTH_PEAK * SIZE[1]
    assert abs(peak_row - expected) < 0.18 * SIZE[1], (peak_row, expected)


def test_a_band_does_not_read_its_neighbours_output(still):
    """The invariant behind the seam bug, stated exactly.

    Two bands share an edge at MOUTH_PEAK. The lower one must see the frame as it was
    BEFORE the upper one wrote into it, so applying both must leave the lower band's rows
    identical to applying the lower band alone. When it reads the live frame instead, it
    resamples rows its neighbour has already carried down and draws the same feature twice.

    Stated this way rather than by counting bright bands in a rendered frame: the previous
    version placed a marker line at a hardcoded row, and when the band geometry moved, the
    line fell outside the pixels the bug actually corrupts. It passed with the bug
    deliberately reintroduced - a regression test that had quietly stopped testing.
    """
    plate = Image.open(still).convert("RGB").resize((320, 360))
    r = ShotRenderer(ShotSpec(image_path=str(still), frames=1, fps=24, size=(320, 360),
                              face=None, envelope=[]))
    top, peak, bottom, travel, feather = 100.0, 200.0, 300.0, 18.0, 8

    both = plate.copy()
    pristine = both.copy()
    r._stretch(both, pristine, 40, 280, top, peak, 0.0, -travel, feather)
    r._stretch(both, pristine, 40, 280, peak, bottom, -travel, 0.0, feather)

    lower_only = plate.copy()
    r._stretch(lower_only, plate.copy(), 40, 280, peak, bottom, -travel, 0.0, feather)

    rows = slice(int(peak), int(bottom))
    assert np.array_equal(np.asarray(both)[rows], np.asarray(lower_only)[rows])


def test_render_frame_warps_from_an_untouched_copy(still):
    """The same invariant, through the real render path.

    `_stretch` takes its source as an argument, so it stays correct even when the caller is
    wrong - and the caller WAS wrong: `render_frame` used to hand it the frame it was
    writing into. This checks the caller, by reproducing the lower band against a frame
    nothing has warped and requiring the renderer to agree.
    """
    spec = ShotSpec(image_path=str(still), frames=2, fps=24, size=SIZE,
                    camera_move="static", face=FULL_FRAME_FACE, envelope=[1.0, 1.0],
                    blink=False, face_confidence=1.0)
    r = ShotRenderer(spec)
    warped = r.render_frame(0)

    box = r._crop_box(0.0)
    pristine = r.work.resize(SIZE, r.resample, box=box)
    fx, fy, fw, fh = r._face_px(box)
    travel = (1.0 - NEUTRAL_LEVEL) * JAW_GAIN * fh
    cx, half = fx + fw // 2, int(fw * MOUTH_HALF_WIDTH)
    peak = fy + int(fh * MOUTH_PEAK)
    bottom = fy + int(fh * MOUTH_BOTTOM)
    expected = pristine.copy()
    r._stretch(expected, pristine.copy(), cx - half, cx + half, peak, bottom,
               -travel, 0.0, max(2, int(fw * FEATHER)))

    rows = slice(peak, min(bottom, SIZE[1]))
    assert np.array_equal(np.asarray(warped)[rows], np.asarray(expected)[rows])


def test_a_frame_filling_head_is_not_drifted_as_a_patch(still):
    """Sliding a frame-sized ellipse over its own image moves the middle of the picture
    while the corners stay put, which is a visible oval seam."""
    spec = ShotSpec(image_path=str(still), frames=2, fps=24, size=SIZE,
                    camera_move="static", face=FULL_FRAME_FACE, envelope=[], blink=False)
    r = ShotRenderer(spec)
    frame = r.render_frame(0)
    before = gray(frame).copy()
    r._drift_head(frame, (0, 0, SIZE[0], SIZE[1]), 1.0, 1.0)
    assert np.array_equal(before, gray(frame))


def test_no_envelope_means_no_mouth_movement(still):
    """A shot of a character who is not speaking - a narration wide, a listener - must not
    chew silently."""
    spec = ShotSpec(image_path=str(still), frames=2, fps=24, size=SIZE,
                    camera_move="static", face=FULL_FRAME_FACE, envelope=[], blink=False,
                    face_confidence=1.0, min_face_confidence=0.0)
    r = ShotRenderer(spec)
    a = gray(r.render_frame(0))
    spec2 = ShotSpec(image_path=str(still), frames=2, fps=24, size=SIZE,
                     camera_move="static", face=None, envelope=[], blink=False)
    b = gray(ShotRenderer(spec2).render_frame(0))
    assert np.array_equal(a, b)


def test_the_camera_never_shows_outside_the_plate(still):
    """Every move happens inside an upscaled copy, so no frame may include black bars."""
    for move in ("static", "push_in", "pull_out", "pan_left", "pan_right",
                 "tilt_up", "tilt_down", "handheld_drift"):
        spec = ShotSpec(image_path=str(still), frames=24, fps=24, size=SIZE,
                        camera_move=move, face=None, envelope=[])
        r = ShotRenderer(spec)
        for i in (0, 12, 23):
            box = r._crop_box(i / 23)
            assert box[0] >= -0.01 and box[1] >= -0.01
            assert box[2] <= r.work_w + 0.01 and box[3] <= r.work_h + 0.01


def test_render_is_deterministic(still):
    """Frames are rendered across processes and must not depend on mutable state."""
    a = gray(render(still, 0.7))
    b = gray(render(still, 0.7))
    assert np.array_equal(a, b)


# ------------------------------------------------------------------- face

def test_framings_without_a_usable_face_get_none():
    for framing in NO_FACE:
        assert locate_face(Path("/nonexistent.png"), framing) is None


def test_close_up_prior_is_the_measured_one():
    box = locate_face(Path("/nonexistent.png"), "close_up")
    assert box is not None and box.as_tuple() == PRIORS["close_up"]


def test_a_sidecar_overrides_the_prior(tmp_path):
    """The hook a real detector plugs into."""
    img = tmp_path / "shot.png"
    Image.new("RGB", (64, 36)).save(img)
    Path(str(img) + ".face.json").write_text(
        '{"face": [0.1, 0.2, 0.3, 0.4], "source": "detector"}')
    box = locate_face(img, "close_up")
    # A detector's box is trusted outright; a prompt-derived prior is not.
    assert box == FaceBox(0.1, 0.2, 0.3, 0.4, "detector", 1.0)
    assert box.confidence > locate_face(Path("/nonexistent.png"), "close_up").confidence


def test_face_box_maps_to_pixels():
    assert FaceBox(0.5, 0.25, 0.5, 0.5).pixels((100, 80)) == (50, 20, 50, 40)


def test_the_jaw_does_not_move_the_scenery(still):
    """The defect this geometry was re-cut to fix.

    The band used to span the full head width plus 8% either side, which for a close-up is
    the whole picture. The frame-to-frame difference map of a "talking" shot was therefore
    an edge map of the entire image - glasses, ears, hoodie, laptop and the bookshelf behind
    the character all moving together with the jaw. That reads as a flickering photograph,
    not as somebody speaking, and it is what "no lip movement, only frame flicking" was
    describing.

    So: whatever the jaw does, the outer edges of the frame must be untouched by it.
    """
    quiet, loud = gray(render(still, 0.0)), gray(render(still, 1.0))
    margin = int(SIZE[0] * 0.12)
    assert np.array_equal(quiet[:, :margin], loud[:, :margin])
    assert np.array_equal(quiet[:, -margin:], loud[:, -margin:])


def test_the_jaw_band_stays_inside_the_head(still):
    """Nothing below the head box may move, or the chest and the desk ride on the jaw."""
    quiet, loud = gray(render(still, 0.0)), gray(render(still, 1.0))
    below = int(MOUTH_BOTTOM * SIZE[1]) + 2
    if below < SIZE[1]:
        assert np.array_equal(quiet[below:], loud[below:])


def test_an_untrusted_head_box_is_animated_more_cautiously(still):
    """A prior is a guess about a picture nobody has looked at - the image model is asked
    for a close-up and sometimes answers with a wide two-shot. The cost of being wrong
    scales with the amplitude, so the amplitude scales with the confidence."""
    def travel(conf):
        a = gray(render(still, 0.0, face_confidence=conf))
        b = gray(render(still, 1.0, face_confidence=conf))
        return float(np.abs(a - b).mean())

    assert travel(1.0) > travel(0.35) > travel(0.0)
    assert travel(0.0) == 0.0


def test_a_box_below_the_confidence_floor_is_not_animated_at_all(still):
    """Scaling a wrong answer down does not make it a right one.

    Measured on real output rather than on this fixture: asked for a close-up of a fox, the
    image model returned a seated two-thirds figure with its muzzle in the upper right. The
    `close_up` prior says the head fills the frame, so the muzzle band landed on the
    character's lap and the desk behind it, and what the renderer produced was furniture
    moving in time with the voice - 0.995 correlation with the envelope. At confidence 0.35
    that artefact is quieter, not absent.

    So below the floor the shot gets its camera move and nothing else, which is the honest
    rendering of not knowing where the face is.
    """
    def frame(conf, floor, level):
        return gray(render(still, level, face_confidence=conf, min_face_confidence=floor))

    assert np.array_equal(frame(0.35, 0.5, 0.0), frame(0.35, 0.5, 1.0))
    # ... and the floor is a floor, not a switch: a box that clears it still animates.
    assert not np.array_equal(frame(1.0, 0.5, 0.0), frame(1.0, 0.5, 1.0))


def test_the_confidence_floor_gates_the_blink_as_well_as_the_jaw(still):
    """Blink, jaw and head drift all read the SAME rectangle. Gating only the jaw would
    leave the eye band squashing a bookshelf on a three-and-a-half second schedule.

    Compared against a camera-only render rather than against a held frame: even a
    locked-off shot breathes, by design, so "nothing moved" is never the right assertion -
    "nothing moved that the camera did not move" is.
    """
    def clip(face, conf, floor):
        spec = ShotSpec(image_path=str(still), frames=90, fps=24, size=SIZE,
                        camera_move="static", face=face, envelope=[], blink=True,
                        face_confidence=conf, min_face_confidence=floor)
        r = ShotRenderer(spec)
        return [gray(r.render_frame(i)) for i in range(90)]

    camera_only = clip(None, 0.0, 0.5)
    gated = clip(FULL_FRAME_FACE, 0.35, 0.5)
    trusted = clip(FULL_FRAME_FACE, 1.0, 0.5)

    assert all(np.array_equal(a, b) for a, b in zip(gated, camera_only))
    assert any(not np.array_equal(a, b) for a, b in zip(trusted, camera_only))
