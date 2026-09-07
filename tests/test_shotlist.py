"""Shot planning: where a scene cuts, and what it costs in images.

The property that matters most here is that `plan_shots` is a pure function of the scene's
dialogue. The art stage plans shots before any audio exists in order to buy the right
pictures, and the animate stage plans them again with measured durations; if the two ever
disagreed, the renderer would look for an image nobody generated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.assemble.shotlist import (                                   # noqa: E402
    FACE_FRAMINGS, MAX_PARTS, PLAN_SHOT_S, WORDS_PER_SECOND, emotion_bucket,
    image_keys, plan_shots, time_shots)


@dataclass
class FakeUtterance:
    duration_s: float
    path: Path = Path("/dev/null")
    envelope: list = None


@dataclass
class FakeCue:
    start_s: float
    utterance: FakeUtterance
    character_id: str | None

    @property
    def end_s(self) -> float:
        return self.start_s + self.utterance.duration_s


@dataclass
class FakeTiming:
    cues: list
    duration_s: float

    def envelope_for(self, character_id, fps):
        n = int(round(self.duration_s * fps)) + 1
        env = [0.0] * n
        for c in self.cues:
            if c.character_id != character_id:
                continue
            for k in range(int(c.start_s * fps), min(n, int(c.end_s * fps))):
                env[k] = 0.8
        return env


def scene(dialogue, idx=1, staging=("a", "b"), shot="medium", emotion="neutral"):
    return {"idx": idx, "shot": shot, "emotion": emotion, "camera_move": "static",
            "staging": {k: {} for k in staging},
            "dialogue": [{"idx": i, "character_id": c, "emotion": e}
                         for i, (c, e) in enumerate(dialogue)]}


def timing_for(sc, gap=0.4, lead=0.35, tail=0.5, line_s=2.0):
    cues, clock = [], lead
    for line in sc["dialogue"]:
        cues.append(FakeCue(clock, FakeUtterance(line_s), line["character_id"]))
        clock += line_s + gap
    return FakeTiming(cues, round(clock - gap + tail, 3))


def test_plan_is_deterministic_across_stages():
    sc = scene([(None, "neutral"), ("a", "angry"), ("b", "happy"), ("a", "angry")])
    first = plan_shots(sc)
    second = plan_shots(dict(sc))
    assert [s.image_key for s in first] == [s.image_key for s in second]
    assert [s.framing for s in first] == [s.framing for s in second]


def test_every_speaker_change_starts_a_new_shot():
    sc = scene([("a", "neutral"), ("b", "neutral"), ("a", "neutral")])
    shots = plan_shots(sc)
    assert [s.speaker for s in shots] == ["a", "b", "a"]


def test_consecutive_lines_by_one_speaker_are_one_shot():
    sc = scene([("a", "neutral"), ("a", "neutral"), ("b", "neutral")])
    shots = plan_shots(sc)
    assert [s.speaker for s in shots] == ["a", "b"]
    assert shots[0].lines == (0, 1)


def test_narration_never_animates_a_face():
    """A narrated line is not spoken by anyone on screen, so it must not open a mouth."""
    sc = scene([(None, "neutral")])
    shot = plan_shots(sc)[0]
    assert shot.speaker is None
    assert not shot.speaking
    assert not shot.animates_face
    assert shot.framing not in FACE_FRAMINGS


def test_speaking_shots_are_framed_for_a_face():
    sc = scene([("a", "neutral"), ("b", "happy")])
    for shot in plan_shots(sc):
        assert shot.animates_face, shot


def test_distinct_images_are_capped():
    """A talkative scene cuts many times without buying an image for every cut."""
    lines = [(c, e) for c, e in
             [("a", "angry"), ("b", "happy"), ("a", "sad"), ("b", "scared"),
              ("a", "excited"), ("b", "neutral"), ("a", "wry")]]
    shots = plan_shots(scene(lines), max_images=3)
    assert len(shots) == 7
    assert len(image_keys(shots)) <= 3


def test_a_reused_setup_keeps_its_own_framing():
    """When a shot falls back to an earlier setup it must claim that setup's FRAMING too -
    the image it is now using is a picture of that framing, whatever it asked for."""
    shots = plan_shots(scene([("a", "angry"), ("a", "happy"), ("a", "sad"),
                              ("a", "excited")]), max_images=2)
    by_key = {}
    for s in shots:
        by_key.setdefault(s.image_key, set()).add(s.framing)
    assert all(len(v) == 1 for v in by_key.values())


def test_emotion_buckets_collapse_to_three():
    assert emotion_bucket("happy") == emotion_bucket("excited") == "warm"
    assert emotion_bucket("angry") == emotion_bucket("scared") == "tense"
    assert emotion_bucket(None) == "neutral"


# ----------------------------------------------------------------- timing

def test_frames_sum_to_the_scene_exactly():
    """Shots are stream-copy concatenated and the audio is laid against the SCENE
    duration, so a frame of rounding drift per shot walks the voices out of sync."""
    sc = scene([(None, "neutral"), ("a", "angry"), ("b", "happy")])
    t = timing_for(sc)
    shots = time_shots(plan_shots(sc), t, fps=24)
    assert sum(s.frames for s in shots) == int(round(t.duration_s * 24))
    assert shots[0].start_frame == 0
    for prev, nxt in zip(shots, shots[1:]):
        assert prev.end_frame == nxt.start_frame


def test_cut_lands_in_the_silence_between_two_lines():
    sc = scene([("a", "neutral"), ("b", "neutral")])
    t = timing_for(sc, gap=1.0)
    shots = time_shots(plan_shots(sc), t, fps=24)
    first_end = shots[0].end_s
    a_end = t.cues[0].end_s
    b_start = t.cues[1].start_s
    assert a_end <= first_end <= b_start


def test_a_long_speech_is_recut_without_buying_another_image():
    sc = scene([("a", "neutral")])
    t = timing_for(sc, line_s=18.0)
    shots = time_shots(plan_shots(sc), t, fps=24, max_shot_s=6.0)
    assert len(shots) > 1
    assert len({s.image_key for s in shots}) == 1
    assert len({s.camera_move for s in shots}) > 1


def test_a_flash_cut_back_to_the_same_setup_is_dropped():
    sc = scene([("a", "neutral"), ("a", "neutral")])
    t = timing_for(sc, line_s=0.3, gap=0.05)
    shots = time_shots(plan_shots(sc), t, fps=24, min_shot_s=1.5)
    assert len(shots) == 1


def test_a_short_line_by_another_speaker_keeps_its_cut():
    """Dropping it would leave a silent face on screen while a different voice speaks -
    the exact failure the shot layer exists to fix."""
    sc = scene([("a", "neutral"), ("b", "neutral"), ("a", "neutral")])
    t = timing_for(sc, line_s=0.4, gap=0.1)
    shots = time_shots(plan_shots(sc), t, fps=24, min_shot_s=2.0)
    assert [s.speaker for s in shots] == ["a", "b", "a"]


def test_a_scene_with_no_audio_still_renders_one_shot():
    sc = scene([])
    shots = time_shots(plan_shots(sc), FakeTiming([], 4.0), fps=24)
    assert len(shots) == 1
    assert shots[0].frames == 96


# ---------------------------------------------------------------- long speeches
#
# A long speech used to be covered by ONE picture with different camera moves panned across
# it. Measured on a finished episode that left 52% of the runtime resting on a still while a
# voice played over it - the "static character, voice from the background" complaint, which
# is a shot problem before it is a lip-sync problem.


def words(n: int) -> str:
    return " ".join(["shabd"] * n)


def spoken(dialogue, **kw):
    """Like `scene`, but the lines carry text - which is what the split is estimated from."""
    sc = scene([(c, e) for c, e, _ in dialogue], **kw)
    for line, (_, _, n) in zip(sc["dialogue"], dialogue):
        line["line"] = words(n)
    return sc


def test_a_long_speech_is_planned_as_several_distinct_pictures():
    long_words = int(WORDS_PER_SECOND * PLAN_SHOT_S * 2.5)      # ~12.5 seconds
    shots = plan_shots(spoken([(None, "neutral", long_words)]), max_images=6)
    assert len(shots) > 1
    # The point is DIFFERENT pictures, not the same one panned across.
    assert len(set(s.image_key for s in shots)) == len(shots)


def test_a_short_speech_stays_one_shot():
    shots = plan_shots(spoken([(None, "neutral", 4)]), max_images=6)
    assert len(shots) == 1 and shots[0].parts == 1


def test_a_speech_is_never_split_past_the_cap():
    shots = plan_shots(spoken([(None, "neutral", 900)]), max_images=9)
    assert len(shots) == MAX_PARTS


def test_the_image_budget_still_binds():
    """Splitting must not be able to talk the art stage into unbounded generations."""
    sc = spoken([(None, "neutral", 300), ("a", "happy", 300), ("b", "sad", 300)])
    assert len(image_keys(plan_shots(sc, max_images=3))) <= 3


def test_the_parts_of_one_speech_divide_its_span_instead_of_overlapping():
    """Every part carries the same line indices, so without dividing the span they would
    all claim the whole speech and land on top of each other."""
    sc = spoken([(None, "neutral", int(WORDS_PER_SECOND * PLAN_SHOT_S * 2.5))])
    shots = plan_shots(sc, max_images=6)
    timing = FakeTiming([FakeCue(0.0, FakeUtterance(12.0), None)], 12.0)
    timed = time_shots(shots, timing, fps=24)
    assert len(timed) == len(shots)
    for a, b in zip(timed, timed[1:]):
        assert a.end_frame == b.start_frame           # contiguous
        assert a.frames > 0
    assert timed[-1].end_frame == round(12.0 * 24)


def test_planning_is_still_pure_and_repeatable():
    """The art stage and the animate stage plan separately and must agree, or the renderer
    looks for an image nobody bought."""
    sc = spoken([(None, "neutral", 40), ("a", "happy", 30)])
    assert [s.image_key for s in plan_shots(sc, max_images=6)] == \
           [s.image_key for s in plan_shots(sc, max_images=6)]
