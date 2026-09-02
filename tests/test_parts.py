"""Part planning.

Parts are cut from measured scene durations. The properties that matter are that every
scene lands in exactly one part, order is preserved, and the final part is never a runt -
ending a series on forty seconds is worse than any window violation.
"""
import pytest

from asa.assemble.parts import Part, part_title, plan_parts

MIN, MAX = 150.0, 180.0


def _flat(parts):
    return [i for p in parts for i in p.scenes]


def test_short_story_stays_one_part():
    parts = plan_parts([40.0] * 3, min_s=MIN, max_s=MAX)
    assert len(parts) == 1
    assert parts[0].scenes == [0, 1, 2]


def test_every_scene_lands_in_exactly_one_part_in_order():
    ds = [17.0, 23.5, 41.0, 12.0, 33.0, 28.0, 19.0, 44.0, 26.0, 31.0,
          22.0, 38.0, 15.0, 29.0, 24.0, 36.0, 21.0, 27.0]
    parts = plan_parts(ds, min_s=MIN, max_s=MAX)
    assert _flat(parts) == list(range(len(ds))), "scenes lost, duplicated or reordered"
    assert [p.index for p in parts] == list(range(1, len(parts) + 1))


def test_part_count_tracks_total_runtime():
    # ~9 minutes of material against a 2.5-3 minute window should be three parts.
    parts = plan_parts([30.0] * 18, min_s=MIN, max_s=MAX)
    assert len(parts) == 3
    for p in parts:
        assert MIN <= p.duration_s <= MAX


def test_no_runt_final_part():
    # Greedy filling to max_s leaves a tiny tail here; balanced cutting must not.
    ds = [25.0] * 25                                   # 625s
    parts = plan_parts(ds, min_s=MIN, max_s=MAX)
    assert len(parts) >= 2
    shortest = min(p.duration_s for p in parts)
    longest = max(p.duration_s for p in parts)
    assert shortest > 0.6 * longest, f"lopsided split: {[p.duration_s for p in parts]}"


def test_never_produces_an_empty_part():
    for n in range(1, 40):
        parts = plan_parts([20.0] * n, min_s=MIN, max_s=MAX)
        assert all(p.scenes for p in parts)
        assert _flat(parts) == list(range(n))


def test_a_single_overlong_scene_cannot_be_split():
    # Scenes are atomic; one 5-minute scene is one part however the window is set.
    parts = plan_parts([300.0], min_s=MIN, max_s=MAX)
    assert len(parts) == 1 and parts[0].duration_s == 300.0


def test_empty_input():
    assert plan_parts([], min_s=MIN, max_s=MAX) == []


class TestPartTitle:
    def test_single_part_gets_no_marker(self):
        p = Part(1, [0], 100.0)
        assert part_title("The Lantern Promise", p, 1) == "The Lantern Promise"

    def test_marker_is_appended(self):
        p = Part(2, [3, 4], 160.0)
        assert part_title("The Lantern Promise", p, 3) == "The Lantern Promise (Part 2 of 3)"

    def test_long_title_is_trimmed_but_marker_survives(self):
        p = Part(2, [0], 160.0)
        out = part_title("x" * 120, p, 3)
        assert len(out) <= 100
        assert out.endswith("(Part 2 of 3)"), "the marker was trimmed instead of the title"

    def test_trailing_punctuation_is_cleaned_before_the_marker(self):
        p = Part(1, [0], 160.0)
        out = part_title("A Story -", p, 2)
        assert out == "A Story (Part 1 of 2)"
