"""Runtime shortfall detection.

The gap: the scene prompt asks for target_minutes * 150 words and nothing verified the
result. A run targeting 3 minutes produced 82 seconds of audio, and job 1 targeting 7
minutes shipped at 1:20. QC cannot catch it - it warns below one minute and never compares
against what was requested, and by then the render is already paid for.
"""
from types import SimpleNamespace

from asa.story.generator import WORDS_PER_MINUTE, StoryGenerator


def _scenes(*word_counts):
    return SimpleNamespace(scenes=[
        SimpleNamespace(narration=" ".join(["word"] * n), dialogue=[])
        for n in word_counts])


def _gen(target_minutes=3.0):
    return StoryGenerator(chain=None, target_minutes=target_minutes)


def test_counts_only_spoken_words():
    sc = SimpleNamespace(scenes=[SimpleNamespace(
        narration="one two three",
        dialogue=[SimpleNamespace(text="four five"), SimpleNamespace(text="six")])])
    assert StoryGenerator.spoken_words(sc) == 6


def test_stage_directions_do_not_count():
    # action / visual_prompt cost render time but produce no audio; counting them would
    # roughly double the estimate and defeat the check entirely.
    sc = SimpleNamespace(scenes=[SimpleNamespace(
        narration="one two", dialogue=[],
        action="a very long stage direction that is never spoken aloud at all",
        visual_prompt="a wide shot of the river at night under a broken lantern")])
    assert StoryGenerator.spoken_words(sc) == 2


def test_runtime_estimate_uses_the_documented_pace():
    g = _gen()
    sc = _scenes(int(WORDS_PER_MINUTE))          # exactly one minute of speech
    assert abs(g.estimated_runtime_s(sc) - 60.0) < 0.01


def test_the_observed_shortfall_is_detected():
    # The real numbers: 15 scenes, ~204 spoken words, 3 minute target -> 82s of audio.
    g = _gen(target_minutes=3.0)
    sc = _scenes(*([14] * 15))
    assert g.estimated_runtime_s(sc) < 3 * 60 * 0.75


def test_a_story_that_meets_target_is_not_flagged():
    g = _gen(target_minutes=3.0)
    sc = _scenes(*([30] * 15))                   # 450 words -> 180s
    assert g.estimated_runtime_s(sc) >= 3 * 60 * 0.75


def test_expansion_prompt_states_the_gap_in_words():
    g = _gen(target_minutes=3.0)
    sc = _scenes(*([14] * 15))
    prompt = g._expand_prompt("ORIGINAL", sc, 180.0)
    assert "210 spoken words" in prompt
    assert "MORE words" in prompt
    assert "Do NOT invent new plot" in prompt, "expansion must not rewrite the story"
    assert "ORIGINAL" in prompt, "the original instructions must be carried through"
