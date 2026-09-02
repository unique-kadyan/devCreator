"""Runtime shortfall detection.

The gap: the scene prompt asks for target_minutes * 150 words and nothing verified the
result. A run targeting 3 minutes produced 82 seconds of audio, and job 1 targeting 7
minutes shipped at 1:20. QC cannot catch it - it warns below one minute and never compares
against what was requested, and by then the render is already paid for.
"""
from types import SimpleNamespace

from asa.story.schema import DialogueLine
from asa.story.generator import WORDS_PER_MINUTE, StoryGenerator


def _scenes(*word_counts):
    return SimpleNamespace(scenes=[
        SimpleNamespace(narration=" ".join(["word"] * n), dialogue=[])
        for n in word_counts])


def _gen(target_minutes=3.0):
    return StoryGenerator(chain=None, target_minutes=target_minutes)


def test_counts_only_spoken_words():
    # Built from the REAL DialogueLine model. The first version of this test used a stub
    # with a `text` attribute, which is not what the schema calls the field - so the test
    # passed while the estimator counted zero words for every story in production.
    sc = SimpleNamespace(scenes=[SimpleNamespace(
        narration="one two three",
        dialogue=[DialogueLine(character_id="a", line="four five"),
                  DialogueLine(character_id="a", line="six")])])
    assert StoryGenerator.spoken_words(sc) == 6


def test_dialogue_uses_the_schema_field_name():
    sc = SimpleNamespace(scenes=[SimpleNamespace(
        narration="", dialogue=[DialogueLine(character_id="a",
                                             line="one two three four")])])
    assert StoryGenerator.spoken_words(sc) == 4, "dialogue is being counted as zero"


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
    # The expansion must lengthen the story, never rewrite it.
    low = prompt.lower()
    assert "do not change the ending" in low, "expansion must not rewrite the story"
    assert "do not invent new characters" in low
    assert "ORIGINAL" in prompt, "the original instructions must be carried through"


class TestExpansionShape:
    """The prompt must ask for the right REMEDY, not just more words.

    One pass took a story from 54s to 225s against a 420s target and stalled there: nine
    scenes were being asked to carry 47 seconds of dialogue each, because the prompt
    forbade adding scenes. A scene stops reading as a single beat past ~12 seconds.
    """

    def test_asks_for_more_scenes_when_the_count_is_far_short(self):
        g = _gen(target_minutes=7.0)
        sc = _scenes(*([15] * 9))
        prompt = g._expand_prompt("ORIG", sc, 420.0)
        assert "up from 9" in prompt
        assert "Add the new scenes BETWEEN existing ones" in prompt
        assert "Keep every existing scene, in order." in prompt

    def test_deepens_instead_when_the_scene_count_is_already_right(self):
        g = _gen(target_minutes=2.0)
        sc = _scenes(*([5] * 10))          # 10 scenes is plenty for 2 minutes
        prompt = g._expand_prompt("ORIG", sc, 120.0)
        assert "Keep the same scenes, in the same order." in prompt
        assert "up from" not in prompt

    def test_never_asks_for_fewer_scenes(self):
        g = _gen(target_minutes=1.0)
        sc = _scenes(*([5] * 30))
        prompt = g._expand_prompt("ORIG", sc, 60.0)
        assert "up from" not in prompt

    def test_scene_count_is_capped(self):
        # A 20 minute target must not ask for 100 scenes; render time is per scene.
        g = _gen(target_minutes=20.0)
        sc = _scenes(*([5] * 4))
        prompt = g._expand_prompt("ORIG", sc, 1200.0)
        assert "about 40 scenes" in prompt

    def test_the_ending_is_protected(self):
        g = _gen(target_minutes=7.0)
        prompt = g._expand_prompt("ORIG", _scenes(*([15] * 9)), 420.0)
        assert "do not change the ending" in prompt.lower()
