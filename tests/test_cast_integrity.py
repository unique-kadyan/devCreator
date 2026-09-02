"""Off-cast speakers are dropped, not fatal.

Asking for MORE scenes during runtime expansion invites the model to invent a speaker.
One invented "old_man_mercer" across two scenes discarded an entire expansion pass and
left the episode at 17% of its target runtime. There is no puppet and no voice for a
character that does not exist, so the line is unusable either way - but raising throws
away every good scene alongside it.
"""
import pytest

from asa.core.errors import ValidationError
from asa.story.generator import StoryGenerator
from asa.story.schema import DialogueLine, Scene, SceneList


# SceneList enforces a minimum of three scenes, so fixtures are padded with narrated
# filler rather than shrunk - the constraint under test is about dialogue, not length.
def _filler(index):
    return _scene(index, [], narration="the river runs on")


def _scene(index, dialogue, narration=""):
    return Scene(index=index, location_id="river", action="a",
                 visual_prompt="a river at night", narration=narration,
                 dialogue=dialogue)


def _check_for(cast_ids):
    """Rebuild the post_check the scenes stage installs, for a given cast."""
    valid = set(cast_ids)

    def _check(sl):
        total = sum(len(sc.dialogue) for sc in sl.scenes)
        dropped = []
        for sc in sl.scenes:
            keep = []
            for d in sc.dialogue:
                (keep if d.character_id in valid else dropped).append(d)
            sc.dialogue = keep
        if total and len(dropped) > total * 0.5:
            raise ValidationError("too many off-cast lines")
        spoken = sum(len(sc.dialogue) for sc in sl.scenes)
        narrated = sum(1 for sc in sl.scenes if sc.narration.strip())
        if spoken + narrated == 0:
            raise ValidationError("silent")
    return _check


def test_a_minority_of_off_cast_lines_is_dropped_not_fatal():
    sl = SceneList(scenes=[
        _scene(1, [DialogueLine(character_id="milo_fox", line="one"),
                   DialogueLine(character_id="wren_owl", line="two")]),
        _scene(2, [DialogueLine(character_id="milo_fox", line="three"),
                   DialogueLine(character_id="old_man_mercer", line="four")]),
        _filler(3),
    ])
    _check_for({"milo_fox", "wren_owl"})(sl)
    speakers = {d.character_id for sc in sl.scenes for d in sc.dialogue}
    assert speakers == {"milo_fox", "wren_owl"}
    assert sum(len(sc.dialogue) for sc in sl.scenes) == 3, "good lines were lost too"


def test_scene_order_and_count_survive_the_drop():
    sl = SceneList(scenes=[
        _scene(1, [DialogueLine(character_id="ghost", line="x")], narration="n1"),
        _scene(2, [DialogueLine(character_id="milo_fox", line="y")]),
        _scene(3, [DialogueLine(character_id="milo_fox", line="z")]),
    ])
    _check_for({"milo_fox"})(sl)
    assert [sc.index for sc in sl.scenes] == [1, 2, 3]


def test_mostly_off_cast_still_raises():
    # A different cast means a different story; salvaging it produces an incoherent
    # episode, so repair is the right answer.
    sl = SceneList(scenes=[
        _scene(1, [DialogueLine(character_id="ghost_a", line="one"),
                   DialogueLine(character_id="ghost_b", line="two")]),
        _scene(2, [DialogueLine(character_id="milo_fox", line="three")]),
        _filler(3),
    ])
    with pytest.raises(ValidationError):
        _check_for({"milo_fox"})(sl)


def test_a_silent_result_still_raises():
    sl = SceneList(scenes=[
        _scene(1, [DialogueLine(character_id="ghost", line="x")]),
        _scene(2, [], narration=""),
        _scene(3, [], narration=""),
    ])
    with pytest.raises(ValidationError):
        _check_for({"milo_fox"})(sl)


def test_dropped_lines_do_not_count_toward_runtime():
    # The runtime estimate must reflect what will actually be spoken, or the expansion
    # loop chases a number the audio stage will never produce.
    sl = SceneList(scenes=[
        _scene(1, [DialogueLine(character_id="milo_fox", line="one two three"),
                   DialogueLine(character_id="ghost", line="a b c d e f g h i j")]),
        _scene(2, []),
        _scene(3, []),
    ])
    _check_for({"milo_fox"})(sl)
    assert StoryGenerator.spoken_words(sl) == 3
