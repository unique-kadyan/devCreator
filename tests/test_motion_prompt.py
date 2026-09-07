"""The per-shot motion prompt: what a hosted video model is actually told to animate."""
from __future__ import annotations

from asa.assemble.shotlist import Shot
from asa.media.video.motion import shot_motion_prompt

CAST = {"c1": {"character_id": "c1", "species": "lion", "appearance": "dark mane",
               "clothing": "a charcoal suit", "accessories": "[]"}}


def _shot(**kw):
    base = dict(index=0, speaker="c1", lines=(0,), framing="close_up", emotion="angry",
                camera_move="push_in", image_key="c1|close_up|tense")
    base.update(kw)
    return Shot(**base)


def test_speaking_shot_names_the_mouth_before_anything_else_moves():
    p = shot_motion_prompt({"action": "The lorry is unloaded in the rain"}, _shot(), CAST)
    assert "mouth moving" in p
    assert p.index("mouth moving") < p.index("camera pushes in")


def test_gesture_scales_with_the_framing():
    """Hands are asked for where there is room for them and nowhere else."""
    tight = shot_motion_prompt({}, _shot(framing="extreme_close_up"), CAST)
    wide = shot_motion_prompt({}, _shot(framing="full"), CAST)
    assert "hand" not in tight and "arm" not in tight
    assert "arm" in wide or "hand" in wide


def test_narration_shot_leads_with_the_place_not_an_invented_speaker():
    """Nobody is on screen being talked to, so a narration shot must not describe listening."""
    p = shot_motion_prompt({"action": "Rain falls on the stopped lorry"},
                           _shot(speaker=None, lines=(), framing="wide"), CAST)
    assert p.startswith("Rain falls on the stopped lorry")
    assert "listening" not in p and "toward the speaker" not in p
    assert "the location alive" in p


def test_a_line_that_lands_on_an_insert_does_not_ask_for_a_mouth():
    """Past its image budget a scene reuses a narration setup, so a speaking shot can end up
    framed on a detail with no character in it. A mouth asked for there gets hallucinated
    onto the object."""
    p = shot_motion_prompt({}, _shot(framing="insert"), CAST)
    assert "mouth" not in p
    assert "no character in frame" in p


def test_identity_uses_the_image_prompt_wording():
    p = shot_motion_prompt({}, _shot(), CAST)
    assert "anthropomorphic lion" in p
    assert "charcoal suit" in p


def test_hindi_action_is_dropped_not_pasted():
    """An English-trained video model has no use for Devanagari, and it crowds out the rest."""
    hindi = shot_motion_prompt({"action": "कबीर ट्रक के पास खड़ा है"}, _shot(), CAST)
    english = shot_motion_prompt({"action": "Kabir stands by the truck"}, _shot(), CAST)
    assert "कबीर" not in hindi
    assert "Kabir stands by the truck" in english


def test_every_shot_forbids_an_invented_cut():
    """The shot list decides where cuts go; a model that adds one destroys the edit."""
    for framing in ("wide", "close_up", "insert", "full", "medium", "over_shoulder"):
        p = shot_motion_prompt({}, _shot(framing=framing), CAST)
        assert "one continuous take" in p


def test_adjacent_shots_of_one_scene_no_longer_ask_for_the_same_motion():
    scene = {"action": "Kabir stands by the truck"}
    a = shot_motion_prompt(scene, _shot(framing="close_up", camera_move="push_in"), CAST)
    b = shot_motion_prompt(scene, _shot(framing="full", camera_move="pan_left",
                                        emotion="happy"), CAST)
    assert a != b


def test_works_without_a_cast():
    """The local renderer ignores this field, so a scene alone must still build a job."""
    assert shot_motion_prompt({}, _shot(), None)


# --------------------------------------------------------------- character_look
# Lives here rather than in a cast test because these two defects were found by reading
# generated motion prompts against the live job_00009 cast, and they reach the IMAGE prompt
# through the same function.

from asa.media.images.scene_image import character_look


def test_absence_words_are_not_pasted_in_as_description():
    """A boar stored with clothing "none" was being asked for as "wearing none", which an
    image model reads as an instruction rather than an absence."""
    look = character_look({"species": "boar", "clothing": "none", "accessories": "[]"})
    assert "wearing none" not in look
    assert "wearing simple everyday clothes" in look


def test_non_english_appearance_is_dropped_not_pasted():
    """A Hindi channel stores appearance in the story's language; FLUX cannot read it, and
    it crowds out the clauses that do steer the generation."""
    look = character_look({"species": "boar", "age_band": "elder",
                           "appearance": "भारी कद, धूसर फर, बड़े नाखून",
                           "clothing": "none", "accessories": "[]"})
    assert "भारी" not in look
    assert look.startswith("an anthropomorphic boar, elder")


def test_english_records_are_untouched():
    look = character_look({"species": "fox", "appearance": "lean red fox",
                           "clothing": "blue hoodie", "accessories": '["a satchel"]'})
    assert look == ("an anthropomorphic fox, lean red fox, wearing blue hoodie, "
                    "with a satchel")
