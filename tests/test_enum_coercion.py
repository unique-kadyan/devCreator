"""Closed vocabularies accept a model's near-miss instead of failing the story.

Three separate runs died on this: `shot`, then `gesture`, then `emotion='calm'` where the
schema wanted 'neutral'. Each time the repair pass produced a *different* near-miss and the
job failed after art, voice and rendering would already have been paid for. The vocabulary
stays closed - the renderer has artwork for these poses and nothing else - but a synonym is
mapped rather than rejected.
"""
import pytest

from asa.media.animation.shots import (CAMERA_MOVES, EMOTIONS, GESTURES, SHOT_TYPES,
                                       TRANSITIONS, EMOTION_SYNONYMS, coerce)
from asa.story.schema import Camera, Scene, Staging


def _scene(**kw):
    base = dict(index=1, location_id="river", action="a", visual_prompt="a river at night")
    base.update(kw)
    return Scene(**base)


class TestEmotion:
    def test_the_word_that_killed_job_3(self):
        assert _scene(emotion="calm").emotion == "neutral"

    @pytest.mark.parametrize("given,want", [
        ("worried", "scared"), ("frustrated", "angry"), ("shocked", "surprised"),
        ("resolute", "determined"), ("eager", "excited"), ("puzzled", "curious"),
        ("sarcastic", "wry"), ("joyful", "happy"), ("wistful", "sad"),
    ])
    def test_synonyms_map(self, given, want):
        assert _scene(emotion=given).emotion == want

    def test_case_and_spacing_are_normalised(self):
        assert _scene(emotion="  Worried ").emotion == "scared"

    def test_unmappable_falls_back_rather_than_failing(self):
        assert _scene(emotion="zzzz").emotion == "neutral"

    def test_valid_values_pass_through_untouched(self):
        for e in EMOTIONS:
            assert _scene(emotion=e).emotion == e


class TestGesture:
    @pytest.mark.parametrize("given,want", [
        ("nod", "talk"), ("standing", "idle"), ("running", "run_cycle"),
        ("leap", "jump"), ("chuckle", "laugh"), ("recoil", "react_shock"),
    ])
    def test_synonyms_map(self, given, want):
        assert Staging(x=0.5, gesture=given).gesture == want

    def test_valid_values_pass_through(self):
        for g in GESTURES:
            assert Staging(x=0.5, gesture=g).gesture == g


class TestShotAndCamera:
    @pytest.mark.parametrize("given,want", [
        ("close-up", "close_up"), ("closeup", "close_up"), ("establishing", "wide"),
        ("ots", "over_shoulder"), ("medium shot", "medium"),
    ])
    def test_shot_synonyms(self, given, want):
        assert _scene(shot=given).shot == want

    @pytest.mark.parametrize("given,want", [
        ("zoom in", "push_in"), ("dolly_out", "pull_out"), ("handheld", "handheld_drift"),
        ("fixed", "static"),
    ])
    def test_camera_move_synonyms(self, given, want):
        assert Camera(move=given).move == want

    def test_optional_to_shot_stays_none(self):
        # None means "no camera move" and must survive coercion, not become 'wide'.
        assert Camera(move="static", to_shot=None).to_shot is None

    def test_transition_synonyms(self):
        assert _scene(transition_in="crossfade").transition_in == "dissolve"
        assert _scene(transition_in="fade to black").transition_in == "fade_black"


class TestCoerceHelper:
    def test_compound_leads_with_a_usable_word(self):
        assert coerce("curious_and_worried", EMOTIONS, EMOTION_SYNONYMS, "neutral") == "curious"

    def test_non_string_falls_back(self):
        assert coerce(None, EMOTIONS, EMOTION_SYNONYMS, "neutral") == "neutral"
        assert coerce(7, EMOTIONS, EMOTION_SYNONYMS, "neutral") == "neutral"

    def test_no_synonym_shadows_a_real_value(self):
        # A synonym pointing at a word outside the vocabulary would coerce to something
        # the renderer cannot draw.
        for vocab, syn in ((EMOTIONS, EMOTION_SYNONYMS),):
            for src, dst in syn.items():
                assert dst in vocab, f"{src} -> {dst} is not in the vocabulary"
                assert src not in vocab, f"{src} is already valid; the synonym is dead code"


class TestEveryEnumFieldIsCoerced:
    """Catch the 'fixed one, missed its twin' failure directly.

    Scene.emotion was coerced and DialogueLine.emotion was not, so the very next run died
    on 'worried' in a dialogue line. Rather than trusting a hand-written list, walk the
    models and assert every field typed with a closed vocabulary has a before-validator.
    """

    def test_no_closed_vocabulary_field_is_left_raw(self):
        from asa.story import schema as sc

        expectations = [
            (sc.Scene, "emotion", "worried", "scared"),
            (sc.Scene, "shot", "closeup", "close_up"),
            (sc.Scene, "transition_in", "crossfade", "dissolve"),
            (sc.DialogueLine, "emotion", "worried", "scared"),
            (sc.Staging, "gesture", "nod", "talk"),
            (sc.Camera, "move", "zoom in", "push_in"),
            (sc.Camera, "from_shot", "closeup", "close_up"),
        ]
        required = {
            sc.Scene: dict(index=1, location_id="l", action="a",
                           visual_prompt="a river at night"),
            sc.DialogueLine: dict(character_id="c", line="hello"),
            sc.Staging: dict(x=0.5),
            sc.Camera: {},
        }
        for model, field, given, want in expectations:
            kwargs = dict(required[model])
            kwargs[field] = given
            got = getattr(model(**kwargs), field)
            assert got == want, f"{model.__name__}.{field}: {given!r} -> {got!r}, want {want!r}"


class TestCrossVocabularyWords:
    """Models put emotion words in the gesture field.

    A live run produced gesture="angry", "excited", "react_surprise" and "think". These are
    not typos to reject - the model is describing a pose it has no word for. Falling
    through to "idle" makes a furious character stand perfectly still, which is worse than
    an approximate pose.
    """

    @pytest.mark.parametrize("given,want", [
        ("angry", "point"), ("excited", "jump"), ("surprised", "react_shock"),
        ("react_surprise", "react_shock"), ("sad", "react_sad"), ("think", "idle"),
        ("happy", "laugh"), ("scared", "react_shock"),
    ])
    def test_emotion_words_map_to_a_physical_action(self, given, want):
        from asa.story.schema import Staging
        assert Staging(x=0.5, gesture=given).gesture == want

    @pytest.mark.parametrize("given,want", [
        ("relieved", "happy"), ("grateful", "happy"), ("desperate", "scared"),
        ("defiant", "determined"), ("suspicious", "curious"), ("lonely", "sad"),
    ])
    def test_observed_emotion_misses_now_map(self, given, want):
        assert _scene(emotion=given).emotion == want

    def test_every_gesture_synonym_targets_a_real_gesture(self):
        from asa.media.animation.shots import GESTURES, GESTURE_SYNONYMS
        for src, dst in GESTURE_SYNONYMS.items():
            assert dst in GESTURES, f"{src} -> {dst} is not a drawable gesture"
