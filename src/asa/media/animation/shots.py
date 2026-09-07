"""Closed vocabularies the LLM selects from. Nothing here is free text."""
from __future__ import annotations

from ...core.logging import get_logger

log = get_logger("shots")

CAMERA_MOVES = ["static", "push_in", "pull_out", "pan_left", "pan_right",
                "tilt_up", "tilt_down", "handheld_drift"]
SHOT_TYPES = ["wide", "full", "medium", "close_up", "extreme_close_up",
              "two_shot", "over_shoulder", "insert"]
GESTURES = ["idle", "talk", "point", "wave", "shrug", "jump",
            "run_cycle", "walk_cycle", "sit", "react_shock", "react_sad", "laugh"]
TRANSITIONS = ["cut", "dissolve", "fade_black", "whip_pan", "iris"]
EMOTIONS = ["neutral", "curious", "happy", "sad", "scared", "angry",
            "surprised", "determined", "wry", "excited"]
EASING = ["linear", "in", "out", "in_out"]

# Synonyms the models reach for that are not in the closed vocabularies above.
#
# These lists are what the LLM must choose from, and small free-tier models routinely
# return a near-miss instead: "calm" for neutral, "worried" for scared, "nod" for talk.
# One such word failed an entire story - the schema rejected it, the repair pass produced a
# different near-miss, and the job died after art, voice and rendering would have been
# paid for. Mapping the obvious synonyms is far better than either widening the vocabulary
# (the renderer has no artwork for "calm") or letting a story die over one adjective.
#
# Anything genuinely unmappable still falls back to the field default, so an unknown word
# degrades to a neutral pose rather than failing.
EMOTION_SYNONYMS = {
    "calm": "neutral", "neutral_calm": "neutral", "content": "happy", "joyful": "happy",
    "cheerful": "happy", "delighted": "happy", "amused": "wry", "playful": "wry",
    "wistful": "sad", "melancholy": "sad", "sorrowful": "sad", "upset": "sad",
    "worried": "scared", "anxious": "scared", "nervous": "scared", "afraid": "scared",
    "frightened": "scared", "alarmed": "scared", "tense": "scared",
    "angry_frustrated": "angry", "frustrated": "angry", "annoyed": "angry",
    "furious": "angry", "irritated": "angry",
    "astonished": "surprised", "shocked": "surprised", "amazed": "surprised",
    "startled": "surprised", "stunned": "surprised",
    "resolute": "determined", "focused": "determined", "brave": "determined",
    "confident": "determined", "hopeful": "determined", "proud": "determined",
    "eager": "excited", "enthusiastic": "excited", "thrilled": "excited",
    "inquisitive": "curious", "puzzled": "curious", "confused": "curious",
    "thoughtful": "curious", "intrigued": "curious", "wondering": "curious",
    "sarcastic": "wry", "dry": "wry", "smug": "wry",
    # Observed in a live run falling through to the default.
    "relieved": "happy", "grateful": "happy", "warm": "happy", "fond": "happy",
    "ashamed": "sad", "guilty": "sad", "lonely": "sad", "disappointed": "sad",
    "desperate": "scared", "panicked": "scared", "uneasy": "scared",
    "annoyed_frustrated": "angry", "indignant": "angry", "defiant": "determined",
    "suspicious": "curious", "cautious": "curious", "uncertain": "curious",
}
GESTURE_SYNONYMS = {
    "nod": "talk", "speak": "talk", "speaking": "talk", "talking": "talk",
    "gesture": "talk", "explain": "talk",
    "stand": "idle", "standing": "idle", "still": "idle", "none": "idle",
    "wait": "idle", "watch": "idle", "look": "idle", "listen": "idle",
    "pointing": "point", "gesture_point": "point", "indicate": "point",
    "waving": "wave", "greet": "wave", "beckon": "wave",
    "shrugging": "shrug", "running": "run_cycle", "run": "run_cycle",
    "walking": "walk_cycle", "walk": "walk_cycle", "approach": "walk_cycle",
    "sitting": "sit", "crouch": "sit", "rest": "sit", "settle": "sit",
    "jumping": "jump", "leap": "jump", "hop": "jump", "fly": "jump",
    "shock": "react_shock", "recoil": "react_shock", "flinch": "react_shock",
    "gasp": "react_shock", "startle": "react_shock",
    "sad": "react_sad", "droop": "react_sad", "slump": "react_sad", "cry": "react_sad",
    "laughing": "laugh", "chuckle": "laugh", "giggle": "laugh", "smile": "laugh",
    # Models put EMOTION words in the gesture field - a live run produced "angry",
    # "excited" and "relieved" as gestures. These are not typos to be rejected; the model
    # is describing a pose it has no word for. Mapping them to the nearest physical action
    # keeps the performance, where falling through to "idle" makes a furious character
    # stand perfectly still.
    "angry": "point", "furious": "point", "annoyed": "shrug", "confused": "shrug",
    "excited": "jump", "happy": "laugh", "delighted": "laugh", "joyful": "laugh",
    "surprised": "react_shock", "react_surprise": "react_shock",
    "shocked": "react_shock", "afraid": "react_shock", "scared": "react_shock",
    "sad": "react_sad", "crying": "react_sad", "upset": "react_sad",
    "relieved": "idle", "calm": "idle", "neutral": "idle",
    "think": "idle", "thinking": "idle", "ponder": "idle", "consider": "idle",
    "curious": "point", "point_at": "point", "reach": "point", "grab": "point",
}
SHOT_SYNONYMS = {
    "closeup": "close_up", "close": "close_up", "cu": "close_up",
    "extreme_closeup": "extreme_close_up", "ecu": "extreme_close_up",
    "extreme_wide": "wide", "establishing": "wide", "long": "wide", "wide_shot": "wide",
    "full_shot": "full", "medium_shot": "medium", "mid": "medium", "ms": "medium",
    "two": "two_shot", "twoshot": "two_shot", "ots": "over_shoulder",
    "over_the_shoulder": "over_shoulder", "detail": "insert", "cutaway": "insert",
}
CAMERA_SYNONYMS = {
    "none": "static", "fixed": "static", "locked": "static", "still": "static",
    "zoom_in": "push_in", "dolly_in": "push_in", "push": "push_in", "in": "push_in",
    "zoom_out": "pull_out", "dolly_out": "pull_out", "pull": "pull_out", "out": "pull_out",
    "pan": "pan_right", "pan_r": "pan_right", "pan_l": "pan_left",
    "tilt": "tilt_up", "handheld": "handheld_drift", "shake": "handheld_drift",
    "drift": "handheld_drift",
}
TRANSITION_SYNONYMS = {
    "hard_cut": "cut", "none": "cut", "straight_cut": "cut",
    "crossfade": "dissolve", "fade": "dissolve", "mix": "dissolve",
    "fade_to_black": "fade_black", "fade_out": "fade_black", "blackout": "fade_black",
    "whip": "whip_pan", "swish_pan": "whip_pan", "iris_in": "iris", "iris_out": "iris",
}


def coerce(value, allowed: list[str], synonyms: dict[str, str], default: str) -> str:
    """Map a model's near-miss onto the closed vocabulary, or fall back to `default`.

    A fallback is LOGGED, not silent. Coercion replaced a hard rejection here, and a
    rejection at least made the problem visible - swapping it for a quiet default would
    trade a loud failure for an invisible one, which is the same trap as config that is
    declared and never read. A synonym hit is expected and stays quiet; only genuinely
    unmappable input is worth a line.
    """
    if not isinstance(value, str):
        return default
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    if v in synonyms:
        return synonyms[v]
    # A compound like "curious_and_worried" usually leads with a usable word.
    head = v.split("_")[0]
    if head in allowed:
        return head
    if head in synonyms:
        return synonyms[head]
    log.info("vocabulary_fallback", value=str(value)[:40], chose=default,
             allowed=len(allowed))
    return default
