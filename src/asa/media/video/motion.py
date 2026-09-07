"""What the character DOES while the line plays, in words a video model can act on.

`ShotJob.motion_prompt` is the only channel through which a hosted model learns anything
about the PERFORMANCE. Everything else it is handed - the still, the voice slice, the frame
count - describes the shot's shape. So if the prompt does not say that a hand comes up on
the accusation, no hand comes up: an image-to-video model animates what it is asked to
animate and otherwise just drifts the camera across the picture it was given.

Until now that channel carried `scenes.action`, truncated to 200 characters. Three things
were wrong with it and each one alone was enough to waste the shot:

  * one string per SCENE, so every shot of a six-line exchange asked for the same motion -
    the reverse angle got the same instruction as the angle it was cutting away from;
  * written as story prose, so it described the BEAT ("Kabir realises he has been lied to")
    rather than the body, and a video model cannot act on a realisation;
  * written in Hindi on a Hindi channel, which an English-trained video model has no useful
    response to - the same finding that made `images/scene_image._english_only` necessary,
    arriving here through a different door.

This module builds one prompt per SHOT out of the four things actually known about it: who
is on screen and what they look like, whether they are the one speaking, how the line is
delivered, and how tightly it is framed. Framing carries as much weight as emotion, because
a gesture has to fit the frame - hands asked for inside an extreme close-up come back as
fingers across the lens.

Nothing here is a prompt for a still. The image stage already bought the picture; this
describes the seconds after it.
"""
from __future__ import annotations

from ..images.scene_image import _english_only, character_look

# How much of a body the framing leaves room for, which decides what a gesture may be asked
# to do. `insert` is its own case: there is no character in it to perform.
_SCALE = {
    "extreme_close_up": "face",
    "close_up": "face",
    "over_shoulder": "face",
    "medium": "hands",
    "two_shot": "body",
    "full": "body",
    "wide": "body",
    "insert": "detail",
}

# The speaking performance, per (scale, emotion bucket). Written as observable body, never
# as interior state: "jaw tight, head pushing forward" is directable, "angry" is not.
#
# Every entry names the mouth first. A model handed a still and a motion prompt will animate
# the listed parts and leave the rest of the frame alone, so a prompt that opens on the
# hands returns gesturing with a closed mouth - which is the failure this whole path exists
# to remove.
_SPEAKING = {
    ("face", "warm"): ("speaking, mouth moving naturally with the dialogue, eyes bright, "
                       "brows lifting, small nods on the stressed words"),
    ("face", "tense"): ("speaking, mouth moving naturally with the dialogue, jaw tight, "
                        "brows drawn low, head pushing forward on the hard words"),
    ("face", "neutral"): ("speaking, mouth moving naturally with the dialogue, steady gaze, "
                          "small head movements between phrases, one slow blink"),
    ("hands", "warm"): ("speaking, mouth moving with the dialogue, open palms lifting and "
                        "turning over as the sentence lands, shoulders loose"),
    ("hands", "tense"): ("speaking, mouth moving with the dialogue, one hand cutting down "
                         "through the air, finger jabbing forward, shoulders squared"),
    ("hands", "neutral"): ("speaking, mouth moving with the dialogue, one hand rising to "
                           "gesture and settling again, weight shifting"),
    ("body", "warm"): ("speaking and gesturing with both hands, stepping in, turning to "
                       "face the others, shoulders open"),
    ("body", "tense"): ("speaking, striding forward, one arm thrown out to the side, "
                        "pointing at the other character, the others turning to look"),
    ("body", "neutral"): ("speaking and gesturing, moving through the space, the others "
                          "turning as the line lands"),
}

# A shot that has a speaker but nothing of them to perform with. `shotlist._fallback_key`
# spends a scene past its image budget by dropping a shot onto a setup already bought, and
# the nearest one is sometimes a narration insert - so a line of dialogue occasionally plays
# over a tight detail with no character in it at all. Asking for a mouth there does not get
# a mouth, it gets one hallucinated onto the object.
_NO_SUBJECT = "no character in frame, movement within the detail itself"

# A NARRATION shot, which is a different thing again and was the one the listener wording
# got wrong. `plan_shots` gives narration no speaker and frames it wide, full or insert:
# there is no one on screen being talked to, so "heads turning toward the speaker" invents a
# speaker the frame does not contain. What a narration shot wants is the place doing what
# the place does while the voice-over runs - which is also the only kind of shot in an
# episode where `scenes.action` is the subject rather than the context, so it is promoted
# ahead of the performance clause in `shot_motion_prompt`.
_NARRATION = {
    "body": ("no dialogue in frame, the location alive - figures moving through it at their "
             "own pace, cloth and foliage stirring, light and shadow shifting"),
    "detail": _NO_SUBJECT,
}

# Camera, in the language a video model uses rather than the pixel language `camera.py`
# uses. The two are the same closed vocabulary (`camera.CAMERA_MOVES`) so a shot moves the
# same direction whichever provider renders it - a hosted clip that pans the opposite way
# from its local fallback makes the cut between them visible.
_CAMERA = {
    "static": "locked-off camera, no camera movement",
    "push_in": "camera pushes in slowly",
    "pull_out": "camera pulls back slowly",
    "pan_left": "camera pans left",
    "pan_right": "camera pans right",
    "tilt_up": "camera tilts up",
    "tilt_down": "camera tilts down",
    "handheld_drift": "handheld camera, slight natural drift",
}

# Two directives that are not description and are not optional.
#
# ONE CONTINUOUS TAKE, because the shot list above this module has already decided where the
# cuts go. A model that invents a cut three seconds into a four-second clip does not add a
# cut, it destroys one: the edit lands mid-gesture on a character the scene had not
# introduced yet, and the clip cannot be trimmed back into shape because both halves are
# short.
#
# NO TEXT, for the reason `CINEMATIC_NEGATIVE` carries it - a generative model asked for a
# lorry will letter the side of it, and the lettering is always garbled.
_CONTINUITY = ("one continuous take, no cuts, no scene change, consistent character design "
               "and clothing throughout, no on-screen text or subtitles")


def _bucket_of(shot) -> str:
    # Imported inside the call rather than at module scope: `assemble.shotlist` is a level
    # ABOVE this package and `assemble.shot_render` imports `media.video`, so a top-level
    # import here closes the loop. The bucketing itself must not be duplicated - a shot
    # whose image was bought as "tense" and whose motion is prompted as "neutral" is a
    # picture and a performance that disagree.
    from ...assemble.shotlist import emotion_bucket
    return emotion_bucket(getattr(shot, "emotion", None))


# The medium, in the fewest words that hold it. A hosted video model is conditioned on the
# input still, so most of the look arrives with the picture - but not all of it: an i2v
# model given a stylised frame and a prompt that never says what it is drifts toward the
# photoreal mean over a five-second clip, and the drift shows up as a cut, because the shot
# either side of it did not drift the same way. One clause per preset, placed LAST with the
# continuity rules rather than first, because it is a guard rail and not the subject.
_MEDIUM = {
    "cartoon": "3D animated cartoon, stylised animal character, consistent cartoon style",
    "photoreal": "photorealistic live-action film, consistent photoreal style",
}


def medium_clause(style: str | None) -> str:
    """The style anchor for a preset, or "" for a preset that has none - an unknown name is
    not an error here the way it is in `scene_image.preset`, because a missing guard rail
    must not fail a render that is otherwise fine."""
    return _MEDIUM.get(str(style or "").strip().lower(), "")


def shot_motion_prompt(scene: dict, shot, cast_by_id: dict | None = None,
                       *, limit: int = 520, art_style: str | None = None) -> str:
    """One shot's performance, as a single English instruction.

    Ordered subject -> action -> camera -> continuity, because that is the order of
    decreasing attention every text-conditioned video model gives a prompt: what is in
    frame has to arrive before what it does, and the continuity clause is a guard rail
    rather than something to spend weight on.
    """
    scale = _SCALE.get(str(getattr(shot, "framing", "") or "").lower(), "body")
    speaking = bool(getattr(shot, "speaking", False)) and scale != "detail"
    parts: list[str] = []

    # Identity first, and taken from the SAME sentence the image prompt used. An image model
    # has no memory between calls and neither does a video model: describing the fox a
    # second way here is how a character changes breed halfway through its own line.
    speaker_id = getattr(shot, "speaker", None)
    look = ""
    if speaker_id and cast_by_id:
        record = cast_by_id.get(speaker_id)
        if record:
            look = character_look(record)
    if look:
        parts.append(look if speaking else f"{look}, in frame")

    # The scene's own action, kept because it is the only thing here that knows a lorry is
    # being unloaded in the rain. Dropped rather than translated when it is not English -
    # a paragraph of Devanagari does not steer the model, it only crowds out the clauses
    # that do.
    action = _english_only(scene.get("action"), 120)
    narration = speaker_id is None

    if speaking:
        parts.append(_SPEAKING[(scale, _bucket_of(shot))])
    elif narration:
        # Action first here: with nobody on screen to perform, it IS the subject.
        if action:
            parts.append(action)
            action = ""
        parts.append(_NARRATION.get(scale, _NARRATION["body"]))
    else:
        parts.append(_NO_SUBJECT)

    if action:
        parts.append(action)

    parts.append(_CAMERA.get(str(getattr(shot, "camera_move", "") or "").lower(),
                             _CAMERA["static"]))
    parts.append(_CONTINUITY)
    parts.append(medium_clause(art_style))
    return ", ".join(p for p in parts if p)[:limit]
