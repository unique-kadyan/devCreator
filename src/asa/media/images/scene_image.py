"""Whole-scene image prompts: characters and setting generated together.

The puppet path draws characters procedurally and composites them onto an empty plate, so
its background prompt deliberately forbids figures ("not a single person or creature
anywhere"). This module is the opposite: one photorealistic frame per scene containing the
cast, their clothes and the real-world location, because a procedural rig cannot reach the
look this channel wants - anthropomorphic animals in school uniforms and suits, in Indian
classrooms, streets and cars.

The hard problem is CONSISTENCY. An image model has no memory between calls, so the same
fox must be described identically in every scene or it changes breed, colour and clothing
shot to shot. `character_look` builds one stable sentence per character from its stored
record and every scene prompt reuses it verbatim.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Two strings, deliberately split, because they answer different questions and only one of
# them is a matter of taste.
#
# CINEMATIC_STYLE is what the subject IS - the medium, the lens, and the construction of an
# anthro character. "a real animal head and a human body" was read loosely: measured on the
# 2026-09-04 test render, the fox came back with digitigrade animal legs and a tail longer
# than its torso, which reads as a mascot rather than as the reference's characters. The
# reference's anthros are adult human bodies - shoulders, biceps, trousers, shoes - wearing
# an animal head, so the body is now described as explicitly as the head is.
CINEMATIC_STYLE = (
    "photorealistic cinematic film still, anthropomorphic animal character with a real "
    "animal head on a full adult human body, human proportions and musculature, "
    "broad shoulders, human arms and hands, straight human legs in trousers, "
    "wearing real tailored clothing, natural fur detail, "
    "shot on 35mm lens, shallow depth of field, high detail")

# CINEMATIC_LOOK is the GRADE, and it is the one thing separating this channel's output
# from the video it is tuned against. That is not an impression; it was measured across
# both on 2026-09-05, sampling a frame every six seconds:
#
#                      mean luma (0-255)   pixels below luma 40
#   reference                       21.1                  85.0%
#   our test render                 92.1                  20.2%
#   job_00010 episode               86.0                  26.4%
#
# The reference is a night film - four fifths of every frame is in shadow and the light is
# a hard key plus practicals. Ours was four times too bright because the old style string
# asked for "soft natural light, colour-graded", which is not a look at all: an image model
# reads it as the daylight default and returns the flat, evenly-lit picture that made the
# output read as stock art with a voice over it.
#
# Overridable from `channel.look_hint`, because this is the taste half and it belongs where
# the operator can turn it without editing a module.
CINEMATIC_LOOK = (
    "low-key night cinematography, single hard key light with practical lamps and "
    "vehicle headlights, deep crushed shadows filling most of the frame, "
    "rain-wet reflective ground, atmospheric haze, "
    "desaturated teal shadows against warm amber highlights, heavy contrast, "
    "fine film grain")


# ------------------------------------------------------------------ style presets
#
# The three strings above are ONE look - the photoreal night film this channel was first
# tuned against. They are kept as module constants because two modules and three tests
# import them by name, and because that look is still a legitimate choice.
#
# But they are not the only choice, and which of them is in force is a channel decision
# rather than a property of the cinematic renderer. A cartoon channel wants the same
# machinery - one frame per camera setup, containing the cast, cut and performed - and a
# completely different set of words in the prompt. Crucially it wants a different NEGATIVE
# too: CINEMATIC_NEGATIVE ends with "cartoon, illustration, drawing, anime, 3d render,
# cgi", so a cartoon look asked for through the photoreal negative is a prompt arguing with
# itself, and the model settles it differently on every generation. That is the same
# art-style drift `_PUPPET_STYLE_WORDS` exists to prevent, arriving from the other side.
#
# So a preset is the triple (subject, look, negative), all three switched together.

@dataclass(frozen=True)
class StylePreset:
    """What the subject is, how it is lit, and what must never be generated."""
    subject: str
    look: str
    negative: str


# The cartoon preset, and every clause in it is load-bearing.
#
#   * ANIMAL FACE, STATED AS A FACE. The whole point of this channel is a talking animal,
#     and a lip-sync model needs a mouth it can find: "expressive animal face, large
#     readable eyes, a clearly defined mouth and muzzle" is what puts one in frame. The
#     photoreal preset never had to say this because a photoreal muzzle is drawn by
#     default; a stylised one is drawn only if asked for, and a cartoon fox with a tiny
#     dot mouth gives the hosted video model nothing to animate.
#   * HUMAN BODY, same as photoreal and for the same measured reason - the model's default
#     for "cartoon fox" is a four-legged mascot.
#   * NO STUDIO NAMED. Naming a studio is the fastest way to this look and it is off the
#     table: `prompts/_blocks/safety_rules.md` forbids evoking an existing studio, and that
#     rule is not suspended because the request happens to be going to an image model
#     rather than to the writer.
CARTOON_STYLE = (
    "modern 3D animated feature film still, stylised cartoon anthropomorphic animal "
    "character with an expressive animal face, large readable eyes, a clearly defined "
    "mouth and muzzle, appealing character design, "
    "full adult human body with human proportions, human arms and hands, "
    "straight legs in trousers, wearing real tailored clothing, "
    "soft stylised fur shading, clean readable silhouette, subsurface scattering, "
    "high quality character animation render, sharp focus")

# The cartoon GRADE. Deliberately the opposite of CINEMATIC_LOOK: that one is a night film
# where four fifths of the frame is in shadow, which is exactly wrong here. A cartoon face
# has to be READ at feed size on a phone, so the light is high-key and the palette is
# saturated - the face carries the performance and a face in crushed shadow carries none.
CARTOON_LOOK = (
    "bright high-key lighting, warm key light with a clean rim light separating the "
    "character from the background, vivid saturated colour palette, "
    "soft shadows, cheerful inviting mood, glossy polished render, "
    "shallow depth of field with a softly blurred background")

# Same three groups as CINEMATIC_NEGATIVE - artefacts, mascot proportions, wrong medium -
# with the third group INVERTED. Photoreal excludes cartoon; cartoon excludes photoreal.
# The daylight group is dropped entirely: this preset asks for bright light on purpose.
CARTOON_NEGATIVE = (
    "text, words, letters, lettering, watermark, signature, logo, subtitles, caption, "
    "extra limbs, extra fingers, fused fingers, deformed hands, malformed anatomy, "
    "human face, human head, humanoid face, mask, costume, cosplay, fursuit, "
    "quadruped, animal legs, digitigrade legs, paws for feet, oversized tail, "
    "chibi, plush toy, sticker, clipart, "
    "photorealistic, photograph, photo, live action, hyperreal, uncanny, "
    "flat 2d vector, line art, sketch, unfinished, "
    "lowres, blurry, jpeg artifacts, muddy colours, dark murky shadows")

def aspect_hint(size: tuple[int, int]) -> str:
    """The frame shape, in the words an image model understands.

    This used to be the literal string "16:9" welded onto the end of CINEMATIC_STYLE, which
    was silently wrong the moment `production.resolution` became portrait: the provider was
    handed a 768x1344 canvas while the prompt asked for widescreen, so the model spent its
    composition budget fighting the frame it had been given. The aspect is a property of the
    configured output, so it is derived from it rather than written down twice.
    """
    w, h = size
    ratio = w / h if h else 1.0
    if ratio > 1.2:
        return "16:9 widescreen"
    if ratio < 0.83:
        return "9:16 vertical, portrait composition"
    return "1:1 square"


def styled_for(size: tuple[int, int], look: str | None = None,
               style: str | None = None) -> str:
    """The full style clause: what the subject is, how it is lit, and the frame shape.

    `style` names a preset (`channel.art_style`); `look` overrides that preset's grade so a
    channel can be regraded from config without leaving its medium. Passing an empty `look`
    drops the grade entirely and leaves the model to its default - which is what the output
    looked like before CINEMATIC_LOOK existed.
    """
    chosen = preset(style)
    grade = chosen.look if look is None else look
    return ", ".join(p for p in (chosen.subject, grade, aspect_hint(size)) if p)

# Deliberately NOT forbidding people or animals - they are the subject here. What must be
# excluded is the failure modes of photoreal generation: garbled on-image text, extra or
# fused limbs, and the uncanny half-human faces the model produces when "anthropomorphic"
# is read as "human with fur".
# Three groups, and the last two were added on 2026-09-05 against measured failures.
#
#   * generation artefacts - garbled on-image text, extra limbs, the uncanny half-human
#     face the model produces when "anthropomorphic" is read as "human with fur";
#   * MASCOT PROPORTIONS. The test render returned a fox with digitigrade animal legs, a
#     tail longer than its body and a head two sizes too big. CINEMATIC_STYLE now asks for
#     a human body positively; these say what that body is not, because the model's default
#     for "anthropomorphic fox" is a furry mascot and a positive clause alone did not beat
#     it;
#   * DAYLIGHT. CINEMATIC_LOOK asks for a night grade and this stops the model reverting to
#     the evenly-lit stock-photo default it was producing - measured at four times the
#     reference's brightness. Naming the lighting to avoid is what makes the positive grade
#     hold; without it the two clauses average out to an overcast afternoon.
CINEMATIC_NEGATIVE = (
    "text, words, letters, lettering, watermark, signature, logo, subtitles, caption, "
    "extra limbs, extra fingers, fused fingers, deformed hands, malformed anatomy, "
    "human face, human head, humanoid face, mask, costume, cosplay, fursuit, "
    "quadruped, animal legs, digitigrade legs, paws for feet, oversized tail, "
    "oversized head, chibi, mascot, plush toy, "
    "bright daylight, midday sun, flat even lighting, overcast, washed out, "
    "cartoon, illustration, drawing, anime, 3d render, cgi, plastic, "
    "lowres, blurry, jpeg artifacts, oversaturated")

PRESETS: dict[str, StylePreset] = {
    "photoreal": StylePreset(CINEMATIC_STYLE, CINEMATIC_LOOK, CINEMATIC_NEGATIVE),
    "cartoon": StylePreset(CARTOON_STYLE, CARTOON_LOOK, CARTOON_NEGATIVE),
}

# `photoreal` rather than `cartoon`, because this is the value every existing caller gets
# when it passes nothing, and silently regrading a running channel is not a default's job.
# `channel.art_style` in config is what actually decides it.
DEFAULT_PRESET = "photoreal"


def preset(name: str | None) -> StylePreset:
    """The named preset, or the default. An unknown name is a config typo and says so
    rather than quietly rendering the wrong channel."""
    key = (name or DEFAULT_PRESET).strip().lower()
    try:
        return PRESETS[key]
    except KeyError:
        raise ValueError(
            f"channel.art_style {name!r} is not a style preset. "
            f"Choose one of: {', '.join(sorted(PRESETS))}") from None


def negative_for(style: str | None = None) -> str:
    """The negative prompt belonging to a preset. Always fetched alongside `styled_for` -
    a subject from one preset and a negative from another cancel each other out."""
    return preset(style).negative

# Real-world grounding. The channel is Indian, so "street" means an Indian street rather
# than the generic western default the model falls back to.
REGION_HINT = ("set in India, authentic Indian surroundings, Indian signage shapes, "
               "Indian street furniture")


def period_hint(period: str | None) -> str:
    """The grounding clause for a story set somewhere other than here and now.

    This exists because `REGION_HINT` / `channel.region_hint` is unconditional, and it is
    right almost all the time: the channel is Indian and "a street" otherwise renders as a
    generic western one. It is wrong exactly when an episode is about something that
    happened elsewhere, in another century - it was putting Indian street furniture in
    Alexandria. So a story's `period` REPLACES the region hint rather than joining it; the
    two cannot both be true, and an image model handed both averages them into a place that
    never existed.

    The "nothing modern" clause is not padding. Asked for an ancient workshop, image models
    reliably add power lines, wristwatches, printed labels and plate glass - the anachronism
    is the default failure of the historical prompt, the way mascot proportions are the
    default failure of the anthro prompt, and it has to be named to be beaten.

    Returns "" for an empty period, which is what "present day" means and what leaves the
    caller's region hint in force.
    """
    period = " ".join(str(period or "").split())
    if not period:
        return ""
    return (f"set in {period}, "
            "historically accurate architecture, clothing, tools and materials for that "
            "exact time and place, "
            "no modern objects anywhere in frame, no electric lighting, no power lines, "
            "no printed text, no plastic, no modern clothing, no wristwatches")

_SHOT_FRAMING = {
    "wide": "wide establishing shot, full environment visible",
    "full": "full body shot",
    "medium": "medium shot from the waist up",
    "close_up": "close-up on the face",
    "extreme_close_up": "extreme close-up",
    "two_shot": "two shot, both characters in frame",
    "over_shoulder": "over-the-shoulder shot",
    "insert": "tight insert shot of a detail",
}


def _clean(text: str, limit: int = 160) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".")[:limit]


# Vocabulary that belongs to the PUPPET path and must never reach a cinematic prompt.
#
# `scenes.visual_prompt` is written for the puppet renderer, which wants an empty background
# plate in a flat storybook style. Every row therefore ends with something like "storybook
# flat-vector illustration, clean bold shapes, muted natural palette, ... no characters, no
# people, no animals" - and the cinematic prompt then appends "photorealistic cinematic film
# still, anthropomorphic animal character with a real animal head". The two instructions
# contradict each other outright, so the image model obeys whichever it likes per
# generation. That is the art-style drift: within one scene, some shots came back as flat 2D
# illustration and others as photoreal 3D, and the "no characters" clause occasionally won
# and returned an empty room.
_PUPPET_STYLE_WORDS = (
    "storybook", "flat-vector", "flat vector", "illustration", "painterly", "cel-shaded",
    "bold shapes", "palette", "rim light", "depth-of-field", "depth of field",
    "ambient haze", "outlines", "texture", "watermark", "vector",
    "no characters", "no people", "no animals", "no text", "no lettering",
    "no signage", "no figures",
)


def _scene_description(text: str, limit: int = 200) -> str:
    """The CONTENT of a scene's visual prompt, with its puppet-era styling stripped out.

    Clause by clause, because the description and the style directions are comma-separated
    siblings in the same string and only the description is wanted here - the look comes
    from CINEMATIC_STYLE, which is the whole point of the cinematic path.

    Never returns empty: if every clause looked like styling, the original is more useful
    than nothing.
    """
    clauses = [c.strip() for c in str(text or "").split(",")]
    kept = [c for c in clauses
            if c and not any(w in c.lower() for w in _PUPPET_STYLE_WORDS)]
    return _clean(", ".join(kept) or str(text or ""), limit)


def _ascii_ratio(text: str) -> float:
    return (sum(c.isascii() for c in text) / len(text)) if text else 1.0


def _english_only(text: str, limit: int = 160) -> str:
    """Drop text the image model cannot read.

    A Hindi channel writes `scenes.action` in Hindi, and it was being pasted straight into
    the FLUX prompt - a paragraph of Devanagari that the model has no useful response to,
    crowding out the parts of the prompt that do steer it. The visual_prompt is required to
    be English (story/prompts.py LANGUAGE_RULES); the action is prose and is not.
    """
    text = _clean(text, limit)
    return text if _ascii_ratio(text) > 0.9 else ""


def _accessories(value) -> str:
    """`characters.accessories` is a JSON array column, so it arrives as the STRING "[]"
    for a character with none - which `_clean` happily passed through, putting the literal
    text "with []" into every image prompt for most of the cast."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return _clean(value, 60)
    if isinstance(value, (list, tuple)):
        return _clean(", ".join(str(v) for v in value if v), 60)
    return _clean(value, 60)


# Values the story model writes when a character HAS no such feature. They are not
# descriptions and must never be pasted into one: a boar stored with clothing "none" was
# being asked for as "wearing none", which an image model reads as an instruction about
# what to wear rather than an absence of clothing.
_ABSENT = {"none", "no", "nil", "null", "n/a", "na", "-", "--", "unknown", "नहीं",
           "कोई नहीं", "कुछ नहीं"}


def _described(value, limit: int) -> str:
    """A stored field, or "" when it does not describe anything an image model can draw.

    Two filters, and both were measured on live cast records rather than guessed at:

      * the absence words above, which arrive as ordinary strings and read as content;
      * non-English prose. A Hindi channel stores `appearance` in the story's language -
        one boar reads "भारी कद, धूसर फर, बड़े नाखून" - and FLUX has no useful response to
        Devanagari. It does not merely ignore it either: it is a third of the prompt's
        budget spent saying nothing, crowding out the species and clothing clauses that
        actually steer the generation. Same finding as `_english_only`, reached from the
        cast table instead of the scene table.

    Dropping is right rather than translating: the species, age and framing still describe
    a drawable character, and a silent fallback beats a mistranslated one that quietly
    changes what the animal looks like between episodes.
    """
    text = _clean(value, limit)
    if not text or text.strip().lower() in _ABSENT:
        return ""
    return _english_only(text, limit)


def character_look(c: dict) -> str:
    """One stable appearance sentence for a character, reused in every scene prompt.

    Built from the stored record rather than re-invented per scene: an image model has no
    memory between calls, so any variation in wording produces a visibly different animal.
    """
    bits = [f"an anthropomorphic {_clean(c.get('species'), 40)}"]
    age = (c.get("age_band") or "").replace("_", " ")
    if age and age != "young adult":
        bits.append(f"{age}")
    appearance = _described(c.get("appearance"), 120)
    if appearance:
        bits.append(appearance)
    clothing = _described(c.get("clothing"), 90)
    bits.append(f"wearing {clothing}" if clothing else "wearing simple everyday clothes")
    accessories = _described(_accessories(c.get("accessories")), 60)
    if accessories:
        bits.append(f"with {accessories}")
    return ", ".join(b for b in bits if b)


def scene_prompt(scene: dict, cast_by_id: dict[str, dict], *,
                 region_hint: str = REGION_HINT, style: str = CINEMATIC_STYLE) -> str:
    """The full image prompt for one scene: who is in it, doing what, where, framed how."""
    present = [cast_by_id[cid] for cid in scene_characters(scene) if cid in cast_by_id]
    if not present:
        # A scene with no staged cast is an establishing shot of the place itself.
        who = "no characters, empty location"
    else:
        who = " and ".join(character_look(c) for c in present[:3])

    framing = _SHOT_FRAMING.get(scene.get("shot") or "medium", _SHOT_FRAMING["medium"])
    parts = [
        who,
        # Same treatment as shot_prompt: the puppet path's flat-storybook styling and any
        # non-English prose must not reach a photoreal generation.
        _english_only(scene.get("action"), 200),
        _scene_description(scene.get("visual_prompt"), 200),
        framing,
        region_hint,
        style,
    ]
    return ". ".join(p for p in parts if p)


# ---------------------------------------------------------------------- shots
#
# A SHOT prompt differs from a scene prompt in what it is a picture of. A scene prompt
# describes a moment: who is there, what is happening, where. A shot prompt describes a
# CAMERA SETUP - one subject, one framing, composed so the renderer knows where the face
# is without looking for it (see media/animation/face.py). The composition clauses below
# are load-bearing, not decoration: "head and shoulders fill the frame, face centred" is
# what makes a fixed head-box prior correct often enough to animate against.

_SHOT_COMPOSITION = {
    # Prescriptive on purpose. "close-up portrait, head and shoulders fill the frame" was
    # obeyed loosely: a lion filled the frame, an owl came back half the size, and a fixed
    # head-box prior cannot be right for both - the jaw band landed on the owl's scarf.
    # Naming the head's edges instead ("top edge to bottom edge, chin near the bottom,
    # eyes in the upper third") put four different species in the same place, which is
    # what makes media/animation/face.py's prior correct without a detector.
    "close_up": ("extreme close-up portrait of the head, the head fills the entire frame "
                 "from the top edge to the bottom edge, chin near the bottom of frame, "
                 "eyes in the upper third, facing camera"),
    # Genuinely tighter than `close_up`, and it has to be. These two were byte-identical,
    # which meant identical prompts, which meant the prompt cache served ONE picture under
    # two filenames - so cutting between them changed nothing on screen. That silently
    # defeats the whole point of covering a long speech with several setups.
    "extreme_close_up": ("extreme close-up on the face, the eyes and the muzzle filling the "
                         "frame, cropped just above the eyes and just below the chin, "
                         "facing camera"),
    "medium": ("medium shot from the waist up, subject centred, facing camera"),
    # The listener must be named as an animal too. Asked only for "the listener's
    # shoulder", the model puts a HUMAN in the foreground - which in an all-animal world
    # breaks the premise in a single frame, and the negative prompt cannot catch it
    # because the back of a head is not a face.
    "over_shoulder": ("over-the-shoulder shot, the speaking character facing camera and "
                      "centred, the blurred shoulder of another anthropomorphic animal "
                      "in the foreground, no humans"),
    "two_shot": "two shot, both characters facing each other in profile",
    "wide": "wide establishing shot of the location, the characters small in the frame",
    "full": "full body shot, the whole character visible, environment around them",
    "insert": "tight insert shot of a detail of the location, no faces",
}

# Emotion reaches the image as a face, not as a mood word: "sad" in a prompt tends to grade
# the whole picture blue, where "downturned mouth" changes the performance.
_EMOTION_FACE = {
    "warm": "warm relaxed expression, eyes bright",
    "tense": "tense expression, brow drawn, eyes hard",
    "neutral": "calm attentive expression",
}

# A mouth caught mid-word gives the animator something to move. A closed, resting mouth
# animated open looks like a hinge; a mouth already parted animates as speech.
SPEAKING_CLAUSE = ("mouth or beak open mid-speech, talking, teeth and tongue partly visible")


def shot_prompt(scene: dict, cast_by_id: dict[str, dict], *, speaker: str | None,
                framing: str, emotion_bucket: str = "neutral", speaking: bool = False,
                region_hint: str = REGION_HINT, style: str = CINEMATIC_STYLE) -> str:
    """The image prompt for ONE camera setup within a scene."""
    composition = _SHOT_COMPOSITION.get(framing, _SHOT_COMPOSITION["medium"])
    parts: list[str] = []

    if speaker and speaker in cast_by_id:
        parts.append(character_look(cast_by_id[speaker]))
        parts.append(composition)
        if speaking:
            parts.append(SPEAKING_CLAUSE)
        parts.append(_EMOTION_FACE.get(emotion_bucket, _EMOTION_FACE["neutral"]))
    else:
        present = [cast_by_id[cid] for cid in scene_characters(scene) if cid in cast_by_id]
        parts.append(" and ".join(character_look(c) for c in present[:2])
                     if present else "no characters, empty location")
        parts.append(composition)

    parts.append(_english_only(scene.get("action"), 160))
    parts.append(_scene_description(scene.get("visual_prompt"), 200))
    parts.extend([region_hint, style])
    return ". ".join(p for p in parts if p)


def scene_characters(scene: dict) -> list[str]:
    """Who is in this scene.

    `staging` is the persisted answer - it is a column on `scenes`, keyed by character_id -
    whereas `characters` is a field of the story schema that is never written to the
    database. Reading `characters` here is how every cinematic prompt came to contain the
    phrase "no characters, empty location" while the action text still described a fox:
    the model was told the place was empty and asked to draw someone in it.
    """
    staged = list((scene.get("staging") or {}).keys())
    if staged:
        return staged
    listed = scene.get("characters") or []
    return [c for c in listed if isinstance(c, str)]
