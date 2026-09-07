"""Prompt assembly. Blocks are files on disk; the composed prompt is hashed into the DB."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from ..characters.species import PROFILES, SPECIES_LIST
from ..media.animation.shots import (CAMERA_MOVES, EMOTIONS, GESTURES,
                                     SHOT_TYPES, TRANSITIONS)

BLOCKS = Path(__file__).resolve().parents[3] / "prompts" / "_blocks"


@lru_cache(maxsize=8)
def block(name: str) -> str:
    return (BLOCKS / f"{name}.md").read_text().strip()


# Which language the audience HEARS. Only the spoken and on-screen text changes: JSON keys,
# enum values, character ids and location ids stay ASCII, because the schema, the renderer
# and the filesystem all key off them.
LANGUAGE_RULES = {
    "hi": ("WRITE IN HINDI.\n"
           "- All `narration`, all dialogue `line` text, the story `title` and the "
           "`description` must be in natural conversational Hindi, in Devanagari script.\n"
           "- Everything else stays in ENGLISH and ASCII: every JSON key, every enum value "
           "(emotion, shot, camera move, gesture, transition), every `character_id`, every "
           "`location_id`, and every `visual_prompt`.\n"
           "- `visual_prompt` describes a picture for an image model that only understands "
           "English, so it must be English even though the story is Hindi.\n"
           "- For the same reason, a `new_character_spec`'s `appearance`, `clothing` and "
           "`accessories` must be ENGLISH. They are not spoken or shown; they are the one "
           "sentence every image prompt reuses to keep a character looking like itself, "
           "and an image model given Devanagari draws a different animal every shot.\n"
           "- Write Hindi as people actually speak it. Everyday Hindustani, not literary "
           "Sanskritised Hindi, and do not transliterate English sentences into "
           "Devanagari."),
}


def language_rules(language: str) -> str:
    return LANGUAGE_RULES.get((language or "en").lower(), "")


def system_prompt(extra: str = "", language: str = "en") -> str:
    # Order is load-bearing. The channel bible says what kind of story this is; the
    # retention bible says how it has to be built to be watched to the end; the subject
    # bible says what is required when an episode asserts something about the real world;
    # and safety_rules comes last because it overrides all three - it is the one block
    # allowed to veto the others, and a model reading in order should hit the veto after
    # the thing being vetoed. safety_rules refers to the subject block by name, so the two
    # ship together or the reference dangles.
    #
    # subject_bible is included UNCONDITIONALLY rather than only for factual topics, and
    # that is deliberate: its first paragraph tells the model to ignore it for fiction, and
    # the failure it guards - a model stating something as true in an episode nobody
    # classified as factual - is exactly the case a conditional block would miss.
    parts = [block("channel_bible"), block("retention_bible"), block("subject_bible"),
             block("safety_rules")]
    rules = language_rules(language)
    if rules:
        parts.append(rules)
    if extra:
        parts.append(extra)
    parts.append("Return ONLY the requested JSON. No prose, no explanation, no markdown "
                 "fence commentary. Do not show your reasoning.")
    return "\n\n---\n\n".join(parts)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def vocab_block() -> str:
    return (f"camera.move: {CAMERA_MOVES}\n"
            f"shot / camera.from_shot / camera.to_shot: {SHOT_TYPES}\n"
            f"staging.gesture: {GESTURES}\n"
            f"transition_in: {TRANSITIONS}\n"
            f"emotion: {EMOTIONS}")


def species_menu() -> str:
    """The casting sheet given to the writer.

    Listing each animal with what it is *for* rather than as a bare enum is the difference
    between a rhino cast as the immovable gatekeeper and a rhino cast as the nimble thief.
    The physical scale is included because a story that puts a mouse and an elephant in the
    same shot needs to know the pipeline will draw them at their real relative heights.
    """
    by_role: dict[str, list[str]] = {}
    for key, sp in PROFILES.items():
        for role in sp.roles:
            by_role.setdefault(role, []).append(key)
    lines = ["Every animal below is fully drawable. Cast the one whose nature fits the "
             "part - that is what makes a character read before it speaks."]
    for role in ("protagonist", "antagonist", "mentor", "ally", "trickster",
                 "comic_relief", "underdog", "narrator"):
        if by_role.get(role):
            lines.append(f"  best as {role}: {', '.join(sorted(by_role[role]))}")
    lines.append("")
    lines.append("Nature and typical setting of each:")
    for key in SPECIES_LIST:
        sp = PROFILES[key]
        lines.append(f"  {key:<9} {', '.join(sp.traits[:4]):<46} "
                     f"[{', '.join(sp.habitats[:3])}] size {sp.build_scale:.2f}x")
    return "\n".join(lines)


def brief_block(brief: str) -> str:
    """Turn a commissioning brief into instructions the story may not wander away from.

    `topic` is a SEED. "A clever fox opens a village bakery" is a starting point, and a
    model that develops it somewhere better has done its job - that latitude is why the
    outline prompt is full of competing preferences (reuse this cast, prefer an under-used
    archetype, match the animal to the setting).

    A BRIEF is the opposite. Someone is paying for this episode to say a particular thing,
    and a beautiful story that says something else is worthless. The first RoleVo ad written
    through this pipeline came back without the product in it at all: the model read the
    brief as a seed, kept the emotional shape, and replaced the product with generic advice.
    Nothing was broken - the prompt had asked for a story, and it got one.

    So a brief is stated as binding, placed first, and repeated in all three calls: the
    outline, the draft and the scene breakdown each restate it, because a requirement that
    only appears in call one has been through two summarisations by the time anyone writes
    dialogue.
    """
    brief = (brief or "").strip()
    if not brief:
        return ""
    return f"""BRIEF - BINDING REQUIREMENTS. This is not a seed to develop; it is a
specification to satisfy. Where it conflicts with any preference stated later in this
prompt - cast reuse, archetype variety, setting - THE BRIEF WINS.

{brief}

Rules for the brief:
- Every named product, brand, character, place and plot point in it MUST appear in the
  finished script. A version that omits one of them is a failed script, not a variation.
- Write a brand or product name in Latin letters, spelled exactly as the brief spells it,
  even when the rest of the script is in another script. Never transliterate it.
- Use the resolution the brief specifies. Do not substitute a generic lesson for it.
- Do not invent capabilities, statistics or promises for a product beyond what the brief
  states. Understating is acceptable; inventing is not.

"""


def facts_block(outline_json: str) -> str:
    """Restate an outline's real-world claims as a binding list for the later calls.

    Same reasoning as `brief_block`, reached from a different direction: the outline JSON is
    passed whole into the draft and scenes calls, so `facts` is technically already there -
    buried in a field the model has no particular reason to treat as a constraint, several
    thousand characters into a prompt about something else. A requirement that is merely
    PRESENT is not a requirement. Pulled to the top and named, it is.

    Reads the JSON rather than taking a StoryOutline, because both callers already hold the
    serialised form and re-parsing it here keeps the prompt layer free of the schema.
    """
    try:
        data = json.loads(outline_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    facts = [str(f) for f in (data.get("facts") or []) if str(f).strip()]
    if not facts:
        return ""
    subject = str(data.get("subject") or "").strip()
    listed = "\n".join(f"  {i}. {f}" for i, f in enumerate(facts, 1))
    return f"""THIS EPISODE IS ABOUT SOMETHING REAL: {subject or '(see facts)'}

These are the ONLY claims this episode may state as true. They are a ceiling, not a
starting point:

{listed}

- Do not add a fact that is not on this list. Not a date, not a number, not a name, not a
  place. If the story needs one and it is not here, write around it.
- Do not contradict one either, including by simplifying it into something false.
- Everything else in the episode is fiction and must read as fiction: the animals, their
  names, their world, what they say and what happens to them.
- Real people may be CREDITED by name in narration. They are never characters, never speak
  and are never drawn.

"""


def subjects_block(subjects: list[str] | None) -> str:
    """The real-world domains this channel covers, from `story.subjects`.

    Advisory rather than binding, and phrased that way in the prompt: a seed topic that is
    plainly fiction must not be dragged into a science lesson because a list of domains
    appeared above it. What the list is FOR is the opposite case - a seed like "the
    Antikythera mechanism" arriving from the subject collector, where the model needs to
    know that treating it factually is wanted rather than a deviation.
    """
    subjects = [str(s).strip() for s in (subjects or []) if str(s).strip()]
    if not subjects:
        return ""
    return (f"""
REAL SUBJECT MATTER THIS CHANNEL COVERS - if, and only if, the seed topic is about one of
these, write the episode as a factual one under the subject rules: fill `subject`, list
every claim in `facts`, and set `period` when it is not present day.
  {', '.join(subjects)}
If the seed is plainly fiction, leave all three empty and write the story.
""")


def outline_prompt(topic: str, keywords: list[str], target_minutes: float,
                   available_characters: list[dict], recent_signatures: list[str],
                   under_used: str, strategy_prefer: str, strategy_avoid: str,
                   max_new_characters: int, archetypes: list[str],
                   brief: str = "", subjects: list[str] | None = None) -> str:
    cast = "\n".join(
        f"- {c['id']} - {c['name']}, {c['species']}, {c['age_band']}. "
        f"{c['personality']} Voice: {c['voice_id']}."
        for c in available_characters) or "  (none yet - you must request new characters)"
    recent = "\n".join(f"- {s}" for s in recent_signatures) or "  (none yet)"
    return f"""Create the outline for ONE original animated short story.

{brief_block(brief)}SEED TOPIC: {topic}
THEME KEYWORDS: {', '.join(keywords) or '(none)'}
TARGET RUNTIME: {target_minutes:.1f} minutes (~{int(target_minutes * 150)} words of narration
and dialogue combined)

AVAILABLE CAST - prefer these. Reusing them costs nothing and builds continuity across
episodes, which is what makes this a show rather than a content farm:
{cast}

You may request AT MOST {max_new_characters} new character(s). A new character is expensive,
so only request one if the story genuinely cannot work without it.

DO NOT retell any of these previously used plot skeletons:
{recent}

UNDER-USED COMBINATIONS - prefer one of these unless the seed clearly points elsewhere:
{under_used or '  (no history yet)'}

PERFORMANCE SIGNAL from this channel's own analytics (advisory, not binding):
  prefer: {strategy_prefer or '(no data yet)'}
  avoid:  {strategy_avoid or '(no data yet)'}

archetype must be one of: {archetypes}
{subjects_block(subjects)}
BEFORE YOU CHOOSE A PLOT, satisfy all five of these. A story that fails any one of them is
a story nobody finishes, however well written the rest of it is:
  1. The story opens ON the trouble. Name the exact moment it starts - not the situation
     that led to it. `hook` is that moment, written as the thing a viewer sees and hears in
     the first five seconds, not as a description of the premise.
  2. There is one concrete question the opening plants and the ending answers. State it in
     `logline` as a question a viewer would actually ask.
  3. Each of the five beats costs the protagonist more than the one before it. If
     `rising` is not visibly worse than `conflict`, replan it.
  4. There is one honest TURN - something the audience believed that is revealed to be
     wrong at or just before the climax, with its evidence planted in `beginning` or
     `conflict`. Say what the turn is inside the `climax` beat.
  5. The stakes are small, specific and human-sized: a job, a debt, a friendship, a
     reputation, a promise. Save-the-world stakes cannot be paid off in
     {target_minutes:.1f} minutes and read as empty when they are not.

CASTING SHEET - choose species that fit the parts. `size` is relative on-screen height, so
a 0.62 mouse really will stand knee-high to a 1.58 elephant:
{species_menu()}

Match the animal to the setting too: a story set on a river suits an otter or a hippo, one
on a savannah suits a lion, elephant or giraffe, one in a snowbound forest suits a wolf,
bear or deer. Do not default to a fox because it is first in the list.

Return JSON exactly matching this shape:
{{
  "title": str,
  "hook": str,
  "logline": str,
  "target_audience": str,
  "genre": str,
  "archetype": str,
  "moral": str,
  "setting": str,
  "beats": {{"beginning": str, "conflict": str, "rising": str, "climax": str,
             "resolution": str}},
  "ending": str,
  "beat_signature": "verb|verb|verb|verb|verb",
  "subject": str,          "" for fiction; otherwise the real thing this is about,
                           e.g. "how the circumference of the Earth was first measured"
  "facts": [str, ...],     [] for fiction; otherwise EVERY claim the episode states as
                           true, 1-8 of them, each a full checkable sentence. A bare label
                           like "Eratosthenes" is rejected. Only put here what you are
                           confident is well established - a shorter honest list beats a
                           longer one with a guess in it.
  "period": str,           "" for present day; otherwise the era AND place, in English,
                           for the image model: "Alexandria, 3rd century BC". This replaces
                           the channel's contemporary setting hint, so it must name the
                           place too or the pictures lose their location entirely.
  "cast": [{{"character_id": str|null, "role": "protagonist|antagonist|ally|mentor|comic_relief",
             "new_character_spec": null | {{
               "name": str, "species": one of
                 ["fox","rabbit","lion","cat","monkey","dog","bear","mouse","owl","goat",
                  "raccoon","hedgehog"],
               "age_band": "child|teen|young_adult|adult|elder",
               "presentation": str, "pronouns": str, "appearance": str,
               "fur_hex": "#RRGGBB", "accent_hex": "#RRGGBB", "eye_hex": "#RRGGBB",
               "clothing_hex": "#RRGGBB", "clothing": str,
               "personality": str, "backstory": str }}}}]
}}"""


def draft_prompt(outline_json: str, target_minutes: float, brief: str = "") -> str:
    return f"""Write the full script for this outline.

{brief_block(brief)}{facts_block(outline_json)}OUTLINE:
{outline_json}

Write {int(target_minutes * 150)} words total across narration and dialogue. Narration is
third-person past tense and sparing - let dialogue and action carry the story. Every beat in
the outline must appear.

THE OPENING, which is the only part most viewers will see:
- The FIRST thing in `beginning` is a line of dialogue, spoken by a character reacting to
  something that has already gone wrong. Not narration, not a description of the setting,
  not a character stating their own name or job.
- That line must name something concrete and specific - a number, an object, an
  accusation, a deadline. Vague trouble is not trouble.
- Nothing may be explained before it happens. Whatever the viewer needs to know arrives
  afterwards, inside an argument about it.

THROUGHOUT:
- Keep every dialogue line under about twenty-five words. People interrupt and contradict;
  they do not deliver paragraphs. Break a long speech across two characters.
- At most three speaking characters in any one stretch of the story.
- Never have a character say the moral. If the lesson is spoken aloud, cut the line - what
  happens has to carry it.
- The LAST line of `resolution` is the best line in the script: a decision, a reversal, or
  a joke that pays off something planted in `beginning`.

Return JSON: {{"beats": {{"beginning": str, "conflict": str, "rising": str, "climax": str,
"resolution": str}}}} where each value is the prose for that beat, including dialogue written
inline as: NAME: "line"."""


def scenes_prompt(story_json: str, draft_json: str, cast: list[dict],
                  existing_locations: list[str], sfx_library: list[str],
                  style: str, target_minutes: float, brief: str = "") -> str:
    cast_ids = "\n".join(f"- {c['id']} ({c['name']}, {c['species']})" for c in cast)
    return f"""Break this story into scenes for a 2D cutout-animation pipeline.

{brief_block(brief)}{facts_block(story_json)}STORY: {story_json}

SCRIPT: {draft_json}

CAST - dialogue character_id MUST be one of these ids exactly:
{cast_ids}

CLOSED VOCABULARIES - you may ONLY use these values. Anything else cannot be rendered:
{vocab_block()}

LOCATIONS ALREADY DRAWN (reuse where possible - a new location costs image credits):
{', '.join(existing_locations) or '(none yet)'}

SFX TAGS ALREADY IN THE LIBRARY (prefer these):
{', '.join(sfx_library) or '(none yet)'}

VISUAL STYLE for every visual_prompt:
{style}

Rules:
- Aim for {max(4, int(target_minutes * 5))} scenes, 6-16 seconds each.
- Scene 1 must land the hook within its first 3 seconds.
- If this episode is about something real, anything the viewer must SEE to follow it - an
  apparatus, a shadow, a gear train, a diagram scratched in the dirt, a tablet - has to be
  in `action` and in `visual_prompt`. Those two fields are all the renderer reads; a
  measurement that exists only in the narration happens off screen.
- `visual_prompt` describes the LOCATION ONLY. Never describe a character - they are drawn
  from fixed assets and any description of them is discarded.
- `location_id` is snake_case and stable: reuse the same id for the same place.
- `staging` maps character_id -> {{x, y, scale, facing, gesture}}. x is 0..1 across frame.
  Characters must not overlap by more than 15%. y is where the FEET sit (0.90-0.95 typical).
- Every dialogue line names exactly one character_id from the cast.
- `sfx` are short snake_case tags.
- index runs 1..n with no gaps.
- Be COMPACT. One or two sentences per `action`, one clause per `visual_prompt`. A long
  reply gets cut off by the token limit and the tail is lost.

Return JSON with EXACTLY these field names. Every field marked required must be present on
every scene; anything else is ignored:

{{"scenes": [{{
  "index": int,                       required, 1..n
  "location_id": str,                 required, snake_case
  "duration_hint_s": float,           required, 6.0-16.0
  "characters": [character_id, ...],  required, may be empty
  "staging": {{"character_id": {{"x": 0.0-1.0, "y": 0.90-0.95, "scale": 0.3-2.0,
                                "facing": "left"|"right", "gesture": gesture}}}},
  "action": str,                      required, what physically happens
  "narration": str,                   may be ""
  "dialogue": [{{"character_id": str, "line": str, "emotion": emotion}}],
  "emotion": emotion,
  "shot": shot_type,
  "camera": {{"move": camera_move, "from_shot": shot_type,
             "to_shot": shot_type|null, "ease": "linear"|"in"|"out"|"in_out"}},
  "visual_prompt": str,               required, at least 12 characters, LOCATION ONLY
  "sfx": [str, ...],                  required, an ARRAY even for one effect
  "music_cue": str,
  "transition_in": transition
}}]}}

Common mistakes that will be rejected: naming the field "duration" instead of
"duration_hint_s"; omitting "action"; giving "sfx" as a bare string instead of an array;
putting the camera fields at the top level instead of inside "camera"."""
