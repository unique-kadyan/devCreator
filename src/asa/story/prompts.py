"""Prompt assembly. Blocks are files on disk; the composed prompt is hashed into the DB."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from ..characters.species import PROFILES, SPECIES_LIST
from ..media.animation.shots import (CAMERA_MOVES, EMOTIONS, GESTURES,
                                     SHOT_TYPES, TRANSITIONS)

BLOCKS = Path(__file__).resolve().parents[3] / "prompts" / "_blocks"


@lru_cache(maxsize=8)
def block(name: str) -> str:
    return (BLOCKS / f"{name}.md").read_text().strip()


def system_prompt(extra: str = "") -> str:
    parts = [block("channel_bible"), block("safety_rules")]
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


def outline_prompt(topic: str, keywords: list[str], target_minutes: float,
                   available_characters: list[dict], recent_signatures: list[str],
                   under_used: str, strategy_prefer: str, strategy_avoid: str,
                   max_new_characters: int, archetypes: list[str]) -> str:
    cast = "\n".join(
        f"- {c['id']} - {c['name']}, {c['species']}, {c['age_band']}. "
        f"{c['personality']} Voice: {c['voice_id']}."
        for c in available_characters) or "  (none yet - you must request new characters)"
    recent = "\n".join(f"- {s}" for s in recent_signatures) or "  (none yet)"
    return f"""Create the outline for ONE original animated short story.

SEED TOPIC: {topic}
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


def draft_prompt(outline_json: str, target_minutes: float) -> str:
    return f"""Write the full script for this outline.

OUTLINE:
{outline_json}

Write {int(target_minutes * 150)} words total across narration and dialogue. Narration is
third-person past tense and sparing - let dialogue and action carry the story. Every beat in
the outline must appear. The first two sentences must be the hook.

Return JSON: {{"beats": {{"beginning": str, "conflict": str, "rising": str, "climax": str,
"resolution": str}}}} where each value is the prose for that beat, including dialogue written
inline as: NAME: "line"."""


def scenes_prompt(story_json: str, draft_json: str, cast: list[dict],
                  existing_locations: list[str], sfx_library: list[str],
                  style: str, target_minutes: float) -> str:
    cast_ids = "\n".join(f"- {c['id']} ({c['name']}, {c['species']})" for c in cast)
    return f"""Break this story into scenes for a 2D cutout-animation pipeline.

STORY: {story_json}

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
