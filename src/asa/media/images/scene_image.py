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

import re

CINEMATIC_STYLE = (
    "photorealistic cinematic film still, anthropomorphic animal character with a real "
    "animal head and a human body, wearing real clothes, natural fur detail, "
    "shot on 50mm lens, shallow depth of field, soft natural light, "
    "colour-graded, high detail, 16:9")

# Deliberately NOT forbidding people or animals - they are the subject here. What must be
# excluded is the failure modes of photoreal generation: garbled on-image text, extra or
# fused limbs, and the uncanny half-human faces the model produces when "anthropomorphic"
# is read as "human with fur".
CINEMATIC_NEGATIVE = (
    "text, words, letters, lettering, watermark, signature, logo, subtitles, caption, "
    "extra limbs, extra fingers, fused fingers, deformed hands, malformed anatomy, "
    "human face, human head, humanoid face, mask, costume, cosplay, fursuit, "
    "cartoon, illustration, drawing, anime, 3d render, cgi, plastic, "
    "lowres, blurry, jpeg artifacts, oversaturated")

# Real-world grounding. The channel is Indian, so "street" means an Indian street rather
# than the generic western default the model falls back to.
REGION_HINT = ("set in India, authentic Indian surroundings, Indian signage shapes, "
               "Indian street furniture")

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


def character_look(c: dict) -> str:
    """One stable appearance sentence for a character, reused in every scene prompt.

    Built from the stored record rather than re-invented per scene: an image model has no
    memory between calls, so any variation in wording produces a visibly different animal.
    """
    bits = [f"an anthropomorphic {_clean(c.get('species'), 40)}"]
    age = (c.get("age_band") or "").replace("_", " ")
    if age and age != "young adult":
        bits.append(f"{age}")
    appearance = _clean(c.get("appearance"), 120)
    if appearance:
        bits.append(appearance)
    clothing = _clean(c.get("clothing"), 90)
    bits.append(f"wearing {clothing}" if clothing else "wearing simple everyday clothes")
    accessories = _clean(c.get("accessories"), 60)
    if accessories:
        bits.append(f"with {accessories}")
    return ", ".join(b for b in bits if b)


def scene_prompt(scene: dict, cast_by_id: dict[str, dict], *,
                 region_hint: str = REGION_HINT, style: str = CINEMATIC_STYLE) -> str:
    """The full image prompt for one scene: who is in it, doing what, where, framed how."""
    present = [cast_by_id[cid] for cid in (scene.get("characters") or [])
               if cid in cast_by_id]
    if not present:
        # A scene with no staged cast is an establishing shot of the place itself.
        who = "no characters, empty location"
    else:
        who = " and ".join(character_look(c) for c in present[:3])

    framing = _SHOT_FRAMING.get(scene.get("shot") or "medium", _SHOT_FRAMING["medium"])
    parts = [
        who,
        _clean(scene.get("action"), 200),
        _clean(scene.get("visual_prompt"), 200),
        framing,
        region_hint,
        style,
    ]
    return ". ".join(p for p in parts if p)
