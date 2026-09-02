"""Structured-output schemas. These ARE the contract with the renderer.

Every field the compositor consumes is a closed enum, so the model cannot emit an
instruction that has no implementation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..characters.species import SPECIES_LIST
from ..media.animation.shots import (CAMERA_MOVES, EASING, EMOTIONS, GESTURES,
                                     SHOT_TYPES, TRANSITIONS)

Archetype = Literal["underdog", "trickster", "redemption", "mystery",
                    "friendship", "survival", "comedy", "family"]
# Bound to the profile table rather than hand-listed: a species the writer may choose is
# by construction a species the puppet builder can draw. There is no path to a cast the
# renderer cannot render.
Species = Literal[tuple(SPECIES_LIST)]             # type: ignore[valid-type]
Role = Literal["protagonist", "antagonist", "ally", "mentor", "comic_relief"]

CameraMove = Literal[tuple(CAMERA_MOVES)]          # type: ignore[valid-type]
ShotType = Literal[tuple(SHOT_TYPES)]              # type: ignore[valid-type]
Gesture = Literal[tuple(GESTURES)]                 # type: ignore[valid-type]
Transition = Literal[tuple(TRANSITIONS)]           # type: ignore[valid-type]
Emotion = Literal[tuple(EMOTIONS)]                 # type: ignore[valid-type]
Easing = Literal[tuple(EASING)]                    # type: ignore[valid-type]


class NewCharacterSpec(BaseModel):
    name: str = Field(max_length=24)
    species: Species
    age_band: Literal["child", "teen", "young_adult", "adult", "elder"] = "young_adult"
    presentation: str = Field(default="", max_length=60)
    pronouns: str = "they/them"
    appearance: str = Field(max_length=300)
    fur_hex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    accent_hex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    eye_hex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    clothing_hex: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    clothing: str = Field(default="", max_length=120)
    personality: str = Field(max_length=300)
    backstory: str = Field(default="", max_length=400)


class CastMember(BaseModel):
    character_id: str | None = None
    role: Role
    new_character_spec: NewCharacterSpec | None = None

    @field_validator("new_character_spec")
    @classmethod
    def _need_one(cls, v, info):
        if v is None and not info.data.get("character_id"):
            raise ValueError("cast member needs character_id or new_character_spec")
        return v


class Beats(BaseModel):
    beginning: str
    conflict: str
    rising: str
    climax: str
    resolution: str


class StoryOutline(BaseModel):
    title: str = Field(max_length=70)
    hook: str = Field(max_length=220)
    logline: str = Field(max_length=320)
    target_audience: str
    genre: str
    archetype: Archetype
    moral: str
    setting: str
    beats: Beats
    ending: str
    beat_signature: str
    cast: list[CastMember] = Field(min_length=1, max_length=5)

    @field_validator("beat_signature")
    @classmethod
    def _five_verbs(cls, v: str) -> str:
        parts = [p.strip().lower() for p in v.split("|") if p.strip()]
        if len(parts) != 5:
            raise ValueError("beat_signature must be exactly 5 pipe-separated verbs")
        return "|".join(parts)

    @field_validator("title")
    @classmethod
    def _no_clickbait(cls, v: str) -> str:
        shouty = [w for w in v.split() if len(w) > 2 and w.isupper()]
        if len(shouty) > 1:
            raise ValueError("title has more than one ALL-CAPS word")
        return v.strip()


class DialogueLine(BaseModel):
    character_id: str
    line: str = Field(max_length=280)
    emotion: Emotion = "neutral"


class Staging(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(default=0.93, ge=0.0, le=1.0)
    scale: float = Field(default=1.0, ge=0.3, le=2.0)
    facing: Literal["left", "right"] = "right"
    gesture: Gesture = "idle"


class Camera(BaseModel):
    move: CameraMove = "static"
    from_shot: ShotType = "wide"
    to_shot: ShotType | None = None
    ease: Easing = "in_out"


class Scene(BaseModel):
    """Tolerant on input, strict on output.

    Free models get the shape right maybe four times in five. Where a near-miss is
    unambiguous - a bare string where a list belongs, `duration` for `duration_hint_s`,
    camera keys hoisted to the top level - coercing it is strictly better than burning a
    repair call, because the repair costs a request against a 50/day budget and often
    introduces a different mistake. Anything genuinely ambiguous still raises.
    """

    model_config = {"populate_by_name": True}

    index: int = Field(ge=1)
    location_id: str
    duration_hint_s: float = Field(default=8.0, ge=1.0, le=30.0)
    characters: list[str] = Field(default_factory=list)
    staging: dict[str, Staging] = Field(default_factory=dict)
    action: str
    narration: str = ""
    dialogue: list[DialogueLine] = Field(default_factory=list)
    emotion: Emotion = "neutral"
    shot: ShotType = "medium"
    camera: Camera = Field(default_factory=Camera)
    visual_prompt: str
    sfx: list[str] = Field(default_factory=list)
    music_cue: str = "neutral"
    transition_in: Transition = "cut"

    @model_validator(mode="before")
    @classmethod
    def _coerce_near_misses(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        for alias, field in (("duration", "duration_hint_s"), ("duration_s",
                             "duration_hint_s"), ("scene_index", "index"),
                            ("location", "location_id"), ("sound_effects", "sfx"),
                            ("music", "music_cue"), ("transition", "transition_in")):
            if alias in d and field not in d:
                d[field] = d.pop(alias)
        if isinstance(d.get("sfx"), str):
            d["sfx"] = [x.strip() for x in d["sfx"].split(",") if x.strip()]
        if isinstance(d.get("characters"), str):
            d["characters"] = [x.strip() for x in d["characters"].split(",") if x.strip()]
        # camera keys hoisted to the top level
        cam = d.get("camera")
        if not isinstance(cam, dict):
            cam = {} if cam is None else {"move": str(cam)}
            for key, target in (("camera_move", "move"), ("move", "move"),
                                ("from_shot", "from_shot"), ("to_shot", "to_shot"),
                                ("ease", "ease")):
                if key in d and isinstance(d[key], (str, type(None))):
                    cam[target] = d[key]
            d["camera"] = cam
        if not d.get("action") and d.get("description"):
            d["action"] = d["description"]
        return d

    @field_validator("visual_prompt")
    @classmethod
    def _location_only(cls, v: str) -> str:
        if len(v) < 12:
            raise ValueError("visual_prompt too short to describe a location")
        return v


class SceneList(BaseModel):
    scenes: list[Scene] = Field(min_length=3, max_length=60)

    @field_validator("scenes")
    @classmethod
    def _sequential(cls, v: list[Scene]) -> list[Scene]:
        for i, s in enumerate(v, start=1):
            if s.index != i:
                raise ValueError(f"scene indices must be 1..n in order; got {s.index} at {i}")
        return v


class TitleCandidate(BaseModel):
    """The model's own scores are advisory and frequently malformed; `metadata.py` scores
    every title in code anyway, so a missing or mistyped `scores` block is not a reason to
    throw the title away."""

    title: str = Field(max_length=100)
    scores: dict[str, float] = Field(default_factory=dict)
    accuracy_justification: str = ""

    @field_validator("title", mode="before")
    @classmethod
    def _trim(cls, v):
        return str(v or "").strip()[:100]

    @field_validator("scores", mode="before")
    @classmethod
    def _numeric_only(cls, v):
        if not isinstance(v, dict):
            return {}
        out = {}
        for k, val in v.items():
            try:
                out[str(k)] = float(val)
            except (TypeError, ValueError):
                continue
        return out


class MetadataDraft(BaseModel):
    """Over-long lists are TRIMMED, not rejected.

    Unlike the scene list, nothing here is a rendering contract - an extra hashtag or a
    long description is trivially droppable. Rejecting a whole draft because the model
    returned six hashtags instead of five spends another request from a 50/day budget to
    fix something a slice fixes for free. (This happened.)
    """

    titles: list[TitleCandidate] = Field(min_length=1)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)

    @field_validator("titles", mode="before")
    @classmethod
    def _cap_titles(cls, v):
        return v[:10] if isinstance(v, list) else v

    @field_validator("tags", "hashtags", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if isinstance(v, str):
            v = [x.strip() for x in v.replace("#", " #").split(",") if x.strip()]
        return [str(x) for x in v][:30] if isinstance(v, list) else v

    @field_validator("hashtags")
    @classmethod
    def _cap_hashtags(cls, v: list[str]) -> list[str]:
        # Generous on purpose: publish.metadata merges these with the channel's own
        # hashtags and applies YouTube's real ceiling. Capping at 5 here would starve
        # that merge before it could dedupe.
        return v[:20]

    @field_validator("description", mode="before")
    @classmethod
    def _cap_description(cls, v):
        if isinstance(v, list):
            v = "\n\n".join(str(x) for x in v)
        return str(v or "")[:4500]


class SafetyReview(BaseModel):
    passed: bool
    made_for_kids_estimate: bool
    flags: list[str] = Field(default_factory=list)
    reasoning: str = ""
