"""Turn a NewCharacterSpec into a rendered, rigged, licence-clean, database-resident cast
member.

Character consistency in this pipeline is not a model property, it is a file property: the
puppet is drawn once and re-composited forever. So this module is the only place a
character's appearance is decided, and once it has run the look is frozen for the life of
the channel.

Voice casting is deterministic on the character id. A character must sound the same in
episode 40 as in episode 1, and a random pick per render would break that quietly.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..core.db import jdump, tx, read
from ..core.errors import ValidationError
from ..core.ledger import add_asset
from ..core.logging import get_logger
from .procedural_puppet import AnimalPuppet, BASE_PALETTE
from .species import profile as species_profile

log = get_logger("characters")

# Kokoro presets grouped so casting can respect an age band and keep an ensemble varied.
# 'a' = American English, 'b' = British. Mixing accents across a village reads as a world
# rather than a glitch, so both are in play.
VOICE_POOL: dict[str, list[str]] = {
    "child":       ["af_sky", "af_nova", "am_puck", "bf_lily", "af_jessica"],
    "teen":        ["af_bella", "am_echo", "bf_alice", "af_river", "am_liam"],
    "young_adult": ["af_heart", "am_michael", "bf_emma", "bm_lewis", "af_kore",
                    "am_adam", "af_aoede"],
    "adult":       ["am_michael", "bm_george", "af_sarah", "bf_isabella", "am_eric",
                    "af_nicole", "bm_daniel"],
    "elder":       ["bm_fable", "bm_george", "am_santa", "af_nicole", "am_fenrir"],
}
NARRATOR_VOICE = "bm_fable"

# Age changes pitch as much as species does; a child bear is still higher than an adult one.
AGE_PITCH = {"child": 3.0, "teen": 1.5, "young_adult": 0.0, "adult": -0.5, "elder": -1.5}
AGE_RATE = {"child": 1.05, "teen": 1.02, "young_adult": 1.0, "adult": 0.99, "elder": 0.94}

PITCH_LIMIT = 4.0          # beyond this, 24 kHz speech starts sounding artificial


@dataclass
class BuiltCharacter:
    character_id: str
    name: str
    species: str
    puppet_dir: Path
    rig_path: Path
    palette: dict[str, str]
    voice_id: str
    voice_pitch_semi: float
    voice_rate: float
    style_hash: str


def slug(name: str, species: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "char"
    return f"{base}_{species}"


def _stable_int(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def cast_voice(character_id: str, age_band: str, species: str, presentation: str,
               taken: set[str] | None = None) -> tuple[str, float, float]:
    """Pick (voice_id, pitch_semitones, rate) deterministically from the character id.

    `taken` lets an ensemble avoid doubling up: two characters sharing a preset AND a pitch
    are genuinely hard to tell apart in a two-hander scene.
    """
    taken = taken or set()
    pool = VOICE_POOL.get(age_band, VOICE_POOL["young_adult"])
    pres = (presentation or "").lower()
    # Presentation is free text on purpose. Only steer when the character says so; never
    # infer a voice from a name.
    if any(w in pres for w in ("she/her", "female", "woman", "girl", "doe", "vixen")):
        pool = [v for v in pool if v[1] == "f"] or pool
    elif any(w in pres for w in ("he/him", "male", "man", "boy", "buck", "tod")):
        pool = [v for v in pool if v[1] == "m"] or pool

    n = _stable_int(character_id, "voice")
    order = [pool[(n + i) % len(pool)] for i in range(len(pool))]
    voice = next((v for v in order if v not in taken), order[0])

    sp = species_profile(species)
    pitch = AGE_PITCH.get(age_band, 0.0) + sp.pitch_bias
    # A deterministic nudge so two same-species, same-age characters still differ.
    pitch += ((_stable_int(character_id, "pitch") % 100) / 100.0 - 0.5) * 1.4
    pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, round(pitch, 2)))
    # Age and species biases multiply, so an elder owl lands near 0.85x. Clamp: below
    # ~0.88x Kokoro's phrasing starts to drag audibly, and emotion pacing multiplies on
    # top of this at render time.
    rate = round(min(1.15, max(0.88, AGE_RATE.get(age_band, 1.0) * sp.rate_bias)), 3)
    return voice, pitch, rate


def _shade(hex_colour: str, factor: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda v: max(0, min(255, int(round(v * factor))))    # noqa: E731
    return f"#{f(r):02X}{f(g):02X}{f(b):02X}"


def palette_from_spec(fur: str, accent: str, eyes: str, clothing: str) -> dict[str, str]:
    """Four author-chosen colours expand into the twelve the puppet actually draws.

    Deriving shadows rather than asking the model for them keeps the shading internally
    consistent - a model asked for twelve hexes reliably produces at least one that fights
    the others.
    """
    return dict(
        BASE_PALETTE,
        fur=fur,
        fur_shadow=_shade(fur, 0.86),
        chest=accent,
        ear_inner=_shade(fur, 0.62),
        eyes=eyes,
        paw=_shade(accent, 0.97),
        hoodie=clothing,
        hoodie_shadow=_shade(clothing, 0.82),
        shorts=_shade(clothing, 0.60),
        mane=_shade(fur, 0.80),
        spine=_shade(fur, 0.55),
        outline=_shade(fur, 0.30),
    )


class CharacterFactory:
    """Builds puppets and registers them. Idempotent: rebuilding an existing character
    reuses its id, palette and voice unless `force` is set."""

    def __init__(self, db: Path, assets_root: Path):
        self.db = Path(db)
        self.root = Path(assets_root)

    # ------------------------------------------------------------------ queries

    def existing(self) -> list[dict]:
        with read(self.db) as con:
            rows = con.execute(
                "SELECT id, name, species, age_band, presentation, pronouns, personality, "
                "       voice_id, appearances, status, palette, clothing "
                "FROM characters WHERE status IN ('ready','draft') ORDER BY appearances DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def taken_voices(self) -> set[str]:
        with read(self.db) as con:
            return {r[0] for r in con.execute("SELECT voice_id FROM characters")}

    # ------------------------------------------------------------------ build

    def build_from_spec(self, spec, force: bool = False) -> BuiltCharacter:
        """`spec` is a story.schema.NewCharacterSpec (duck-typed so tests can pass a stub)."""
        cid = slug(spec.name, spec.species)
        with read(self.db) as con:
            row = con.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
        if row and not force:
            log.info("character_exists_reusing", character_id=cid)
            d = self.root / "characters" / cid
            return BuiltCharacter(
                character_id=cid, name=row["name"], species=row["species"],
                puppet_dir=d, rig_path=d / "rig.json",
                palette=json.loads(row["palette"] or "{}"), voice_id=row["voice_id"],
                voice_pitch_semi=row["voice_pitch_semi"], voice_rate=row["voice_rate"],
                style_hash=row["style_hash"] or "")

        palette = palette_from_spec(spec.fur_hex, spec.accent_hex, spec.eye_hex,
                                    spec.clothing_hex)
        voice, pitch, rate = cast_voice(cid, spec.age_band, spec.species,
                                        spec.presentation, self.taken_voices())

        out_dir = self.root / "characters" / cid
        rig = AnimalPuppet(cid, palette, spec.species).build(out_dir)
        self._register(cid, spec, palette, rig, voice, pitch, rate, out_dir)
        log.info("character_built", character_id=cid, species=spec.species, voice=voice,
                 pitch=pitch, layers=len(rig.layers))
        return BuiltCharacter(cid, spec.name, spec.species, out_dir, out_dir / "rig.json",
                              palette, voice, pitch, rate, rig.style_hash)

    def _register(self, cid, spec, palette, rig, voice, pitch, rate, out_dir) -> None:
        rel = str(out_dir.relative_to(self.root.parent)) if self.root.parent in out_dir.parents \
            else str(out_dir)
        with tx(self.db) as con:
            con.execute("""
                INSERT INTO characters (id, name, species, age_band, presentation, pronouns,
                    appearance, palette, clothing, accessories, personality, backstory,
                    voice_id, voice_pitch_semi, voice_rate, style_hash, puppet_dir, rig_json,
                    status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ready')
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, species=excluded.species, palette=excluded.palette,
                    appearance=excluded.appearance, personality=excluded.personality,
                    backstory=excluded.backstory, style_hash=excluded.style_hash,
                    puppet_dir=excluded.puppet_dir, rig_json=excluded.rig_json,
                    status='ready', updated_at=datetime('now')
            """, (cid, spec.name, spec.species, spec.age_band, spec.presentation,
                  spec.pronouns, spec.appearance, jdump(palette), spec.clothing, "[]",
                  spec.personality, spec.backstory, voice, pitch, rate, rig.style_hash,
                  rel, f"{rel}/rig.json"))

        # Every layer is a first-class asset. Procedural art is ours outright, but the
        # ledger must still be able to prove that for any frame we ship.
        for layer in rig.layers.values():
            add_asset(self.db, out_dir / layer.file, kind="character_layer",
                      source="procedural", license_code="CC0",
                      source_ref=f"asa:procedural_puppet/{spec.species}",
                      meta={"character_id": cid, "style_hash": rig.style_hash})

    # ------------------------------------------------------------------ helpers

    def ensure_cast(self, cast: list[dict]) -> dict[str, BuiltCharacter]:
        """Build every member of a story cast that does not exist yet.

        Returns id -> BuiltCharacter for the whole cast, including reused members, so the
        caller can look up rigs and voices without a second query.
        """
        out: dict[str, BuiltCharacter] = {}
        for member in cast:
            spec = member.get("spec")
            cid = member["id"]
            if spec is not None:
                built = self.build_from_spec(spec)
                out[built.character_id] = built
                continue
            with read(self.db) as con:
                row = con.execute("SELECT * FROM characters WHERE id = ?", (cid,)).fetchone()
            if row is None:
                raise ValidationError(
                    f"cast references character {cid!r} which is not in the database and "
                    f"carries no spec to build it from")
            d = self.root / "characters" / cid
            if not (d / "rig.json").exists():
                raise ValidationError(
                    f"character {cid!r} is registered but its puppet is missing at {d}. "
                    f"Rebuild it with: asa character rebuild {cid}")
            out[cid] = BuiltCharacter(cid, row["name"], row["species"], d, d / "rig.json",
                                      json.loads(row["palette"] or "{}"), row["voice_id"],
                                      row["voice_pitch_semi"], row["voice_rate"],
                                      row["style_hash"] or "")
        return out

    def bump_appearances(self, ids: list[str]) -> None:
        with tx(self.db) as con:
            con.executemany(
                "UPDATE characters SET appearances = appearances + 1, "
                "updated_at = datetime('now') WHERE id = ?", [(i,) for i in ids])
