"""Species geometry, silhouette and casting suitability.

Two jobs, deliberately in one file because they must not drift apart:

1. **Geometry.** Every species produces the *same* rig - identical layer keys, anchors and
   viseme names - so the compositor never learns what animal it is drawing and swapping the
   cast costs nothing downstream. Only proportions, silhouette features and palette change.
   Silhouette is what makes a species readable at thumbnail size, which is the only size
   most viewers ever see.

2. **Casting suitability.** Which animal belongs in which story. A rhino cast as the
   nimble thief and a mouse cast as the immovable gatekeeper both read as mistakes even to
   a viewer who could not say why. `suggest()` turns a scene's needs into a ranked list, so
   the writer picks from animals that fit rather than from whatever it thought of first.

Sizes are in the 1024x1024 puppet canvas. `build_scale` is the multiplier the compositor
applies so a mouse and an elephant standing together read at plausible relative heights.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeciesProfile:
    key: str
    # ---- silhouette -------------------------------------------------------
    ear: str = "triangle"       # triangle | round | long | tuft | fan | tiny | none
    ear_w: float = 1.0
    ear_h: float = 1.0
    ear_tilt: float = 0.0       # outward lean at the tip, canvas px
    head_r: tuple[int, int] = (150, 138)
    muzzle: tuple[int, int] = (86, 58)
    muzzle_dy: int = 62
    nose_r: tuple[int, int] = (22, 16)
    body_r: tuple[int, int] = (142, 176)
    tail: str = "bushy"         # bushy | thin | tuftend | stub | curl | none
    tail_len: float = 1.0
    tail_thick: float = 1.0
    # ---- distinguishing features -----------------------------------------
    whiskers: bool = False
    horns: str = "none"         # none | goat | bull | rhino | antler | ossicone
    mane: bool = False
    trunk: bool = False
    spines: bool = False
    mask: bool = False
    stripes: bool = False
    spots: bool = False
    tusks: bool = False
    mouth_style: str = "muzzle"  # muzzle | beak
    facial_disc: bool = False   # the flat owl face; a crow has a beak but no disc
    neck_len: float = 0.0       # extra neck height in canvas px (giraffe, camel)
    # ---- staging ----------------------------------------------------------
    build_scale: float = 1.0    # relative on-screen height vs. the fox baseline
    # ---- voice ------------------------------------------------------------
    pitch_bias: float = 0.0     # semitones
    rate_bias: float = 1.0
    # ---- casting ----------------------------------------------------------
    traits: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    habitats: tuple[str, ...] = ()


_P = SpeciesProfile

PROFILES: dict[str, SpeciesProfile] = {

    # ---------------------------------------------------------------- small
    "mouse": _P("mouse", "round", 1.35, 1.30, 22, (138, 132), (62, 44), 56, (16, 12),
                (118, 152), "thin", 1.2, 0.35, whiskers=True, build_scale=0.62,
                pitch_bias=4.0, rate_bias=1.10,
                traits=("small", "quick", "overlooked", "clever", "timid"),
                roles=("underdog", "trickster", "comic_relief"),
                habitats=("village", "kitchen", "barn", "field")),

    "hedgehog": _P("hedgehog", "round", 0.70, 0.70, 4, (144, 136), (66, 48), 58, (18, 14),
                   (140, 162), "stub", 0.25, 1.0, whiskers=True, spines=True,
                   build_scale=0.66, pitch_bias=2.0, rate_bias=1.02,
                   traits=("small", "defensive", "gentle", "stubborn"),
                   roles=("underdog", "ally", "comic_relief"),
                   habitats=("garden", "forest", "hedgerow", "village")),

    "squirrel": _P("squirrel", "tuft", 0.72, 0.9, -4, (140, 134), (64, 46), 56, (16, 13),
                   (124, 156), "bushy", 1.25, 1.35, whiskers=True, build_scale=0.66,
                   pitch_bias=3.4, rate_bias=1.12,
                   traits=("quick", "hoarding", "restless", "chatty"),
                   roles=("comic_relief", "trickster", "ally"),
                   habitats=("forest", "park", "orchard", "wood")),

    "rabbit": _P("rabbit", "long", 0.62, 2.35, 14, (146, 140), (74, 52), 60, (18, 14),
                 (134, 168), "stub", 0.35, 1.15, build_scale=0.78,
                 pitch_bias=2.5, rate_bias=1.06,
                 traits=("quick", "anxious", "watchful", "gentle"),
                 roles=("underdog", "ally", "protagonist"),
                 habitats=("meadow", "burrow", "field", "garden", "forest")),

    "cat": _P("cat", "triangle", 0.86, 0.82, -4, (150, 142), (72, 50), 60, (20, 15),
              (132, 168), "thin", 1.15, 0.55, whiskers=True, build_scale=0.86,
              pitch_bias=1.5, rate_bias=1.0,
              traits=("independent", "aloof", "graceful", "sly", "curious"),
              roles=("trickster", "antagonist", "mentor", "protagonist"),
              habitats=("village", "rooftop", "market", "harbour", "house")),

    "fox": _P("fox", "triangle", 1.0, 1.0, 0, (150, 138), (86, 58), 62, (22, 16),
              (142, 176), "bushy", 1.0, 1.0, build_scale=1.0,
              pitch_bias=0.0, rate_bias=1.0,
              traits=("clever", "quick", "resourceful", "charming", "sly"),
              roles=("protagonist", "trickster", "underdog"),
              habitats=("forest", "village", "meadow", "market", "hedgerow")),

    "raccoon": _P("raccoon", "round", 0.95, 0.95, 6, (150, 140), (78, 54), 60, (22, 16),
                  (142, 174), "bushy", 1.05, 1.05, mask=True, build_scale=0.92,
                  pitch_bias=0.5, rate_bias=1.04,
                  traits=("mischievous", "dexterous", "opportunistic", "nocturnal"),
                  roles=("trickster", "comic_relief", "antagonist"),
                  habitats=("village", "river", "town", "attic", "market")),

    "otter": _P("otter", "tiny", 0.5, 0.5, 0, (144, 132), (76, 52), 58, (20, 16),
                (132, 172), "thin", 1.1, 0.9, whiskers=True, build_scale=0.84,
                pitch_bias=1.8, rate_bias=1.08,
                traits=("playful", "social", "agile", "warm"),
                roles=("comic_relief", "ally", "protagonist"),
                habitats=("river", "lake", "shore", "sea", "marsh")),

    "monkey": _P("monkey", "round", 1.3, 1.15, 26, (146, 140), (78, 60), 62, (20, 16),
                 (132, 168), "curl", 1.35, 0.4, build_scale=0.86,
                 pitch_bias=1.0, rate_bias=1.08,
                 traits=("agile", "mischievous", "curious", "loud", "clever"),
                 roles=("trickster", "comic_relief", "ally"),
                 habitats=("jungle", "temple", "market", "forest", "canopy")),

    "owl": _P("owl", "tuft", 0.46, 0.55, -6, (162, 152), (52, 40), 52, (26, 22),
              (150, 176), "none", 0.0, 1.0, mouth_style="beak", facial_disc=True,
              build_scale=0.84,
              pitch_bias=-1.0, rate_bias=0.90,
              traits=("wise", "watchful", "patient", "solitary", "nocturnal"),
              roles=("mentor", "narrator", "ally"),
              habitats=("forest", "belfry", "night", "wood", "ruin")),

    "crow": _P("crow", "none", 0.0, 0.0, 0, (144, 136), (46, 34), 50, (26, 20),
               (134, 164), "thin", 0.7, 1.2, mouth_style="beak", build_scale=0.74,
               pitch_bias=0.8, rate_bias=1.06,
               traits=("clever", "watchful", "opportunistic", "wry"),
               roles=("trickster", "narrator", "antagonist"),
               habitats=("village", "field", "market", "ruin", "town")),

    # --------------------------------------------------------------- medium
    "dog": _P("dog", "round", 1.05, 1.25, 18, (154, 144), (94, 62), 64, (24, 18),
              (148, 180), "bushy", 0.8, 0.9, build_scale=1.0,
              pitch_bias=-0.5, rate_bias=1.02,
              traits=("loyal", "eager", "brave", "warm", "trusting"),
              roles=("ally", "protagonist", "comic_relief"),
              habitats=("village", "farm", "harbour", "town", "house")),

    "wolf": _P("wolf", "triangle", 1.0, 1.15, 4, (156, 144), (98, 62), 66, (26, 19),
               (154, 186), "bushy", 1.05, 1.1, build_scale=1.12,
               pitch_bias=-2.5, rate_bias=0.96,
               traits=("proud", "loyal", "dangerous", "disciplined", "wary"),
               roles=("antagonist", "mentor", "protagonist"),
               habitats=("forest", "mountain", "snow", "wild", "tundra")),

    "goat": _P("goat", "long", 0.75, 1.1, 26, (142, 148), (84, 62), 70, (20, 16),
               (140, 174), "stub", 0.3, 0.8, horns="goat", build_scale=0.94,
               pitch_bias=-1.5, rate_bias=0.98,
               traits=("stubborn", "sure-footed", "contrary", "hardy"),
               roles=("comic_relief", "ally", "antagonist"),
               habitats=("mountain", "farm", "cliff", "village", "hill")),

    "boar": _P("boar", "tiny", 0.6, 0.6, 8, (156, 142), (104, 70), 66, (30, 22),
               (166, 190), "stub", 0.28, 0.9, tusks=True, build_scale=1.06,
               pitch_bias=-3.0, rate_bias=0.94,
               traits=("stubborn", "blunt", "territorial", "tough"),
               roles=("antagonist", "comic_relief"),
               habitats=("forest", "wood", "farm", "mud", "thicket")),

    "deer": _P("deer", "long", 0.62, 1.15, 20, (144, 146), (78, 56), 66, (20, 16),
               (138, 182), "stub", 0.3, 0.7, horns="antler", spots=True,
               build_scale=1.06, pitch_bias=0.5, rate_bias=0.98,
               traits=("gentle", "watchful", "graceful", "shy"),
               roles=("ally", "protagonist", "underdog"),
               habitats=("forest", "meadow", "wood", "glade", "snow")),

    "panther": _P("panther", "round", 0.82, 0.78, -2, (154, 144), (82, 56), 62, (24, 17),
                  (150, 184), "thin", 1.25, 0.55, whiskers=True, build_scale=1.18,
                  pitch_bias=-3.2, rate_bias=0.92,
                  traits=("silent", "dangerous", "graceful", "solitary", "watchful"),
                  roles=("antagonist", "mentor", "protagonist"),
                  habitats=("jungle", "night", "ruin", "canopy", "cave")),

    "tiger": _P("tiger", "round", 0.86, 0.80, 2, (162, 150), (94, 64), 64, (28, 20),
                (168, 194), "thin", 1.2, 0.7, whiskers=True, stripes=True,
                build_scale=1.30, pitch_bias=-3.8, rate_bias=0.92,
                traits=("powerful", "proud", "dangerous", "solitary"),
                roles=("antagonist", "mentor", "protagonist"),
                habitats=("jungle", "forest", "temple", "river", "snow")),

    "leopard": _P("leopard", "round", 0.84, 0.78, 0, (156, 146), (86, 58), 62, (25, 18),
                  (152, 186), "thin", 1.3, 0.6, whiskers=True, spots=True,
                  build_scale=1.18, pitch_bias=-3.0, rate_bias=0.94,
                  traits=("stealthy", "graceful", "opportunistic", "solitary"),
                  roles=("antagonist", "trickster", "protagonist"),
                  habitats=("savannah", "jungle", "cliff", "canopy", "night")),

    "lion": _P("lion", "round", 0.80, 0.72, 8, (162, 152), (92, 62), 62, (26, 19),
               (162, 190), "tuftend", 1.1, 0.5, mane=True, build_scale=1.26,
               pitch_bias=-3.5, rate_bias=0.94,
               traits=("proud", "commanding", "brave", "vain", "loyal"),
               roles=("mentor", "antagonist", "protagonist"),
               habitats=("savannah", "plain", "rock", "grassland", "throne")),

    "bear": _P("bear", "round", 0.86, 0.80, 10, (168, 156), (96, 66), 64, (28, 20),
               (176, 198), "stub", 0.3, 1.3, build_scale=1.34,
               pitch_bias=-4.0, rate_bias=0.92,
               traits=("strong", "slow", "gentle", "solitary", "protective"),
               roles=("mentor", "ally", "antagonist"),
               habitats=("forest", "mountain", "river", "snow", "cave")),

    # ---------------------------------------------------------------- large
    "buffalo": _P("buffalo", "tiny", 0.62, 0.55, 12, (168, 150), (108, 74), 66, (32, 24),
                  (192, 200), "tuftend", 0.75, 0.55, horns="bull", build_scale=1.42,
                  pitch_bias=-4.0, rate_bias=0.88,
                  traits=("stoic", "immovable", "communal", "patient", "strong"),
                  roles=("mentor", "ally", "antagonist"),
                  habitats=("plain", "savannah", "river", "grassland", "dust")),

    "bull": _P("bull", "tiny", 0.60, 0.55, 12, (164, 148), (106, 74), 66, (32, 24),
               (188, 198), "tuftend", 0.8, 0.5, horns="bull", build_scale=1.40,
               pitch_bias=-4.0, rate_bias=0.90,
               traits=("proud", "hot-tempered", "strong", "direct"),
               roles=("antagonist", "ally"),
               habitats=("farm", "field", "market", "ring", "village")),

    "rhino": _P("rhino", "tiny", 0.55, 0.50, 8, (170, 152), (112, 76), 64, (34, 26),
                (198, 202), "stub", 0.26, 0.7, horns="rhino", build_scale=1.46,
                pitch_bias=-4.0, rate_bias=0.86,
                traits=("armoured", "short-sighted", "blunt", "immovable", "solitary"),
                roles=("antagonist", "ally", "comic_relief"),
                habitats=("savannah", "plain", "mud", "grassland", "river")),

    "elephant": _P("elephant", "fan", 2.6, 2.3, 54, (174, 156), (86, 62), 60, (26, 22),
                   (206, 206), "thin", 0.55, 0.45, trunk=True, tusks=True,
                   build_scale=1.58, pitch_bias=-4.0, rate_bias=0.84,
                   traits=("wise", "patient", "strong", "remembering", "gentle"),
                   roles=("mentor", "ally", "narrator"),
                   habitats=("savannah", "jungle", "river", "plain", "temple")),

    "hippo": _P("hippo", "tiny", 0.5, 0.45, 6, (168, 148), (116, 80), 62, (34, 26),
                (200, 198), "stub", 0.24, 0.8, build_scale=1.44,
                pitch_bias=-4.0, rate_bias=0.86,
                traits=("blunt", "territorial", "heavy", "unexpectedly quick"),
                roles=("antagonist", "comic_relief", "ally"),
                habitats=("river", "lake", "marsh", "mud", "shore")),

    "giraffe": _P("giraffe", "tiny", 0.55, 0.65, 10, (132, 128), (74, 54), 58, (20, 16),
                  (140, 190), "tuftend", 0.6, 0.5, horns="ossicone", spots=True,
                  neck_len=250.0, build_scale=1.52, pitch_bias=-1.0, rate_bias=0.90,
                  traits=("far-seeing", "aloof", "gentle", "awkward"),
                  roles=("ally", "narrator", "comic_relief"),
                  habitats=("savannah", "plain", "acacia", "grassland")),

    "camel": _P("camel", "tiny", 0.6, 0.6, 8, (140, 138), (92, 66), 68, (24, 20),
                (168, 194), "thin", 0.7, 0.6, neck_len=110.0, build_scale=1.30,
                pitch_bias=-2.0, rate_bias=0.88,
                traits=("enduring", "sardonic", "patient", "hardy"),
                roles=("ally", "comic_relief", "mentor"),
                habitats=("desert", "dune", "caravan", "oasis", "market")),

    "horse": _P("horse", "long", 0.6, 1.0, 14, (144, 146), (98, 66), 70, (24, 20),
                (162, 192), "thin", 1.0, 1.2, mane=True, build_scale=1.30,
                pitch_bias=-2.0, rate_bias=0.96,
                traits=("swift", "loyal", "proud", "willing"),
                roles=("ally", "protagonist", "mentor"),
                habitats=("farm", "plain", "road", "village", "meadow")),
}

DEFAULT = PROFILES["fox"]

# The species the story writer is offered. Keeping this identical to PROFILES means a
# schema-legal species is always renderable - there is no path to an unrenderable cast.
SPECIES_LIST: list[str] = sorted(PROFILES)


def profile(species: str) -> SpeciesProfile:
    """Unknown species falls back to fox geometry rather than failing a render.

    A story that names an animal with no profile is still a story worth making; it just
    gets generic proportions, and the palette carries the identity.
    """
    return PROFILES.get((species or "").strip().lower(), DEFAULT)


def known() -> list[str]:
    return SPECIES_LIST


def build_scale(species: str) -> float:
    return profile(species).build_scale


def relative_scale(species: str, reference: str = "fox") -> float:
    """How tall this species should be drawn beside `reference`.

    Used by the compositor so a mouse and an elephant in one shot look like a mouse and an
    elephant, instead of two puppets of identical height in different colours.
    """
    return profile(species).build_scale / max(0.01, profile(reference).build_scale)


def suggest(role: str | None = None, traits: tuple[str, ...] | list[str] = (),
            habitat: str | None = None, exclude: set[str] | None = None,
            limit: int = 8) -> list[tuple[str, float]]:
    """Rank species by how well they fit a role, a set of traits and a setting.

    Scoring is intentionally simple and additive so the reasoning is inspectable: role fit
    is worth most (a mentor who reads as a mentor carries the scene), traits next, habitat
    last because a story can legitimately move an animal somewhere unexpected - that is
    often the premise.
    """
    exclude = exclude or set()
    want = {t.strip().lower() for t in traits if t and t.strip()}
    hab = (habitat or "").strip().lower()
    out: list[tuple[str, float]] = []
    for key, sp in PROFILES.items():
        if key in exclude:
            continue
        score = 0.0
        if role:
            r = role.strip().lower()
            if r in sp.roles:
                score += 1.0 - 0.12 * sp.roles.index(r)     # first listed fits best
        if want:
            score += 0.75 * len(want & set(sp.traits)) / len(want)
        if hab:
            if any(h in hab or hab in h for h in sp.habitats):
                score += 0.4
        out.append((key, round(score, 4)))
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out[:limit]


def casting_brief(role: str | None = None, traits: tuple[str, ...] | list[str] = (),
                  habitat: str | None = None, exclude: set[str] | None = None,
                  limit: int = 6) -> str:
    """A human-readable line for the story prompt naming the animals that would fit."""
    picks = [(k, s) for k, s in suggest(role, traits, habitat, exclude, limit) if s > 0]
    if not picks:
        return ""
    return ", ".join(f"{k} ({', '.join(PROFILES[k].traits[:3])})" for k, _ in picks)
