"""Score, deduplicate and select topics.

The scoring is transparent on purpose: eight named components, weights in config, and every
component stored on the row so a bad pick can be explained after the fact. A learned model
would be more accurate eventually and completely opaque now, with no training data.

Deduplication uses the *story skeleton*, not the topic string. Two topics phrased
differently that would produce the same episode are duplicates, and the thing that catches
that is the beat signature plus the animal/archetype cooldown - not text similarity.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict
from pathlib import Path

from ..core.db import jdump, read, tx
from ..core.logging import get_logger
from .collectors import ANIMALS, ARCHETYPES, Candidate

log = get_logger("scoring")

DEFAULT_WEIGHTS = {"trend": 0.15, "search": 0.12, "competition": 0.13, "emotional": 0.15,
                   "entertainment": 0.10, "story": 0.15, "thumbnail": 0.10,
                   "long_form": 0.10}

# Words that signal a story has somewhere to go emotionally. Crude, but it separates
# "otters hold hands while sleeping" from "new taxonomy of rodent dentition".
EMOTION_WORDS = re.compile(
    r"\b(lost|found|alone|friend|rescue|save|brave|afraid|hope|home|family|lose|"
    r"return|forgive|betray|trust|secret|promise|help|share|first|last|never|"
    r"together|apart|defend|protect|grief|joy|stubborn|proud)\b", re.I)
VISUAL_WORDS = re.compile(
    r"\b(night|storm|fire|snow|river|forest|market|bridge|mountain|cave|garden|"
    r"lantern|flood|tower|ice|desert|rain|moon|bakery|workshop|boat|door)\b", re.I)
CONFLICT_WORDS = re.compile(
    r"\b(but|until|despite|refuse|must|cannot|against|before|unless|race|steal|"
    r"break|missing|wrong|too late|only)\b", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _hits(pattern: re.Pattern, text: str, cap: int = 4) -> float:
    return min(1.0, len(set(m.group(0).lower() for m in pattern.finditer(text))) / cap)


def score_candidate(c: Candidate, animal_cooldown: dict[str, int],
                    archetype_cooldown: dict[str, int],
                    strategy: dict[str, float] | None = None) -> dict:
    text = f"{c.topic} {' '.join(c.keywords)}"
    sig = c.signals

    emotional = _hits(EMOTION_WORDS, text)
    thumbnail = _hits(VISUAL_WORDS, text)
    story = 0.45 * _hits(CONFLICT_WORDS, text) + 0.55 * emotional
    words = len(text.split())
    # A topic needs enough substance to carry seven minutes but not so much that it is
    # really three episodes wearing a trench coat.
    long_form = max(0.0, 1.0 - abs(words - 16) / 22.0)
    entertainment = 0.5 * thumbnail + 0.5 * min(1.0, len(c.keywords) / 6)

    trend = float(sig.get("freshness", 0.0)) * 0.6 + float(sig.get("seasonal", 0.0)) * 0.4
    search = float(sig.get("search_demand", 0.35))
    # competition_score is inverted on purpose: HIGHER means LESS crowded.
    competition = 1.0 - float(sig.get("competition", 0.5))

    scores = {"trend": trend, "search": search, "competition": competition,
              "emotional": emotional, "entertainment": entertainment, "story": story,
              "thumbnail": thumbnail, "long_form": long_form}

    if strategy:
        for key in ("animal:" + (c.primary_animal or ""), "archetype:" + (c.archetype or "")):
            if key in strategy:
                # Analytics nudges, never dictates: +/-0.12 at the extremes.
                bump = max(-0.12, min(0.12, (strategy[key] - 0.5) * 0.24))
                scores["story"] = max(0.0, min(1.0, scores["story"] + bump))

    # Cooldowns are a hard damper, not a weight. Three fox episodes in a row is how a
    # channel reads as a farm even when every individual episode is fine.
    if c.primary_animal and animal_cooldown.get(c.primary_animal, 99) < 3:
        scores["entertainment"] *= 0.45
    if c.archetype and archetype_cooldown.get(c.archetype, 99) < 4:
        scores["story"] *= 0.55
    return {k: round(v, 4) for k, v in scores.items()}


def overall(scores: dict, weights: dict | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    total = sum(w.values()) or 1.0
    return round(sum(scores.get(k, 0.0) * v for k, v in w.items()) / total, 4)


def _skeleton(c: Candidate) -> str:
    """A coarse identity for the story a topic would produce."""
    key = f"{c.primary_animal or '?'}|{c.archetype or '?'}|" + "|".join(
        sorted(set(_norm(c.topic).split()))[:8])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _cooldowns(db: Path) -> tuple[dict[str, int], dict[str, int]]:
    """How many episodes ago each animal / archetype last appeared."""
    with read(db) as con:
        rows = con.execute(
            "SELECT archetype FROM stories ORDER BY id DESC LIMIT 12").fetchall()
        cast = con.execute(
            "SELECT c.species, MAX(s.id) AS last_id FROM story_cast sc "
            "JOIN stories s ON s.id = sc.story_id JOIN characters c ON c.id = sc.character_id "
            "GROUP BY c.species").fetchall()
        newest = con.execute("SELECT MAX(id) FROM stories").fetchone()[0] or 0
    arche = {}
    for i, r in enumerate(rows):
        arche.setdefault(r["archetype"], i)
    animal = {r["species"]: max(0, newest - r["last_id"]) for r in cast}
    return animal, arche


def ingest(db: Path, candidates: list[Candidate], weights: dict | None = None,
           strategy: dict[str, float] | None = None) -> int:
    """Score and store. Duplicates are recorded as `duplicate`, not silently dropped -
    knowing a source keeps producing the same idea is itself useful."""
    animal_cd, arche_cd = _cooldowns(db)
    with read(db) as con:
        seen = {r[0] for r in con.execute(
            "SELECT notes FROM research_topics WHERE notes IS NOT NULL")}
    inserted = 0
    with tx(db) as con:
        for c in candidates:
            skel = _skeleton(c)
            scores = score_candidate(c, animal_cd, arche_cd, strategy)
            status = "duplicate" if skel in seen else "new"
            seen.add(skel)
            con.execute("""
                INSERT INTO research_topics (topic, keywords, primary_animal, archetype,
                    source, source_ref, trend_score, search_score, competition_score,
                    emotional_score, entertainment_score, story_score, thumbnail_score,
                    long_form_score, overall_score, status, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (c.topic[:400], jdump(c.keywords), c.primary_animal, c.archetype,
                  c.source, c.source_ref, scores["trend"], scores["search"],
                  scores["competition"], scores["emotional"], scores["entertainment"],
                  scores["story"], scores["thumbnail"], scores["long_form"],
                  overall(scores, weights), status, skel))
            inserted += 1
    log.info("topics_ingested", n=inserted)
    return inserted


def select_next(db: Path, min_score: float = 0.35) -> dict | None:
    """Highest-scoring unused topic. Returns None rather than lowering the bar."""
    with read(db) as con:
        row = con.execute(
            "SELECT * FROM research_topics WHERE status = 'new' AND overall_score >= ? "
            "ORDER BY overall_score DESC, id DESC LIMIT 1", (min_score,)).fetchone()
    return dict(row) if row else None


def mark_used(db: Path, topic_id: int) -> None:
    with tx(db) as con:
        con.execute("UPDATE research_topics SET status = 'used' WHERE id = ?", (topic_id,))


def mark_rejected(db: Path, topic_id: int, reason: str) -> None:
    with tx(db) as con:
        con.execute("UPDATE research_topics SET status = 'rejected', reject_reason = ? "
                    "WHERE id = ?", (reason[:300], topic_id))
