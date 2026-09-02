"""Score, deduplicate and select topics.

The scoring is transparent on purpose: eight named components, weights in config, and every
component stored on the row so a bad pick can be explained after the fact. A learned model
would be more accurate eventually and completely opaque now, with no training data.

Deduplication is primarily by *story skeleton*, not topic string: two topics phrased
differently that would produce the same episode are duplicates, and what catches that is
the beat signature plus the animal/archetype cooldown, not text similarity.

Identical topic text is checked as well, and that is not redundant. The skeleton folds in
primary_animal, which is a heuristic guess *about* the text rather than part of it, so one
sentence ingested twice can hash two ways and occupy two selectable rows. Skeleton
comparison alone let a job re-select a topic an earlier episode had already been built
from - see mark_used.
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


# Real-world human news, which must never become an animated animal story.
#
# This is not squeamishness, it is the single worst thing this pipeline could do
# unattended: the scorer REWARDS exactly the vocabulary a disaster headline uses. Its
# emotional list is lost/found/alone/family/rescue/save/afraid/hope/home and its visual
# list includes river/flood/storm/fire, so "Family fear for missing hydropower worker after
# Nepal flood" - a real missing person - scored top of a live research run and would have
# been auto-published as a talking-animal short.
#
# Unconditional terms are ones that cannot appear innocently in an animal story.
UNSUITABLE_ALWAYS = re.compile(
    r"\b(murder(ed|s)?|manslaughter|homicide|terror(ism|ist)?|bombing|airstrike|"
    r"genocide|massacre|shooting|stabbed|rape|abuse|arrest(ed)?|convicted|sentenced|"
    r"lawsuit|indicted|election|parliament|president|prime minister|senator|sanctions|"
    r"pandemic|outbreak|suicide|funeral|obituary|war|troops|refugees?|hostage|"
    # Death words are unconditional. Requiring a companion "person" word missed
    # "One still missing after Grand Canyon floods kill two and prompt rescue efforts",
    # which names nobody - the victims are "two" - and so passed a harm+person test while
    # being exactly the kind of story this filter exists to stop.
    r"kill(s|ed|ing)?|dead|deaths?|death toll|died|dies|fatal(ities|ity)?|"
    r"casualt(y|ies)|drowned|perished|slain|corpse)\b", re.I)

# These two only reject TOGETHER. "Rescue" and "lost" are the raw material of the stories
# this channel exists to tell; they are only a problem when the subject is a real person.
UNSUITABLE_HARM = re.compile(
    r"\b(injured|missing|victims?|survivors?|evacuat(ed|ion)|disaster|tragedy|"
    r"crash(ed)?|collapsed?|trapped|stranded|toll|feared)\b", re.I)
UNSUITABLE_PERSON = re.compile(
    r"\b(family|families|man|men|woman|women|child|children|people|persons?|worker(s)?|"
    r"residents?|villagers?|tourists?|driver|students?|teacher|mother|father|son|daughter|"
    r"couple|crew|passengers?|police|officials?|troops|soldiers?|patients?)\b", re.I)


def unsuitable_reason(text: str) -> str | None:
    """Why this topic must not become an episode, or None if it is fine.

    Returns a reason rather than a bool so the rejection is recorded on the row and can be
    explained later - a silently dropped topic is indistinguishable from a scoring miss.
    """
    hit = UNSUITABLE_ALWAYS.search(text)
    if hit:
        return f"real-world news topic ({hit.group(0).lower()})"
    harm, person = UNSUITABLE_HARM.search(text), UNSUITABLE_PERSON.search(text)
    if harm and person:
        return (f"appears to describe real people coming to harm "
                f"({person.group(0).lower()} / {harm.group(0).lower()})")
    return None


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
        # Topic text is compared alongside the skeleton because the skeleton is not a
        # stable identity: it mixes in primary_animal, a heuristic guess about the text.
        # Two ingests of one sentence can disagree about the animal and slip past a
        # skeleton-only check as two distinct topics.
        seen_text = {_norm(r[0]) for r in con.execute(
            "SELECT topic FROM research_topics WHERE topic IS NOT NULL")}
    inserted = 0
    rejected = 0
    with tx(db) as con:
        for c in candidates:
            skel = _skeleton(c)
            text = _norm(c.topic)
            scores = score_candidate(c, animal_cd, arche_cd, strategy)
            unsuitable = unsuitable_reason(f"{c.topic} {' '.join(c.keywords)}")
            if unsuitable:
                status, rejected = "rejected", rejected + 1
            elif skel in seen or text in seen_text:
                status = "duplicate"
            else:
                status = "new"
            seen.add(skel)
            seen_text.add(text)
            con.execute("""
                INSERT INTO research_topics (topic, keywords, primary_animal, archetype,
                    source, source_ref, trend_score, search_score, competition_score,
                    emotional_score, entertainment_score, story_score, thumbnail_score,
                    long_form_score, overall_score, status, notes, reject_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (c.topic[:400], jdump(c.keywords), c.primary_animal, c.archetype,
                  c.source, c.source_ref, scores["trend"], scores["search"],
                  scores["competition"], scores["emotional"], scores["entertainment"],
                  scores["story"], scores["thumbnail"], scores["long_form"],
                  overall(scores, weights), status, skel, unsuitable))
            inserted += 1
    log.info("topics_ingested", n=inserted, rejected=rejected)
    return inserted


def select_next(db: Path, min_score: float = 0.35) -> dict | None:
    """Highest-scoring unused topic. Returns None rather than lowering the bar."""
    with read(db) as con:
        row = con.execute(
            "SELECT * FROM research_topics WHERE status = 'new' AND overall_score >= ? "
            "ORDER BY overall_score DESC, id DESC LIMIT 1", (min_score,)).fetchone()
    return dict(row) if row else None


def mark_used(db: Path, topic_id: int) -> None:
    """Mark a topic used, and retire every other row describing the same story.

    The dedupe key in `notes` folds in primary_animal, which is *derived* from the topic
    text by a heuristic rather than being part of it. The same sentence can therefore hash
    two different ways - it happened here, one row carrying 'fox' and its twin None - and
    both rows stay selectable. Marking only the row that was picked then lets a later job
    select the twin and reproduce an episode the channel has already published, which is
    the one failure an unattended pipeline must not have. Retiring by normalised topic text
    is what actually holds, because that text is the thing the story is generated from.
    """
    with read(db) as con:
        row = con.execute("SELECT topic FROM research_topics WHERE id = ?",
                          (topic_id,)).fetchone()
        others = con.execute(
            "SELECT id, topic FROM research_topics WHERE status = 'new' AND id != ?",
            (topic_id,)).fetchall()
    key = _norm(row["topic"]) if row else ""
    twins = [r["id"] for r in others if key and _norm(r["topic"]) == key]
    with tx(db) as con:
        con.execute("UPDATE research_topics SET status = 'used' WHERE id = ?", (topic_id,))
        for tid in twins:
            con.execute("UPDATE research_topics SET status = 'duplicate', "
                        "reject_reason = ? WHERE id = ?",
                        (f"same topic as used topic {topic_id}", tid))
    if twins:
        log.info("topic_twins_retired", topic_id=topic_id, retired=twins)


def mark_rejected(db: Path, topic_id: int, reason: str) -> None:
    with tx(db) as con:
        con.execute("UPDATE research_topics SET status = 'rejected', reject_reason = ? "
                    "WHERE id = ?", (reason[:300], topic_id))
