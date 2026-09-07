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
    # `electrocuted` was added from live data rather than from imagination: "Wild elephant
    # electrocuted in Chittoor district" was sitting among the highest-scoring SELECTABLE
    # topics in the table. It names no person, so the harm+person pair could not catch it,
    # and an animal killed by infrastructure is the single worst thing this channel could
    # turn into a cartoon.
    r"electrocut(ed|ion)|mauled|poach(ed|ing)|culled?|"
    # `deadly` is NOT caught by `\bdead\b` - the word boundary sees "deadly" as its own
    # word - and that gap put "How a mountain collapse triggered Nepal's deadly flood in
    # just 7 minutes" SECOND in the selectable pool on a live run. A fatal disaster is the
    # exact story this filter exists to keep off an animated animal channel, and it got
    # there through one missing suffix.
    r"deadly|kill(s|ed|ing)?|dead|deaths?|death toll|died|dies|fatal(ly|ities|ity)?|"
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


# Local relevance. Adding Indian feeds put Indian stories in the CANDIDATE pool but nothing
# preferred them once there, so a Madagascar frog discovery still outscored a story about
# animals the audience lives beside. This is a nudge, not a filter: a genuinely great
# non-Indian wildlife fact should still win, it just no longer wins by default.
REGION_WORDS = re.compile(
    r"\b(india|indian|bharat|delhi|mumbai|kolkata|chennai|bengaluru|bangalore|hyderabad|"
    r"pune|jaipur|lucknow|kerala|punjab|gujarat|rajasthan|assam|odisha|bihar|goa|"
    r"maharashtra|karnataka|tamil nadu|bengal|himalaya|himalayan|ganga|ganges|yamuna|"
    r"western ghats|sundarbans|thar|deccan|monsoon|"
    r"tiger|elephant|peacock|langur|mongoose|nilgai|blackbuck|sloth bear|"
    r"gharial|macaque|leopard|banyan|bullock|buffalo)\b", re.I)

REGION_BONUS = 0.12

# Curiosity, which is what an explainer has instead of an emotional arc. The words are the
# shape of a question a viewer wants closed - a first, an oldest, a how, a thing that
# survived or was worked out without the tool you would expect.
CURIOSITY_WORDS = re.compile(
    r"\b(first|oldest|earliest|how|why|discovered|invented|measured|calculated|proved|"
    r"predicted|built|survived|preserved|hidden|ancient|secret|mystery|puzzle|"
    r"before|without|still|centuries|thousand)\b", re.I)

# What a factual topic gets back, and it is large on purpose. MEASURED, not chosen: five
# real subject candidates from the live collector scored 0.207 to 0.254 against
# select_next's 0.35 floor, so NONE of them could ever be selected - the channel could be
# configured to cover ancient technology and would still never make an episode about it.
#
# The deficit is structural rather than a judgement about quality. `emotional` and `story`
# are measured by matching a fiction vocabulary (lost, promise, betray, must, until) against
# the topic SENTENCE, and a subject candidate's sentence is an encyclopedia extract - it
# scores near zero however good the episode would be. The bonus therefore lands on exactly
# those two components, scaled by how much curiosity the sentence actually carries, rather
# than being spread across `thumbnail` and `entertainment` where the measurement was not
# broken.
#
# Gated on `signals["subject"]`, which ONLY `collect_subjects` sets. The animal-fact
# collector sets `factual` and is deliberately not lifted: it produces fiction seeded by a
# true detail, and it scores fine on the fiction axes because that is what it is.
SUBJECT_BONUS = 0.55
# Floor, so a flatly-worded true sentence is not shut out for lacking the vocabulary - the
# same failure the bonus exists to fix, one level down.
SUBJECT_CURIOSITY_FLOOR = 0.45


def region_score(text: str) -> float:
    """0..1 for how strongly a topic is rooted in the channel's own region."""
    hits = len(set(m.group(0).lower() for m in REGION_WORDS.finditer(text)))
    return min(1.0, hits / 2.0)


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

    # Region relevance rides on `story` rather than becoming a ninth weighted component:
    # the weights are in config and documented, and silently changing their arithmetic
    # would make every historical score incomparable with every new one.
    scores["story"] = min(1.0, scores["story"] + REGION_BONUS * region_score(text))

    # Real subject matter, for the same structural reason and by the same mechanism. See
    # SUBJECT_BONUS: the fiction vocabulary cannot measure an encyclopedia sentence, so
    # `emotional` and `story` read near zero for reasons that have nothing to do with how
    # good the episode would be.
    subject = float(sig.get("subject", 0.0))
    if subject > 0:
        curiosity = SUBJECT_CURIOSITY_FLOOR + (1.0 - SUBJECT_CURIOSITY_FLOOR) * _hits(
            CURIOSITY_WORDS, text)
        lift = SUBJECT_BONUS * subject * curiosity
        scores["emotional"] = min(1.0, scores["emotional"] + lift)
        scores["story"] = min(1.0, scores["story"] + lift)

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


# A collected topic is a headline - an RSS title, a search result - and 400 characters is
# already generous for one. A MANUAL topic is something a person typed, and for an ad or a
# commissioned episode that is a brief: the product, the beats it has to hit, the wording
# that must survive. Truncating it to 400 silently deleted a RoleVo ad's entire product
# description and produced a story that never mentioned the product at all.
#
# So the cap depends on where the text came from, and passing it is LOGGED either way. A
# brief that quietly loses its second half is far worse than one that is refused.
COLLECTED_TOPIC_CHARS = 400
MANUAL_BRIEF_CHARS = 4000


def _topic_text(c) -> str:
    limit = MANUAL_BRIEF_CHARS if c.source == "manual" else COLLECTED_TOPIC_CHARS
    if len(c.topic) <= limit:
        return c.topic
    log.warning("topic_truncated", source=c.source, was=len(c.topic), kept=limit)
    return c.topic[:limit]


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
            """, (_topic_text(c), jdump(c.keywords), c.primary_animal, c.archetype,
                  c.source, c.source_ref, scores["trend"], scores["search"],
                  scores["competition"], scores["emotional"], scores["entertainment"],
                  scores["story"], scores["thumbnail"], scores["long_form"],
                  overall(scores, weights), status, skel, unsuitable))
            inserted += 1
    log.info("topics_ingested", n=inserted, rejected=rejected)
    return inserted


# The selection floor. 0.35 predates both bonuses and any real measurement, and on a live
# table of 306 collected topics the highest-scoring SELECTABLE one reached 0.347 - so the
# floor rejected the entire pool and every episode ever made on this install came from a
# hand-typed `--topic`. It is left at 0.35 anyway rather than quietly lowered, because what
# sits immediately below it is not merely weaker material: at 0.347 the table also holds
# "Wild elephant electrocuted in Chittoor district; Pawan Kalyan orders inquiry", a real
# news item naming a real politician. Lowering the bar admits those, so it is the
# operator's call and `research.min_score` is where they make it.
DEFAULT_MIN_SCORE = 0.35


def _first_still_suitable(rows) -> tuple[dict | None, list[dict]]:
    """The best row today's filter still accepts, plus the ones it no longer does.

    `ingest` applies `unsuitable_reason` when a row is INSERTED, so a row stored last week
    carries last week's verdict forever. That is fine until the filter is tightened, which
    is exactly when it matters: adding `deadly` to the pattern did nothing about the row
    already sitting SECOND in the selectable pool - "How a mountain collapse triggered
    Nepal's deadly flood in just 7 minutes" - because nothing re-reads it.

    So the filter runs again at the moment of use, and anything it now rejects is retired
    with its reason rather than merely skipped. Cheap - one regex over at most ten rows -
    and it means every future tightening cleans the standing pool, not just the next
    collection.
    """
    chosen: dict | None = None
    rejected: list[dict] = []
    # Every fetched row is examined, not just the ones above the winner. Returning at the
    # first acceptable topic would leave a newly-unsuitable row sitting just below it,
    # still `new`, waiting to be selected the moment the winner is used - which is the
    # whole failure this re-check exists to close, deferred by exactly one episode.
    for raw in rows:
        row = dict(raw)
        why = unsuitable_reason(f"{row['topic']} {row.get('keywords') or ''}")
        if not why:
            if chosen is None:
                chosen = row
            continue
        log.warning("stored_topic_now_unsuitable", topic_id=row["id"], reason=why,
                    topic=str(row["topic"])[:100])
        rejected.append({"id": row["id"], "reason": why})
    return chosen, rejected


def _retire(db: Path, rejected: list[dict]) -> None:
    if not rejected:
        return
    with tx(db) as con:
        for row in rejected:
            con.execute("UPDATE research_topics SET status='rejected', reject_reason=? "
                        "WHERE id=?", (row["reason"], row["id"]))


def select_next(db: Path, min_score: float = DEFAULT_MIN_SCORE) -> dict | None:
    """Highest-scoring unused topic. Returns None rather than lowering the bar."""
    with read(db) as con:
        rows = con.execute(
            "SELECT * FROM research_topics WHERE status = 'new' AND overall_score >= ? "
            "ORDER BY overall_score DESC, id DESC LIMIT 10", (min_score,)).fetchall()
    chosen, rejected = _first_still_suitable(rows)
    _retire(db, rejected)
    return chosen


# Collectors whose candidates are ABOUT something real rather than seeded by it.
SUBJECT_SOURCES = ("wikipedia_subject",)


def select_subject(db: Path, min_score: float = 0.0) -> dict | None:
    """The best unused topic that is about real subject matter.

    A separate door rather than a thumb on the general scale, and the measurement is why.
    Even with SUBJECT_BONUS a factual candidate lands around 0.30-0.35 while a strong RSS
    wildlife story reaches 0.40, so on a mixed pool the subject topics compete honestly and
    honestly lose most days. Rigging the scorer until they win would mean deciding, in
    arithmetic nobody reads, that this is now a science channel.

    `min_score` defaults to 0 because the caller has already made the editorial choice by
    asking for a subject at all - the floor exists to stop a WEAK topic being picked when a
    better one was available, and here there is no better one by definition. Unsuitable
    topics are still excluded: `ingest` marks those `rejected`, and this only reads `new`.
    """
    placeholders = ",".join("?" * len(SUBJECT_SOURCES))
    with read(db) as con:
        rows = con.execute(
            f"SELECT * FROM research_topics WHERE status = 'new' "
            f"AND source IN ({placeholders}) AND overall_score >= ? "
            f"ORDER BY overall_score DESC, id DESC LIMIT 10",
            (*SUBJECT_SOURCES, min_score)).fetchall()
    chosen, rejected = _first_still_suitable(rows)
    _retire(db, rejected)
    return chosen


def near_miss(db: Path, min_score: float = DEFAULT_MIN_SCORE) -> dict | None:
    """The best topic the floor just rejected, so a run that selects nothing can say WHY.

    Without this, `select_next` returning None is indistinguishable from an empty table,
    and the operator's next move differs completely between the two: collect more, or lower
    a threshold that is rejecting everything it is shown.
    """
    with read(db) as con:
        row = con.execute(
            "SELECT * FROM research_topics WHERE status = 'new' AND overall_score < ? "
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
