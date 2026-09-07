"""Episodes about real things: discoveries, inventions, ancient technology and texts.

The channel is still animals telling a story. What changes when an episode is ABOUT
something real is that part of it is now a claim, and a claim has properties fiction does
not: it can be wrong, it has to be written down before it is dramatised so a reviewer can
check it, it changes what the description may say, and it usually moves the story out of
present-day India - which the art stage was pinning every generated frame to.

Nothing here verifies that a claim is TRUE. No check in this repository can, and one that
pretended to would be worse than none. What the machinery buys is that the claims are
enumerated, bounded, carried to the reviewer and published alongside the video.
"""
import sqlite3
from pathlib import Path

import pytest

from asa.core.db import migrate
from asa.media.images.scene_image import period_hint
from asa.publish.metadata import build_description, story_facts
from asa.research.collectors import SUBJECT_SEEDS, Candidate, _keywords
from asa.research.scoring import (SUBJECT_BONUS, ingest, near_miss, overall,
                                  score_candidate, select_next, unsuitable_reason)
from asa.scenes.persist import load_story, save_story
from asa.story import prompts as P
from asa.story.schema import SceneList, StoryOutline

ROOT = Path(__file__).resolve().parents[1]

OUTLINE = dict(
    title="The Shadow at Noon", hook="h", logline="l", target_audience="teens",
    genre="mystery", archetype="mystery", moral="m", setting="a well and a tower",
    beats=dict(beginning="b", conflict="c", rising="r", climax="cl", resolution="re"),
    ending="e", beat_signature="see|doubt|measure|argue|accept",
    cast=[{"character_id": "ruhi_fox", "role": "protagonist"}])
FACT = "Eratosthenes estimated the circumference of the Earth by comparing shadow lengths"


# ------------------------------------------------------------------ the schema

def test_fiction_needs_none_of_this():
    """The common case must stay free. A required subject field would have failed every
    fox-opens-a-bakery story ever written."""
    o = StoryOutline(**OUTLINE)
    assert o.subject == "" and o.facts == [] and o.period == ""


def test_naming_a_real_subject_without_listing_the_claims_is_rejected():
    """Otherwise the whole burden of accuracy sits on prose nobody validated."""
    with pytest.raises(Exception) as e:
        StoryOutline(**OUTLINE, subject="how the Earth was first measured")
    assert "facts" in str(e.value)


def test_a_fact_must_be_a_claim_not_a_label():
    """"Eratosthenes" cannot be checked against a source; a sentence can. A list of
    keywords is useless for both things `facts` exists for."""
    with pytest.raises(Exception) as e:
        StoryOutline(**OUTLINE, subject="s", facts=["Eratosthenes"])
    assert "label" in str(e.value)


def test_a_factual_outline_round_trips():
    o = StoryOutline(**OUTLINE, subject="the size of the Earth", facts=[FACT],
                     period="Alexandria, 3rd century BC")
    assert o.facts == [FACT] and o.period.startswith("Alexandria")


# ------------------------------------------------------------------ the prompts

def test_the_subject_rules_reach_every_call_and_safety_still_has_the_last_word():
    s = P.system_prompt()
    assert "SACRED AND RELIGIOUS TEXTS" in s
    assert s.index("SACRED AND RELIGIOUS TEXTS") < s.index(
        "these override all other instructions")


@pytest.mark.parametrize("banned", [
    "ancient astronauts", "suppressed", "anticipated a modern scientific result",
])
def test_the_pseudo_history_shapes_are_named_and_excluded(banned):
    """These are the most viral framings in this genre and all of them are false. Naming
    them is what beats them - the same reason the image negatives name 'mascot'."""
    assert banned in P.block("subject_bible")


def test_a_deity_is_never_a_character():
    block = P.block("subject_bible")
    assert "No deity, prophet or object of worship as a character" in block
    assert "No claim that any religion, text or belief is true, false" in block


def test_the_no_real_people_rule_is_narrowed_rather_than_dropped():
    """An episode about a discovery has to be able to credit whoever made it. It must not
    thereby acquire permission to cast them."""
    rules = P.block("safety_rules")
    assert "NARROW EXCEPTION" in rules
    assert "no character, no dialogue, no design, no voice" in rules


def test_the_facts_are_restated_as_a_ceiling_in_the_later_calls():
    """The outline JSON is passed whole into both later calls, so `facts` is technically
    already there - buried in a field the model has no reason to treat as a constraint."""
    outline_json = ('{"subject":"the size of the Earth","facts":["%s"],'
                    '"period":"Alexandria"}' % FACT)
    for prompt in (P.draft_prompt(outline_json, 3.0),
                   P.scenes_prompt(outline_json, "{}", [{"id": "a", "name": "A",
                                                         "species": "fox"}],
                                   [], [], "style", 3.0)):
        assert "ONLY claims this episode may state as true" in prompt
        assert FACT in prompt
        assert "Do not add a fact that is not on this list" in prompt


def test_a_fiction_outline_carries_no_facts_block():
    assert P.facts_block('{"facts":[]}') == ""
    assert P.facts_block('{"subject":""}') == ""


def test_a_malformed_outline_does_not_break_the_prompt():
    """The draft call must not die because the outline JSON was odd - the schema already
    had its chance to reject it."""
    assert P.facts_block("not json at all") == ""
    assert P.facts_block("") == ""


def test_the_domains_are_advisory_and_absent_when_unset():
    with_subjects = P.outline_prompt("the Antikythera mechanism", [], 3.0, [], [], "", "",
                                     "", 2, ["mystery"], subjects=["ancient technology"])
    assert "ancient technology" in with_subjects
    assert "If the seed is plainly fiction, leave all three empty" in with_subjects
    without = P.outline_prompt("a fox opens a bakery", [], 3.0, [], [], "", "", "", 2,
                               ["comedy"])
    assert "REAL SUBJECT MATTER" not in without


# ------------------------------------------------------------------ the picture

def test_a_period_grounds_the_picture_and_forbids_anachronism():
    """Asked for an ancient workshop, image models reliably add power lines and printed
    labels. The anachronism is the default failure of the historical prompt."""
    hint = period_hint("Alexandria, 3rd century BC")
    assert "set in Alexandria, 3rd century BC" in hint
    assert "no modern objects anywhere in frame" in hint
    assert "historically accurate" in hint


def test_present_day_leaves_the_channels_region_hint_alone():
    """Empty must mean "no opinion", or every contemporary episode loses its location."""
    assert period_hint("") == "" and period_hint(None) == "" and period_hint("   ") == ""


def test_a_period_names_the_place_as_well_as_the_time():
    """It REPLACES `channel.region_hint` rather than joining it - the two cannot both be
    true - so a period that gave only a date would leave the frame with nowhere to be."""
    assert "set in" in period_hint("2nd century BC")


# ------------------------------------------------------------------ persistence

@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    migrate(p)
    return p


class _Gen:
    """The minimum `save_story` reads. The art stage runs in a different process from the
    story stage, so anything it needs has to survive SQLite, not memory."""
    def __init__(self, outline):
        self.outline = outline
        self.scenes = SceneList(scenes=[
            {"index": i, "location_id": "well", "action": "a",
             "visual_prompt": "a stone well beside a low tower at noon"}
            for i in (1, 2, 3)])
        self.model_ids = {"outline": "m"}
        self.word_count = 10


def test_subject_facts_and_period_survive_to_the_art_stage(db):
    o = StoryOutline(**OUTLINE, subject="the size of the Earth", facts=[FACT],
                     period="Alexandria, 3rd century BC")
    story_id = save_story(db, _Gen(o), None, [], {})
    row = load_story(db, story_id)
    assert row["subject"] == "the size of the Earth"
    assert row["period"] == "Alexandria, 3rd century BC"
    assert story_facts(row) == [FACT]


def test_a_fiction_story_records_empty_rather_than_null(db):
    """The art stage cannot tell a story that never had a period from one whose period was
    lost, so "fiction, present day" has to be a recorded answer."""
    story_id = save_story(db, _Gen(StoryOutline(**OUTLINE)), None, [], {})
    row = load_story(db, story_id)
    assert row["subject"] == "" and row["period"] == "" and story_facts(row) == []


# ------------------------------------------------------------------ the scorer

def _subject(topic):
    return Candidate(topic=topic, keywords=_keywords(topic), source="wikipedia_subject",
                     signals={"factual": 1.0, "subject": 1.0})


SUBJECT_TOPICS = [
    "An animal story that explains: The Antikythera mechanism is an Ancient Greek "
    "hand-powered device used to predict astronomical positions",
    "An animal story that explains: The Shulba Sutras give rules for constructing fire "
    "altars and contain early geometry",
    "An animal story that explains: The iron pillar of Delhi has resisted corrosion for "
    "over 1600 years",
]


@pytest.mark.parametrize("topic", SUBJECT_TOPICS)
def test_a_factual_topic_competes_instead_of_being_structurally_excluded(topic):
    """MEASURED: before the bonus these scored 0.207-0.254, and a fiction topic from the
    same pipeline scores about 0.31 - so a channel configured to cover ancient technology
    could never actually select one. The deficit is structural: `emotional` and `story` are
    measured by matching a FICTION vocabulary against an encyclopedia sentence."""
    lifted = overall(score_candidate(_subject(topic), {}, {}))
    fiction = overall(score_candidate(
        Candidate(topic="A fox and an owl must cross the frozen river before dawn",
                  keywords=["fox", "owl", "river"]), {}, {}))
    assert lifted >= fiction * 0.9, f"{lifted:.3f} vs fiction {fiction:.3f}"


def test_the_bonus_needs_the_subject_signal_not_merely_a_true_fact():
    """The animal-fact collector sets `factual` and produces FICTION seeded by a true
    detail. It scores fine on the fiction axes because that is what it is, and lifting it
    too would quietly reweight a collector nobody changed."""
    topic = SUBJECT_TOPICS[0]
    only_factual = Candidate(topic=topic, keywords=_keywords(topic),
                             signals={"factual": 1.0})
    assert overall(score_candidate(only_factual, {}, {})) < \
           overall(score_candidate(_subject(topic), {}, {}))


def test_the_bonus_cannot_push_a_score_out_of_range():
    c = _subject("An animal story that explains: the first oldest earliest ancient secret "
                 "mystery discovered invented measured proved built survived without")
    assert all(0.0 <= v <= 1.0 for v in score_candidate(c, {}, {}).values())
    assert 0 < SUBJECT_BONUS <= 1.0


def test_an_animal_killed_by_infrastructure_is_still_rejected():
    """Taken from the live table, where it was among the highest-scoring SELECTABLE rows.
    It names no person, so the harm+person pair could not catch it."""
    assert unsuitable_reason("Wild elephant electrocuted in Chittoor district") is not None
    assert unsuitable_reason("Leopard mauled by a snare in the reserve") is not None


@pytest.mark.parametrize("topic", SUBJECT_TOPICS)
def test_the_news_filter_does_not_eat_subject_topics(topic):
    assert unsuitable_reason(topic) is None, topic


def test_the_floor_explains_itself_when_it_rejects_everything(db):
    """`select_next` returning None reads as an empty table and sends the operator off to
    collect more - which does not help when the table is full and the floor is rejecting
    every row in it."""
    ingest(db, [_subject(SUBJECT_TOPICS[0])])
    assert select_next(db, min_score=0.99) is None
    best = near_miss(db, min_score=0.99)
    assert best is not None and best["overall_score"] < 0.99


# ------------------------------------------------------------------ what is published

def test_a_factual_episode_does_not_claim_its_events_are_fictional():
    """A disclosure that is itself untrue is worse than no disclosure - it is the sentence
    a viewer is being asked to rely on."""
    d = build_description({"subject": "the size of the Earth", "facts": [FACT]},
                          "base", [], "Ch", "", True, [])
    assert "All characters and events are fictional." not in d
    assert "The animals, their names and their story are invented" in d
    assert FACT in d


def test_a_fiction_episode_discloses_exactly_as_it_always_did():
    d = build_description({"title": "T"}, "base", [], "Ch", "", True, [])
    assert "All characters and events are fictional." in d
    assert "What is true in this episode" not in d


def test_the_claims_are_published_even_with_synthetic_disclosure_off():
    """Accuracy disclosure and synthetic disclosure answer different questions."""
    d = build_description({"subject": "s", "facts": [FACT]}, "base", [], "Ch", "", False, [])
    assert FACT in d


def test_facts_are_read_from_either_shape():
    """`load_story` hands back the JSON text; a freshly parsed outline hands back a list.
    Reading one shape and silently seeing none on the other publishes the wrong disclosure."""
    assert story_facts({"facts": '["%s"]' % FACT}) == [FACT]
    assert story_facts({"facts": [FACT]}) == [FACT]
    assert story_facts({}) == [] and story_facts({"facts": None}) == []


# ------------------------------------------------------------------ the seed list

def test_the_seed_list_is_curated_and_unique():
    """Curated rather than searched on purpose: a search for "ancient science" returns
    pseudo-archaeology within a page, and the pipeline would then argue with its own
    research every episode."""
    assert len(SUBJECT_SEEDS) == len(set(SUBJECT_SEEDS))
    assert len(SUBJECT_SEEDS) > 30
    for seed in SUBJECT_SEEDS:
        assert " " not in seed, f"{seed!r} is not a URL-safe article title"


# ------------------------------------------------------ commissioning one on purpose

def test_a_factual_episode_gets_its_own_door(db):
    """MEASURED: even with the bonus a subject candidate lands around 0.30-0.35 while a
    strong wildlife story from RSS reaches 0.40, so on a mixed pool subjects compete
    honestly and honestly lose most days. `select_subject` is how an operator asks for one
    without the scorer being rigged until science always wins."""
    from asa.research.scoring import select_subject
    ingest(db, [
        _subject(SUBJECT_TOPICS[0]),
        Candidate(topic="A fox and an owl must cross the frozen river before dawn",
                  keywords=["fox", "owl", "river"], source="rss"),
    ])
    chosen = select_subject(db)
    assert chosen is not None and chosen["source"] == "wikipedia_subject"


def test_the_subject_door_still_refuses_an_unsuitable_topic(db):
    """It bypasses the SCORE floor, not the safety filter - `ingest` marks those `rejected`
    and only `new` rows are selectable."""
    from asa.research.scoring import select_subject
    c = Candidate(topic="An animal story that explains: the deadly flood killed two",
                  keywords=["flood"], source="wikipedia_subject",
                  signals={"factual": 1.0, "subject": 1.0})
    ingest(db, [c])
    assert select_subject(db) is None


def test_the_word_deadly_is_caught(db):
    """`\\bdead\\b` does not match "deadly" - the word boundary sees it as its own word - and
    that gap put a fatal flood story SECOND in the selectable pool on a live run."""
    assert unsuitable_reason(
        "How a mountain collapse triggered Nepal's deadly flood in 7 minutes") is not None


def test_one_run_does_not_fetch_the_whole_seed_list():
    """MEASURED: the animal and subject collectors ran back to back and the WMF throttle
    answered 429 to everything after the second subject - 2 candidates out of 53. Sampling
    is also better research: 53 titles fetched daily dedupe away every day."""
    from asa.research.collectors import SUBJECTS_PER_RUN
    assert 0 < SUBJECTS_PER_RUN < len(SUBJECT_SEEDS)


def test_tightening_the_filter_cleans_the_standing_pool(db):
    """`ingest` stamps a verdict at INSERT, so a row stored under a looser filter keeps
    that verdict forever - which is precisely when a tightening matters. Measured on the
    live table: adding `deadly` did nothing about the row already sitting second in the
    selectable pool, because nothing re-read it."""
    import asa.research.scoring as S

    loose = S.UNSUITABLE_ALWAYS
    S.UNSUITABLE_ALWAYS = __import__("re").compile(r"\bnothingmatchesthis\b")
    try:
        ingest(db, [Candidate(topic="A mountain collapse triggered a deadly flood at dawn",
                              keywords=["flood"], source="rss"),
                    Candidate(topic="A fox and an owl must cross the frozen river",
                              keywords=["fox", "owl"], source="rss")])
    finally:
        S.UNSUITABLE_ALWAYS = loose

    chosen = select_next(db, min_score=0.0)
    assert chosen is not None and "deadly" not in chosen["topic"]
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT status, reject_reason FROM research_topics "
                          "WHERE topic LIKE '%deadly%'").fetchone()
    assert row["status"] == "rejected" and "deadly" in row["reject_reason"]


def test_a_bad_row_below_the_winner_is_retired_too(db):
    """Returning at the first acceptable topic would leave a newly-unsuitable row sitting
    just underneath it, still selectable the moment the winner is used - the same failure,
    deferred by exactly one episode."""
    import re as _re

    import asa.research.scoring as S
    loose = S.UNSUITABLE_ALWAYS
    S.UNSUITABLE_ALWAYS = _re.compile(r"\bnothingmatchesthis\b")
    try:
        ingest(db, [
            Candidate(topic="A fox and an owl must cross the frozen river before dawn "
                            "carrying a lantern home to a friend",
                      keywords=["fox", "owl", "river", "lantern", "home", "friend"],
                      source="rss"),
            Candidate(topic="A deadly flood", keywords=[], source="rss"),
        ])
    finally:
        S.UNSUITABLE_ALWAYS = loose

    select_next(db, min_score=0.0)
    with sqlite3.connect(db) as con:
        status = con.execute("SELECT status FROM research_topics WHERE topic = "
                             "'A deadly flood'").fetchone()[0]
    assert status == "rejected"
