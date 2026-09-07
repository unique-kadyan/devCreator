"""A commissioned episode must say what it was commissioned to say.

The failure these pin down: an ad brief describing a product, its features and its tagline
was stored truncated at 400 characters, and what survived was passed to the model as a
"SEED TOPIC" - a starting point it is explicitly invited to develop away from. The script
came back without the product in it at all. Nothing errored; the pipeline had been asked
for a story and it wrote one.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.research.collectors import Candidate                          # noqa: E402
from asa.research.scoring import (COLLECTED_TOPIC_CHARS,               # noqa: E402
                                  MANUAL_BRIEF_CHARS, ingest)
from asa.story.prompts import (brief_block, draft_prompt,              # noqa: E402
                               outline_prompt, scenes_prompt)

BRIEF = ("A 60-second advertisement. " + "x" * 900 +
         " The product is RoleVo and its tagline is 'Apply less. Interview more.'")


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    sqlite3.connect(p).executescript((ROOT / "migrations/001_initial.sql").read_text())
    return p


def stored_topic(db) -> str:
    with sqlite3.connect(db) as con:
        return con.execute("SELECT topic FROM research_topics ORDER BY id DESC").fetchone()[0]


def test_a_hand_written_brief_survives_ingest_whole(db):
    """The truncation that deleted a product description. 400 characters is a headline
    budget; a brief is not a headline."""
    assert len(BRIEF) > COLLECTED_TOPIC_CHARS
    ingest(db, [Candidate(topic=BRIEF, source="manual")])
    assert stored_topic(db) == BRIEF
    assert "RoleVo" in stored_topic(db)


def test_a_collected_topic_is_still_capped(db):
    """An RSS headline has no business being four thousand characters long."""
    ingest(db, [Candidate(topic="y" * 1200, source="rss")])
    assert len(stored_topic(db)) == COLLECTED_TOPIC_CHARS


def test_even_a_brief_has_a_ceiling(db):
    ingest(db, [Candidate(topic="z" * (MANUAL_BRIEF_CHARS + 500), source="manual")])
    assert len(stored_topic(db)) == MANUAL_BRIEF_CHARS


# ------------------------------------------------------------------ prompts

def test_no_brief_leaves_the_ordinary_prompt_untouched():
    """Storytelling is the normal case and must not be turned into contract fulfilment."""
    p = outline_prompt("a clever fox opens a bakery", [], 7.0, [], [], "", "", "", 2,
                       ["comedy"])
    assert "BRIEF" not in p
    assert brief_block("") == "" and brief_block("   ") == ""


def test_the_brief_outranks_the_prompt_s_own_preferences():
    """The outline prompt is full of competing preferences - reuse this cast, prefer an
    under-used archetype. A brief that is merely one more preference loses to them."""
    block = brief_block("Sell RoleVo.")
    assert "THE BRIEF WINS" in block
    assert "MUST appear" in block


def test_the_brief_is_restated_in_all_three_calls():
    """A requirement stated only in the outline has been through two summarisations before
    anyone writes a line of dialogue."""
    brief = "The product is RoleVo."
    assert brief in outline_prompt("seed", [], 1.0, [], [], "", "", "", 2, ["comedy"],
                                   brief=brief)
    assert brief in draft_prompt("{}", 1.0, brief=brief)
    assert brief in scenes_prompt("{}", "{}", [], [], [], "style", 1.0, brief=brief)


def test_the_brief_protects_a_brand_from_transliteration():
    """The channel writes in Hindi; a brand written in Devanagari is a different brand."""
    assert "Latin letters" in brief_block("Sell RoleVo.")


def test_the_brief_forbids_inventing_product_claims():
    """An ad that promises what the product does not do is a liability, and a free model
    asked to be persuasive will happily invent a statistic."""
    block = brief_block("Sell RoleVo.")
    assert "Do not invent capabilities" in block
