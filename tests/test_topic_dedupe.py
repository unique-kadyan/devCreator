"""Regression tests for topic deduplication.

The bug these pin down: an unattended pipeline re-selected a topic an earlier episode had
already been built from, because the same sentence occupied two rows. The dedupe key mixes
in primary_animal, which is a heuristic guess about the topic text rather than part of it,
so one row carried 'fox' and its twin None and the two hashed differently.
"""
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from asa.research.collectors import Candidate
from asa.research.scoring import ingest, mark_used, select_next

TOPIC = "A lost lantern must be returned before the river floods the market bridge at night"


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    sqlite3.connect(p).executescript((ROOT / "migrations/001_initial.sql").read_text())
    return p


def _status(db, topic_id):
    with sqlite3.connect(db) as con:
        return con.execute("SELECT status FROM research_topics WHERE id = ?",
                           (topic_id,)).fetchone()[0]


def test_identical_text_with_different_animal_is_still_a_duplicate(db):
    # The exact shape that escaped the old check: same sentence, disagreeing animal guess.
    ingest(db, [Candidate(topic=TOPIC, keywords=["lantern"], primary_animal="fox")])
    ingest(db, [Candidate(topic=TOPIC, keywords=["lantern"], primary_animal=None)])
    with sqlite3.connect(db) as con:
        statuses = [r[0] for r in con.execute(
            "SELECT status FROM research_topics ORDER BY id")]
    assert statuses == ["new", "duplicate"]


def test_using_a_topic_retires_its_twins(db):
    # Twins can predate the fix, so mark_used has to cope with them already in the table.
    with sqlite3.connect(db) as con:
        for animal in ("fox", None):
            con.execute("INSERT INTO research_topics (topic, primary_animal, source, "
                        "status, overall_score, notes) VALUES (?,?,'manual','new',0.9,?)",
                        (TOPIC, animal, f"skel-{animal}"))
    first = select_next(db)
    mark_used(db, first["id"])
    with sqlite3.connect(db) as con:
        rows = dict(con.execute("SELECT id, status FROM research_topics").fetchall())
    assert rows[first["id"]] == "used"
    twin = next(i for i in rows if i != first["id"])
    assert rows[twin] == "duplicate", "the twin stayed selectable; the episode repeats"
    assert select_next(db) is None


def test_mark_used_leaves_unrelated_topics_selectable(db):
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO research_topics (topic, source, status, overall_score, "
                    "notes) VALUES (?, 'manual', 'new', 0.9, 'a')", (TOPIC,))
        con.execute("INSERT INTO research_topics (topic, source, status, overall_score, "
                    "notes) VALUES ('A badger guards the last bridge lamp', 'manual', "
                    "'new', 0.8, 'b')")
    used = select_next(db)
    mark_used(db, used["id"])
    nxt = select_next(db)
    assert nxt is not None and nxt["id"] != used["id"]
    assert _status(db, nxt["id"]) == "new"
