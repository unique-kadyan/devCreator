"""Real-world news must never become an animated animal story.

This is the worst thing the pipeline could do unattended, and the scoring model actively
steers into it: the emotional keywords are lost/found/alone/family/rescue/save/afraid/hope
and the visual ones include river/flood/storm/fire, so a disaster headline scores high on
every axis. "Family fear for missing hydropower worker after Nepal flood" - a real missing
person - came top of a live research run.
"""
import sqlite3
from pathlib import Path

import pytest

from asa.research.collectors import Candidate
from asa.research.scoring import ingest, select_next, unsuitable_reason

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    sqlite3.connect(p).executescript((ROOT / "migrations/001_initial.sql").read_text())
    return p


class TestRejected:
    @pytest.mark.parametrize("headline", [
        "Family fear for missing hydropower worker after Nepal flood",
        "Man dies after dog attack in village",
        "Death toll rises as residents evacuated from flooded town",
        "Three children missing after boat capsizes",
        "Police arrest man over stabbing",
        "President announces sanctions after airstrike",
        "Funeral held for victims of the crash",
        # Named nobody - the victims are "two" - so a harm+person test passed it. It came
        # top of a live run AFTER the first version of this filter shipped.
        "One still missing after Grand Canyon floods kill two and prompt rescue efforts",
        "Wildfire kills three in coastal town",
        "Two dead after building collapse",
    ])
    def test_human_news_is_rejected(self, headline):
        assert unsuitable_reason(headline) is not None, headline

    def test_reason_is_specific_enough_to_audit(self):
        reason = unsuitable_reason("Family fear for missing worker after flood")
        assert "family" in reason and "missing" in reason


class TestKept:
    @pytest.mark.parametrize("topic", [
        "A lost lantern must be returned before the river floods the market bridge at night",
        "Otters hold hands while sleeping so they do not drift apart",
        "A fox and an owl must cross the frozen river before dawn",
        "Rescue dog saves family from house fire",
        "A hedgehog searches for a lost friend in the winter garden",
        "The trickster raccoon steals the baker's last loaf",
        "Scientists tracked 4,500 animals and found a surprising human effect",
        "Otters use tools to open shellfish along the river bank",
    ])
    def test_story_material_survives(self, topic):
        # rescue / lost / saves are the raw material of this channel; they must only
        # trigger when the subject is a real person.
        assert unsuitable_reason(topic) is None, topic


class TestIngest:
    def test_unsuitable_topics_are_never_selectable(self, db):
        ingest(db, [
            Candidate(topic="Family fear for missing hydropower worker after Nepal flood",
                      keywords=["flood", "nepal"], source="rss"),
            Candidate(topic="A fox guards a lantern on the market bridge at night",
                      keywords=["fox", "lantern", "bridge"], source="rss"),
        ])
        chosen = select_next(db, min_score=0.0)
        assert chosen is not None
        assert "fox" in chosen["topic"]

    def test_rejection_reason_is_recorded_on_the_row(self, db):
        ingest(db, [Candidate(topic="Man dies after landslide", keywords=[],
                              source="rss")])
        with sqlite3.connect(db) as con:
            con.row_factory = sqlite3.Row
            row = dict(con.execute(
                "SELECT status, reject_reason FROM research_topics").fetchone())
        assert row["status"] == "rejected"
        assert row["reject_reason"], "a silent drop is indistinguishable from a scoring miss"
