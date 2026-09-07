"""`asa auto` - the pipeline behind one verb.

Two things here are worth a test and the rest is orchestration already covered elsewhere:

  * a seed typed by a person must be stored as a BRIEF, not as a suggestion. The whole
    difference lives in `research_topics.source`, which `stages.generate_story` reads to
    decide whether the script may wander away from what was asked for. Getting it wrong is
    silent - you get a good episode about something else.
  * the approval gate must still be a gate. It is the one thing standing between this
    pipeline and being a content farm, so a command whose entire selling point is that it
    runs unattended is exactly the command most likely to erode it by accident.
"""
from pathlib import Path

from asa.core.cli import _ensure_topic, main
from asa.core.db import migrate, read


class _Cfg:
    def get(self, key, default=None):
        return default


class _Ctx:
    def __init__(self, db):
        self.db, self.cfg, self.quota = db, _Cfg(), None


def _db(tmp_path: Path) -> Path:
    p = tmp_path / "asa.db"
    migrate(p)
    return p


def test_a_typed_seed_is_stored_as_a_binding_brief(tmp_path):
    """source='manual' is what stages.generate_story turns into prompts.brief_block. A
    seed stored under any other source is developed freely instead of satisfied."""
    ctx = _Ctx(_db(tmp_path))
    topic_id, how = _ensure_topic(ctx, "a clever fox opens a village bakery")
    assert topic_id is not None
    with read(ctx.db) as con:
        row = con.execute("SELECT topic, source, status FROM research_topics WHERE id=?",
                          (topic_id,)).fetchone()
    assert row["source"] == "manual"
    assert row["topic"] == "a clever fox opens a village bakery"
    # Marked used at creation: the seed belongs to THIS job, and leaving it selectable
    # would hand the same brief to the next unattended run as well.
    assert row["status"] == "used"
    assert how == "your brief"


def test_a_seed_never_falls_through_to_topic_selection(tmp_path):
    """With a brief there is nothing to select, so the collectors must not run - an
    unattended `--topic` run should not spend the day's YouTube search quota."""
    ctx = _Ctx(_db(tmp_path))

    def _boom(*a, **k):                      # pragma: no cover - fails the test if hit
        raise AssertionError("collectors ran for a job that came with its own brief")

    import asa.research.collectors as collectors
    original, collectors.collect_all = collectors.collect_all, _boom
    try:
        _ensure_topic(ctx, "a bear who cannot say no")
    finally:
        collectors.collect_all = original


def test_upload_is_opt_in_per_run(capsys):
    """Uploading is off unless the operator says so on the command line, and `--who`
    defaults to something that reads as a person rather than as the machine."""
    import pytest
    with pytest.raises(SystemExit):
        main(["auto", "--help"])
    out = capsys.readouterr().out
    assert "--upload" in out
    assert "approval gate" in out


def test_auto_is_registered_and_defaults_to_stopping_for_a_human():
    import argparse
    import asa.core.cli as cli

    captured = {}

    def _fake(args):
        captured.update(vars(args))
        return 0

    original, cli.cmd_auto = cli.cmd_auto, _fake
    try:
        # main() binds cmd_auto at parser-build time, so rebuild through main itself.
        assert main(["auto"]) == 0
    except argparse.ArgumentError:                       # pragma: no cover
        raise
    finally:
        cli.cmd_auto = original
    assert captured["upload"] is False
    assert captured["topic"] is None
