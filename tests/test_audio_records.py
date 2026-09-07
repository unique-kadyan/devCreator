"""One synthesised clip per line, however many times the audio stage runs.

The bug: `_record_audio` INSERTed unconditionally and nothing in the schema stopped a
second row for the same line. The audio stage is re-runnable by design, and every later
stage rebuilds its timeline from this table - so one retry doubled every scene's duration.
The shot list stretched to fill it, the subtitles drifted, and the assembled video ran 197
seconds against a 119-second soundtrack before QC caught the drift.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.core.db import migrate, read                                   # noqa: E402
from asa.media.audio.build import _record_audio                         # noqa: E402
from asa.media.tts.base import Utterance                                # noqa: E402


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    migrate(p, ROOT / "migrations")
    with sqlite3.connect(p) as con:
        con.execute("INSERT INTO stories (id, title, hook, logline, target_audience, "
                    "genre, archetype, moral, setting, beat_beginning, beat_conflict, "
                    "beat_rising, beat_climax, beat_resolution, ending, beat_signature) "
                    "VALUES (1,'t','h','l','a','g','comedy','m','s','a','b','c','d','e','f','v|v|v|v|v')")
        con.execute("INSERT INTO scenes (id, story_id, idx, action) VALUES (1,1,1,'a')")
        con.execute("INSERT INTO dialogue (id, scene_id, idx, character_id, line) "
                    "VALUES (1,1,0,NULL,'line one')")
    return p


def utterance(path: str, seconds: float) -> Utterance:
    return Utterance(text="line one", path=Path(path), duration_s=seconds,
                     sample_rate=24000, character_id=None, provider="kokoro", voice_id="hf_beta",
                     text_sha256="abc")


def rows(db):
    with read(db) as con:
        return [dict(r) for r in con.execute(
            "SELECT path, duration_s FROM audio WHERE dialogue_id = 1")]


def test_one_row_per_line_however_often_the_stage_reruns(db):
    _record_audio(db, 1, 1, utterance("/a/one.wav", 3.0), None)
    _record_audio(db, 1, 1, utterance("/a/one.wav", 3.0), None)
    _record_audio(db, 1, 1, utterance("/a/one.wav", 3.0), None)
    assert len(rows(db)) == 1


def test_a_rerun_replaces_the_clip_it_recorded(db):
    """An edited line re-synthesises to a new file and a new duration; the row must follow
    the audio that now exists on disk, not the one that used to."""
    _record_audio(db, 1, 1, utterance("/a/old.wav", 3.0), None)
    _record_audio(db, 1, 1, utterance("/a/new.wav", 5.5), None)
    assert rows(db) == [{"path": "/a/new.wav", "duration_s": 5.5}]


def test_the_schema_refuses_a_duplicate_even_by_raw_insert(db):
    """The guarantee is in the index, not only in the one code path that writes here."""
    _record_audio(db, 1, 1, utterance("/a/one.wav", 3.0), None)
    with pytest.raises(sqlite3.IntegrityError):
        with sqlite3.connect(db) as con:
            con.execute("INSERT INTO audio (scene_id, dialogue_id, kind, path, duration_s) "
                        "VALUES (1, 1, 'narration', '/a/dupe.wav', 3.0)")


def test_sfx_and_music_may_still_repeat(db):
    """The index is partial on purpose: those rows carry no dialogue_id and a scene may
    have any number of them."""
    with sqlite3.connect(db) as con:
        for i in range(3):
            con.execute("INSERT INTO audio (scene_id, dialogue_id, kind, path, duration_s) "
                        "VALUES (1, NULL, 'sfx', ?, 1.0)", (f"/a/sfx{i}.wav",))
    with read(db) as con:
        assert con.execute("SELECT COUNT(*) FROM audio WHERE kind='sfx'").fetchone()[0] == 3
