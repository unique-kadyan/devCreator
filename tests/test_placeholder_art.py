"""An episode built on placeholder art must not reach a viewer.

A burst of image-provider rate limits put the flat vector fallback into five of ten shots
of a finished ad, and sixteen mechanical QC checks passed it: the container was valid, the
audio was in sync, the loudness was right, the licences were clean. Nothing looked at what
was in the frame. These tests pin down both halves of the fix - QC fails the episode, and
the cache never serves a placeholder again once a real provider is back.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.media.images.base import ImageCache                            # noqa: E402
from asa.media.images.factory import FALLBACK_PROVIDER, ImageChain      # noqa: E402
from asa.qc.checks import _placeholder_art                              # noqa: E402


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    sqlite3.connect(p).executescript((ROOT / "migrations/001_initial.sql").read_text())
    return p


def add_asset_row(db, path, source):
    """Mirrors ledger.add_asset: `assets.path` is UNIQUE and the real insert upserts, so a
    regenerated file replaces its own row rather than adding a second one."""
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO assets (kind, path, sha256, source, usage_allowed) "
                    "VALUES ('scene_image', ?, 'x', ?, 1) "
                    "ON CONFLICT(path) DO UPDATE SET source = excluded.source",
                    (str(path), source))


def test_placeholder_art_fails_the_episode(db, tmp_path):
    img = tmp_path / "scene_001_abc.png"
    img.write_bytes(b"x")
    add_asset_row(db, img, "procedural")
    scenes = [{"plate_path": str(img)}]
    assert _placeholder_art(db, scenes) == [str(img)]


def test_real_art_passes(db, tmp_path):
    img = tmp_path / "scene_001_abc.png"
    img.write_bytes(b"x")
    add_asset_row(db, img, "huggingface")
    assert _placeholder_art(db, [{"plate_path": str(img)}]) == []


def test_regenerating_a_placeholder_clears_it(db, tmp_path):
    """Paths are reused when the art stage re-runs, so only the CURRENT provider counts -
    otherwise a repaired episode would fail forever on its own history."""
    img = tmp_path / "scene_001_abc.png"
    img.write_bytes(b"x")
    add_asset_row(db, img, "procedural")
    add_asset_row(db, img, "huggingface")
    assert _placeholder_art(db, [{"plate_path": str(img)}]) == []


def test_puppet_mode_has_nothing_to_check(db):
    """Puppet episodes composite characters over background plates and never generate a
    scene image, so this check must stay silent rather than guess."""
    assert _placeholder_art(db, [{"plate_path": None}]) == []


# ------------------------------------------------------------------- cache

class FakeProvider:
    def __init__(self, name, available=True):
        self.name = name
        self.available = available


def chain(tmp_path, providers):
    return ImageChain(providers, ImageCache(tmp_path / "cache"), tmp_path / "t.db")


def test_a_cached_placeholder_is_ignored_while_a_real_provider_is_up(tmp_path):
    """The poisoning bug: a placeholder cached under the prompt key made the failure
    permanent, because every later re-run scored a cache hit on it."""
    c = chain(tmp_path, [FakeProvider("huggingface"), FakeProvider(FALLBACK_PROVIDER)])
    c.cache.sidecar("k", {"provider": FALLBACK_PROVIDER})
    assert c._usable("k") is False


def test_a_cached_placeholder_is_reused_when_nothing_better_exists(tmp_path):
    """With every real provider down, the placeholder is still the best there is - and
    regenerating an identical one costs time for no gain."""
    c = chain(tmp_path, [FakeProvider("huggingface", available=False),
                         FakeProvider(FALLBACK_PROVIDER)])
    c.cache.sidecar("k", {"provider": FALLBACK_PROVIDER})
    assert c._usable("k") is True


def test_real_art_is_always_reused(tmp_path):
    c = chain(tmp_path, [FakeProvider("huggingface"), FakeProvider(FALLBACK_PROVIDER)])
    c.cache.sidecar("k", {"provider": "huggingface"})
    assert c._usable("k") is True
