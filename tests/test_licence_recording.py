"""A provider's declared licence must reach the LEDGER, not just the GeneratedImage.

The bug this pins down: `GeneratedImage.license_code` exists so a provider can declare the
licence its output really carries, and its docstring says that declaration "wins over the
table". Both ledger writes in `factory` ignored it and recorded
`LICENCE_BY_PROVIDER[provider]` instead - which for `replicate` is the fail-closed floor
UNKNOWN. So 108 flux-schnell stills, Apache-2.0 every one, were filed as UNKNOWN, and
`audit()` blocks UNKNOWN because its `usage_allowed` is 'unknown'.

It failed in the worst direction available: not an unlicensed asset published, but a
correctly licensed episode refused at QC, three stages and two paid hours after the images
were bought. `test_the_declared_licence_reaches_the_generated_image` already covered the
provider half and passed throughout - the gap was strictly between that property and the
INSERT, so that is what these two assert.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.media.images.base import GeneratedImage, ImageCache            # noqa: E402
from asa.media.images.factory import ImageChain                         # noqa: E402


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    sqlite3.connect(p).executescript((ROOT / "migrations/001_initial.sql").read_text())
    return p


class StubProvider:
    """A metered provider: one service, several models, a licence that differs per model.

    `declares` None is the provider that says nothing and must fall back to the table.
    """

    available = True

    def __init__(self, name="replicate", declares="APACHE-2.0"):
        self.name = name
        self.declares = declares

    def generate(self, prompt, dest, seed=None, negative=None, size=None):
        Image.new("RGB", (16, 16), (9, 9, 9)).save(dest)
        return GeneratedImage(
            path=dest, provider=self.name, model_id="black-forest-labs/flux-schnell",
            prompt_sha="k", seed=seed,
            meta={"licence": self.declares} if self.declares else {})


def licence_of(db, path) -> str | None:
    """`add_asset` stores the path RELATIVE to the project root, so match on the name."""
    with sqlite3.connect(db) as con:
        row = con.execute(
            "SELECT l.license_code FROM assets a LEFT JOIN licenses l ON l.id = a.license_id "
            "WHERE a.path LIKE ?", (f"%{Path(path).name}",)).fetchone()
    return row[0] if row else None


def build(db, tmp_path, provider):
    return ImageChain([provider], ImageCache(tmp_path / "cache"), db, size=(16, 16))


def test_a_declared_licence_is_what_the_ledger_records(db, tmp_path):
    """The regression. Recorded UNKNOWN before the fix, which `audit()` refuses to ship."""
    chain = build(db, tmp_path, StubProvider(declares="APACHE-2.0"))
    plate = chain.scene(1, "a wet platform", tmp_path / "out", negative="")
    assert licence_of(db, plate.path) == "APACHE-2.0"


def test_a_provider_that_declares_nothing_still_falls_to_the_fail_closed_floor(db, tmp_path):
    """The fix must not become a way to publish something nobody licensed: a provider with
    no declaration is UNKNOWN, and UNKNOWN is exactly what the audit stops."""
    chain = build(db, tmp_path, StubProvider(name="mystery", declares=None))
    plate = chain.scene(2, "a wet platform", tmp_path / "out", negative="")
    assert licence_of(db, plate.path) == "UNKNOWN"
