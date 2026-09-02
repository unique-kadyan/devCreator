"""Image provider protocol plus the content-addressed cache every provider goes through.

The cache is the reason this pipeline can afford image generation at all. A background is
keyed by (prompt, size, negative, style) - not by scene id - so a location reused in
episode 12 costs zero credits, a re-run after a crash costs zero credits, and an edited
prompt correctly misses.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass
class GeneratedImage:
    path: Path
    provider: str
    model_id: str
    prompt_sha: str
    cached: bool = False
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def license_code(self) -> str:
        return LICENCE_BY_PROVIDER.get(self.provider, "UNKNOWN")


# Fail-closed: a provider absent from this map yields UNKNOWN, and the ledger refuses to
# publish UNKNOWN. Adding a provider therefore forces a licence decision.
LICENCE_BY_PROVIDER = {
    "procedural": "CC0",          # drawn by this repo's own code
    "huggingface": "APACHE-2.0",  # only Apache-2.0 / permissive models are configured
    "pollinations": "UNKNOWN",    # terms do not clearly grant commercial reuse
}


class ImageProvider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def generate(self, prompt: str, out_path: Path, size: tuple[int, int],
                 negative: str = "", seed: int | None = None) -> GeneratedImage: ...


def prompt_key(prompt: str, size: tuple[int, int], negative: str, style: str = "") -> str:
    blob = json.dumps({"p": prompt.strip(), "s": list(size), "n": negative.strip(),
                       "y": style.strip()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class ImageCache:
    """Content-addressed store. Entries never expire: an image that matched a prompt last
    month still matches it today, and re-rolling art for a reused location would break the
    visual continuity the whole design depends on."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        # Two-level fan-out: a channel running for years accumulates thousands of plates and
        # a single flat directory makes every stat() slow.
        return self.root / key[:2] / f"{key}.png"

    def get(self, key: str) -> Path | None:
        p = self.path_for(key)
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:                                      # noqa: BLE001
            # A truncated file from an interrupted download is worse than a miss, because
            # it fails deep inside the compositor instead of here.
            p.unlink(missing_ok=True)
            return None
        return p

    def put(self, key: str, src: Path) -> Path:
        dst = self.path_for(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        return dst

    def sidecar(self, key: str, data: dict) -> None:
        p = self.path_for(key).with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, sort_keys=True))

    def read_sidecar(self, key: str) -> dict:
        p = self.path_for(key).with_suffix(".json")
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
