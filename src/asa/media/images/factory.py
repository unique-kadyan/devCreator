"""The background pipeline: prompt -> cache -> provider chain -> ledger -> parallax plates.

Ordering matters. The cache is checked before any provider, because a reused location must
be pixel-identical to its first appearance or the show loses continuity. The ledger is
written before the plate is usable, because an asset with no licence row is an asset that
cannot legally ship, and finding that out at upload time is too late.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ...core.db import jdump, read, tx
from ...core.errors import AllProvidersExhausted, ProviderError, QuotaExhausted
from ...core.ledger import add_asset
from ...core.logging import get_logger
from ..animation.parallax import multiplane
from .base import GeneratedImage, ImageCache, LICENCE_BY_PROVIDER, prompt_key
from .huggingface import HuggingFaceImages
from .pollinations import PollinationsImages
from .procedural import ProceduralImages

log = get_logger("images")

# Two failure modes are worth naming, because both survived a milder prompt and both are
# visible in the finished frames:
#
#   1. FLUX puts PEOPLE in a street scene unless told, repeatedly and positively, that the
#      place is deserted. "no people" as a negative is not enough - the positive prompt has
#      to describe an empty place. This matters here beyond aesthetics: a human extra in an
#      all-animal world breaks the premise in a single frame.
#   2. FLUX puts LETTERING on shopfronts and it comes out as garbage ("BAKANY"). Again the
#      positive prompt has to ask for blank signs; a negative alone does not hold.
STYLE_SUFFIX = (
    "flat storybook illustration, hand-painted gouache texture, soft rim light, "
    "clean shapes, limited palette, wide establishing composition, "
    "completely deserted and empty, not a single person or creature anywhere, "
    "no figures at all, all signs and boards are blank with no writing on them, "
    "empty stage set waiting for actors")
NEGATIVE = ("text, words, letters, lettering, writing, signage text, shop sign text, "
            "labels, watermark, signature, logo, numbers, "
            "person, people, human, humans, man, woman, child, crowd, figures, "
            "silhouettes of people, pedestrians, bystanders, "
            "animal, animals, creature, character, face, "
            "photo, photorealistic, 3d render, ugly, deformed, blurry")


@dataclass
class Plate:
    location_id: str
    path: Path
    layers_dir: Path
    provider: str
    model_id: str
    cached: bool


class ImageChain:
    def __init__(self, providers: list, cache: ImageCache, db: Path,
                 size: tuple[int, int] = (1024, 576)):
        self.providers = [p for p in providers if p is not None]
        self.cache = cache
        self.db = Path(db)
        self.size = tuple(size)

    # ------------------------------------------------------------------ core

    def background(self, location_id: str, visual_prompt: str,
                   out_dir: Path, seed: int | None = None) -> Plate:
        prompt = f"{visual_prompt.strip().rstrip('.')}. {STYLE_SUFFIX}"
        key = prompt_key(prompt, self.size, NEGATIVE)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "plate.png"

        hit = self.cache.get(key)
        if hit is not None:
            meta = self.cache.read_sidecar(key)
            dest.write_bytes(hit.read_bytes())
            log.info("background_cache_hit", location=location_id,
                     provider=meta.get("provider", "?"))
            self._record(location_id, visual_prompt, dest, out_dir, meta)
            return Plate(location_id, dest, out_dir, meta.get("provider", "cache"),
                         meta.get("model_id", "?"), cached=True)

        gen = self._generate(prompt, dest, seed)
        self.cache.put(key, dest)
        meta = {"provider": gen.provider, "model_id": gen.model_id, "seed": gen.seed,
                "prompt": prompt, "negative": NEGATIVE, "size": list(self.size),
                **gen.meta}
        self.cache.sidecar(key, meta)

        add_asset(self.db, dest, kind="background", source=gen.provider,
                  license_code=LICENCE_BY_PROVIDER.get(gen.provider, "UNKNOWN"),
                  source_ref=gen.model_id,
                  meta={"location_id": location_id, "prompt_sha": key, "seed": gen.seed})
        self._record(location_id, visual_prompt, dest, out_dir, meta)
        return Plate(location_id, dest, out_dir, gen.provider, gen.model_id, cached=False)

    def scene(self, scene_id: int, prompt: str, out_dir: Path,
              negative: str, seed: int | None = None) -> Plate:
        """One finished frame per SCENE - characters, clothes and setting together.

        Separate from `background()` because the two want opposite things. A background
        plate is an empty stage: its prompt forbids figures so puppets can be composited
        on top. A cinematic scene IS the figures, so it carries its own negative prompt
        (which excludes the failure modes of photoreal generation instead) and is keyed per
        scene rather than per location - two scenes in one classroom are different images.
        """
        key = prompt_key(prompt, self.size, negative)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"scene_{scene_id:03d}.png"

        hit = self.cache.get(key)
        if hit is not None:
            meta = self.cache.read_sidecar(key)
            dest.write_bytes(hit.read_bytes())
            log.info("scene_image_cache_hit", scene=scene_id,
                     provider=meta.get("provider", "?"))
            return Plate(str(scene_id), dest, out_dir, meta.get("provider", "cache"),
                         meta.get("model_id", "?"), cached=True)

        gen = self._generate_with(prompt, dest, seed, negative)
        self.cache.put(key, dest)
        self.cache.sidecar(key, {"provider": gen.provider, "model_id": gen.model_id,
                                 "seed": gen.seed, "prompt": prompt,
                                 "negative": negative, "size": list(self.size),
                                 **gen.meta})
        add_asset(self.db, dest, kind="scene_image", source=gen.provider,
                  license_code=LICENCE_BY_PROVIDER.get(gen.provider, "UNKNOWN"),
                  source_ref=gen.model_id,
                  meta={"scene_id": scene_id, "prompt_sha": key, "seed": gen.seed})
        return Plate(str(scene_id), dest, out_dir, gen.provider, gen.model_id,
                     cached=False)

    def _generate_with(self, prompt: str, dest: Path, seed: int | None,
                       negative: str) -> GeneratedImage:
        """Like `_generate` but with a caller-supplied negative prompt."""
        failures: list[str] = []
        for p in self.providers:
            if not getattr(p, "available", False):
                failures.append(f"{p.name}: unavailable")
                continue
            try:
                return p.generate(prompt, dest, self.size, negative, seed)
            except Exception as e:                                    # noqa: BLE001
                failures.append(f"{p.name}: {str(e)[:80]}")
                log.warning("scene_image_provider_failed", provider=p.name,
                            error=str(e)[:140])
        raise ProviderError("every image provider failed: " + " | ".join(failures),
                            provider="images")

    def _generate(self, prompt: str, dest: Path, seed: int | None) -> GeneratedImage:
        failures: list[str] = []
        for p in self.providers:
            if not getattr(p, "available", False):
                failures.append(f"{p.name}: unavailable")
                continue
            try:
                return p.generate(prompt, dest, self.size, NEGATIVE, seed)
            except QuotaExhausted as e:
                failures.append(f"{p.name}: quota - {e}")
                log.warning("image_provider_quota", provider=p.name)
            except ProviderError as e:
                failures.append(f"{p.name}: {e}")
                log.warning("image_provider_failed", provider=p.name, error=str(e)[:160])
        raise AllProvidersExhausted("every image provider failed: " + " | ".join(failures))

    def _record(self, location_id: str, visual_prompt: str, plate: Path, out_dir: Path,
                meta: dict) -> None:
        rel = str(plate.parent)
        with tx(self.db) as con:
            con.execute("""
                INSERT INTO locations (id, name, description, visual_prompt, plate_dir,
                                       layers, uses)
                VALUES (?,?,?,?,?,?,1)
                ON CONFLICT(id) DO UPDATE SET
                    uses = uses + 1, plate_dir = excluded.plate_dir,
                    layers = excluded.layers
            """, (location_id, location_id.replace("_", " ").title(), visual_prompt[:400],
                  visual_prompt, rel, jdump(["far.png", "mid.png", "near.png"])))

    # ------------------------------------------------------------------ plates

    def plates(self, plate: Path, world: tuple[int, int], out_dir: Path) -> list[Path]:
        """Bake the multiplane split to disk so parallel render workers can mmap it
        instead of each recomputing three LANCZOS resizes of a 2.7k plate."""
        out_dir.mkdir(parents=True, exist_ok=True)
        names = ["far.png", "mid.png", "near.png"]
        paths = [out_dir / n for n in names]
        if all(p.exists() for p in paths):
            return paths
        with Image.open(plate) as im:
            layers = multiplane(im.copy(), world)
        for bl, path in zip(layers, paths):
            bl.image.save(path)
        return paths


def build_image_chain(cfg, db: Path) -> ImageChain:
    """Assemble from config, skipping providers with no credentials.

    `procedural` is always appended last whether or not it is listed, because a chain that
    can run out is a chain that fails a render at 3am for no recoverable reason.
    """
    order = cfg.get("providers.image.chain", ["huggingface", "pollinations", "procedural"])
    size = tuple(cfg.get("providers.image.huggingface.size", [1024, 576]))
    built, skipped = [], []
    for name in order:
        if name == "huggingface":
            token = cfg.secret("HF_TOKEN", required=False)
            p = HuggingFaceImages(
                token=token,
                models=cfg.get("providers.image.huggingface.models"),
                provider=cfg.get("providers.image.huggingface.provider", "auto"),
                negative_hints=cfg.get("providers.image.huggingface.negative_hints", ""))
        elif name == "pollinations":
            p = PollinationsImages(
                enabled=bool(cfg.get("providers.image.pollinations.enabled", False)))
        elif name == "procedural":
            p = ProceduralImages()
        else:
            skipped.append(f"{name}(unknown)")
            continue
        if p.available:
            built.append(p)
        else:
            skipped.append(f"{name}(unavailable)")
    if not any(getattr(p, "name", "") == "procedural" for p in built):
        built.append(ProceduralImages())
    log.info("image_chain_built", active=[p.name for p in built], skipped=skipped)
    cache = ImageCache(cfg.path("paths.cache", "data/cache") / "images")
    return ImageChain(built, cache, db, size)
