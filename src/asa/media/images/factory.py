"""The background pipeline: prompt -> cache -> provider chain -> ledger -> parallax plates.

Ordering matters. The cache is checked before any provider, because a reused location must
be pixel-identical to its first appearance or the show loses continuity. The ledger is
written before the plate is usable, because an asset with no licence row is an asset that
cannot legally ship, and finding that out at upload time is too late.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ...core.db import jdump, read, tx
from ...core.errors import (AllProvidersExhausted, ProviderError, QuotaExhausted,
                            RateLimited)
from ...core.ledger import add_asset
from ...core.logging import get_logger
from ..animation.parallax import multiplane
from .base import GeneratedImage, ImageCache, prompt_key
from .huggingface import HuggingFaceImages
from .local import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL
from .local import LocalDiffusionImages
from .pollinations import PollinationsImages
from .replicate_images import DEFAULT_MODEL as DEFAULT_REPLICATE_MODEL
from .replicate_images import ReplicateImages
from .procedural import ProceduralImages

log = get_logger("images")

# How many times one provider may be waited out on a 429 before the chain moves on, and the
# longest single wait worth taking.
#
# This exists because `_latch_quota` says a rate limit "clears in seconds and the provider is
# still the best one available" - and then nothing acted on that. A RateLimited fell through
# the generic handler to the NEXT provider, which on this machine is CPU diffusion at 10-17
# MINUTES an image. Measured on job 11: Replicate served one still, 429'd on the second, and
# the art stage settled into local rendering for an estimated fifteen hours - having been
# eight seconds away from the picture it wanted.
#
# Bounded on purpose. A provider that answers 429 forever is telling you something real, and
# the chain still has to reach the free floor rather than sleep out the stage's budget.
RATE_LIMIT_TRIES = 4
RATE_LIMIT_MAX_WAIT_S = 30.0


def _wait_out(provider: str, err: RateLimited, attempt: int) -> bool:
    """Sleep off one 429. False when the provider has had its chances."""
    if attempt >= RATE_LIMIT_TRIES:
        return False
    wait = min(max(float(getattr(err, "retry_after_s", 5.0) or 5.0), 1.0),
               RATE_LIMIT_MAX_WAIT_S)
    log.info("image_provider_rate_limited", provider=provider, attempt=attempt,
             waiting_s=round(wait, 1),
             detail="waiting rather than falling to the next provider - the one below is "
                    "slower or worse, and a 429 clears on its own")
    time.sleep(wait)
    return True


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


# The last-resort provider. Its output is a flat vector placeholder with no characters in
# it, which exists so a render at 3am degrades instead of dying - it is NOT a picture of the
# scene, and it must never be cached as though it were.
#
# It was. A RoleVo ad lost its second half to a burst of HuggingFace rate limits, and the
# five placeholder frames went into the prompt-keyed cache alongside the real ones. Every
# re-run then scored a cache hit and reused them: the episode could not be repaired by
# running the art stage again, and the failure was permanent for the life of the cache.
FALLBACK_PROVIDER = "procedural"


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
        # Providers that have told us their allowance is gone. See `_latch_quota`.
        self._spent: dict[str, str] = {}

    # ------------------------------------------------------------------ credit

    def _latch_quota(self, provider: str, reason: str) -> None:
        """Remember, for the rest of this run, that a provider's allowance is spent.

        MEASURED, on job 10: HuggingFace answers 402 after its free stills are gone, and
        without this every remaining image asked it again anyway. Each ask walks HF's whole
        configured model list before giving up, which costs about 45 seconds - so roughly
        twenty images ate the art stage's entire 900s budget and the stage failed with most
        of the episode ungenerated, while a working provider sat next in the chain the whole
        time.

        Latched on QuotaExhausted ONLY, never on RateLimited. The two look alike and are
        opposite: a rate limit clears in seconds and the provider is still the best one
        available, whereas an exhausted allowance does not come back inside a run. Latching
        a rate limit would give away a good provider for the rest of an episode.
        """
        if provider in self._spent:
            return
        self._spent[provider] = reason
        log.warning("image_provider_out_of_credit", provider=provider,
                    reason=reason[:160],
                    detail="skipping it for the rest of this run rather than "
                           "re-learning this on every image")

    def _live(self) -> list:
        """Providers worth asking right now: available, and not known to be broke."""
        return [p for p in self.providers
                if getattr(p, "available", False) and p.name not in self._spent]

    # ------------------------------------------------------------------ core

    def background(self, location_id: str, visual_prompt: str,
                   out_dir: Path, seed: int | None = None) -> Plate:
        prompt = f"{visual_prompt.strip().rstrip('.')}. {STYLE_SUFFIX}"
        key = prompt_key(prompt, self.size, NEGATIVE)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "plate.png"

        hit = self.cache.get(key)
        if hit is not None and self._usable(key):
            meta = self.cache.read_sidecar(key)
            dest.write_bytes(hit.read_bytes())
            log.info("background_cache_hit", location=location_id,
                     provider=meta.get("provider", "?"))
            self._record(location_id, visual_prompt, dest, out_dir, meta)
            return Plate(location_id, dest, out_dir, meta.get("provider", "cache"),
                         meta.get("model_id", "?"), cached=True)

        gen = self._generate(prompt, dest, seed)
        if gen.provider != FALLBACK_PROVIDER:
            self.cache.put(key, dest)
        meta = {"provider": gen.provider, "model_id": gen.model_id, "seed": gen.seed,
                "prompt": prompt, "negative": NEGATIVE, "size": list(self.size),
                **gen.meta}
        self.cache.sidecar(key, meta)

        add_asset(self.db, dest, kind="background", source=gen.provider,
                  license_code=gen.license_code,
                  source_ref=gen.model_id,
                  meta={"location_id": location_id, "prompt_sha": key, "seed": gen.seed})
        self._record(location_id, visual_prompt, dest, out_dir, meta)
        return Plate(location_id, dest, out_dir, gen.provider, gen.model_id, cached=False)

    def scene(self, scene_id: int, prompt: str, out_dir: Path, negative: str,
              seed: int | None = None, name: str | None = None) -> Plate:
        """One finished frame per SCENE - characters, clothes and setting together.

        Separate from `background()` because the two want opposite things. A background
        plate is an empty stage: its prompt forbids figures so puppets can be composited
        on top. A cinematic scene IS the figures, so it carries its own negative prompt
        (which excludes the failure modes of photoreal generation instead) and is keyed per
        scene rather than per location - two scenes in one classroom are different images.
        """
        key = prompt_key(prompt, self.size, negative)
        out_dir.mkdir(parents=True, exist_ok=True)
        # `name` is how the shot renderer addresses several setups within one scene. The
        # CACHE key is still the prompt, so two shots that ask for the same picture cost
        # one generation however many files they are copied to.
        dest = out_dir / (name or f"scene_{scene_id:03d}.png")

        hit = self.cache.get(key)
        if hit is not None and self._usable(key):
            meta = self.cache.read_sidecar(key)
            dest.write_bytes(hit.read_bytes())
            log.info("scene_image_cache_hit", scene=scene_id,
                     provider=meta.get("provider", "?"))
            return Plate(str(scene_id), dest, out_dir, meta.get("provider", "cache"),
                         meta.get("model_id", "?"), cached=True)

        gen = self._generate_with(prompt, dest, seed, negative)
        if gen.provider != FALLBACK_PROVIDER:
            self.cache.put(key, dest)
        self.cache.sidecar(key, {"provider": gen.provider, "model_id": gen.model_id,
                                 "seed": gen.seed, "prompt": prompt,
                                 "negative": negative, "size": list(self.size),
                                 **gen.meta})
        add_asset(self.db, dest, kind="scene_image", source=gen.provider,
                  license_code=gen.license_code,
                  source_ref=gen.model_id,
                  meta={"scene_id": scene_id, "prompt_sha": key, "seed": gen.seed})
        return Plate(str(scene_id), dest, out_dir, gen.provider, gen.model_id,
                     cached=False)

    def _usable(self, key: str) -> bool:
        """Whether a cache hit is a real picture, or a placeholder worth retrying past.

        A cached placeholder is only worth reusing when there is still nothing better
        available - so this asks whether any real provider is up right now, rather than
        assuming the outage that produced it is permanent. Rate limits are usually over in
        minutes; a poisoned cache entry is forever.
        """
        meta = self.cache.read_sidecar(key)
        if meta.get("provider") != FALLBACK_PROVIDER:
            return True
        # `_live()` rather than `available`, so a provider already known to be out of credit
        # does not count as "something better is up". Otherwise a cached placeholder is
        # rejected, the chain is walked, the spent provider fails again, and the same
        # placeholder is regenerated - the retry this check exists to justify never happens.
        real = [p for p in self._live() if p.name != FALLBACK_PROVIDER]
        if real:
            log.info("placeholder_cache_ignored", retry_with=[p.name for p in real])
            return False
        return True

    def _generate_with(self, prompt: str, dest: Path, seed: int | None,
                       negative: str) -> GeneratedImage:
        """Like `_generate` but with a caller-supplied negative prompt."""
        failures: list[str] = []
        for p in self.providers:
            if p.name in self._spent:
                failures.append(f"{p.name}: allowance spent earlier in this run")
                continue
            if not getattr(p, "available", False):
                failures.append(f"{p.name}: unavailable")
                continue
            attempt = 0
            while True:
                try:
                    return p.generate(prompt, dest, self.size, negative, seed)
                except QuotaExhausted as e:
                    self._latch_quota(p.name, str(e))
                    failures.append(f"{p.name}: quota - {str(e)[:80]}")
                    break
                except RateLimited as e:
                    attempt += 1
                    if _wait_out(p.name, e, attempt):
                        continue
                    failures.append(f"{p.name}: rate limited after {attempt} waits")
                    log.warning("scene_image_provider_failed", provider=p.name,
                                error=str(e)[:140])
                    break
                except Exception as e:                                # noqa: BLE001
                    failures.append(f"{p.name}: {str(e)[:80]}")
                    log.warning("scene_image_provider_failed", provider=p.name,
                                error=str(e)[:140])
                    break
        raise ProviderError("every image provider failed: " + " | ".join(failures),
                            provider="images")

    def _generate(self, prompt: str, dest: Path, seed: int | None) -> GeneratedImage:
        failures: list[str] = []
        for p in self.providers:
            if p.name in self._spent:
                failures.append(f"{p.name}: allowance spent earlier in this run")
                continue
            if not getattr(p, "available", False):
                failures.append(f"{p.name}: unavailable")
                continue
            attempt = 0
            while True:
                try:
                    return p.generate(prompt, dest, self.size, NEGATIVE, seed)
                except QuotaExhausted as e:
                    self._latch_quota(p.name, str(e))
                    failures.append(f"{p.name}: quota - {e}")
                    break
                except RateLimited as e:
                    attempt += 1
                    if _wait_out(p.name, e, attempt):
                        continue
                    failures.append(f"{p.name}: rate limited after {attempt} waits")
                    log.warning("image_provider_failed", provider=p.name,
                                error=str(e)[:160])
                    break
                except ProviderError as e:
                    failures.append(f"{p.name}: {e}")
                    log.warning("image_provider_failed", provider=p.name,
                                error=str(e)[:160])
                    break
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
        elif name == "local":
            base = "providers.image.local"
            p = LocalDiffusionImages(
                enabled=bool(cfg.get(f"{base}.enabled", False)),
                model=cfg.get(f"{base}.model", DEFAULT_LOCAL_MODEL),
                steps=int(cfg.get(f"{base}.steps", 20)),
                guidance=float(cfg.get(f"{base}.guidance", 7.0)),
                threads=cfg.get(f"{base}.threads"),
                cache_dir=cfg.get(f"{base}.cache_dir"))
        elif name == "replicate":
            base = "providers.image.replicate"
            p = ReplicateImages(
                token=cfg.secret("REPLICATE_API_TOKEN", required=False),
                model=cfg.get(f"{base}.model", DEFAULT_REPLICATE_MODEL),
                steps=int(cfg.get(f"{base}.steps", 4)),
                megapixels=str(cfg.get(f"{base}.megapixels", "1")),
                enabled=bool(cfg.get(f"{base}.enabled", False)),
                max_images=int(cfg.get(f"{base}.max_images", 400)))
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
