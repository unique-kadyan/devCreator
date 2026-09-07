"""Replicate image generation - the metered path that keeps the art stage moving.

WHY THIS EXISTS. The image chain was [huggingface, local, pollinations, procedural] and
every rung of it had run out of road. HuggingFace bills a monthly credit window that this
account exhausts after a couple of stills (measured 2026-09-04: one probe image succeeded,
the second call in the same minute fell through). `local` is CPU diffusion on a machine
with no GPU - measured at 600s per 1024x576 image, which is 30 hours for an episode, and
worse at the portrait sizes this pipeline now asks for. `pollinations` is keyless but its
terms grant nothing, so everything it makes is published under an accepted-risk code.

Replicate is already a funded account here for video. FLUX.1-schnell on it is the same
model HuggingFace was routing to, at four steps and a few seconds per image, and Apache-2.0
so the ledger has a real licence to record rather than a risk to accept.

LICENCE. Same fail-closed rule as media/images/local.py: only models on MODELS may be run,
each names the licence its OUTPUT carries, and the provider declares that per image through
`GeneratedImage.meta["licence"]` rather than leaning on the by-provider table. A slug that
is not listed is refused rather than published under a code nobody checked.

NEGATIVE PROMPTS. FLUX has no negative conditioning - it is a guidance-distilled model and
the schema has no such field. The negative is therefore NOT sent, and callers must not
assume it was honoured. It still takes part in the cache key, because an image generated
under a different negative is a different request even when this provider ignored it.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import httpx
from PIL import Image

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.logging import get_logger
from .base import GeneratedImage, prompt_key

log = get_logger("img_replicate")

API = "https://api.replicate.com/v1"

# slug -> the licence the model's OUTPUT carries. Fail-closed: anything absent is refused.
MODELS: dict[str, str] = {
    "black-forest-labs/flux-schnell": "APACHE-2.0",
    "black-forest-labs/flux-dev": "FLUX-DEV-NONCOMMERCIAL",   # listed so it is REFUSED
    "bytedance/seedream-3": "UNKNOWN",                        # listed so it is REFUSED
}

# Only these may actually be published. The others are in MODELS so that pointing config at
# one produces a clear refusal naming the licence, rather than a "model not allowed" that
# reads like a typo.
PUBLISHABLE = {"APACHE-2.0"}

DEFAULT_MODEL = "black-forest-labs/flux-schnell"

# The aspect ratios flux-schnell offers, as width/height. The pipeline asks for a pixel
# size; this is how that becomes the enum the model accepts. Read off the live schema
# 2026-09-04.
_RATIOS: dict[str, float] = {
    "1:1": 1.0, "16:9": 16 / 9, "21:9": 21 / 9, "3:2": 1.5, "2:3": 2 / 3,
    "4:5": 0.8, "5:4": 1.25, "3:4": 0.75, "4:3": 4 / 3, "9:16": 9 / 16, "9:21": 9 / 21,
}


def nearest_aspect(size: tuple[int, int]) -> str:
    """The offered ratio closest to the size asked for.

    Closest in LOG ratio, not in linear difference: 16:9 and 21:9 are 0.36 apart linearly
    while 9:16 and 9:21 are 0.13, so a linear metric quietly prefers the portrait options
    for every landscape request. The comparison that matters is proportional.
    """
    import math
    w, h = size
    want = math.log((w / h) if h else 1.0)
    return min(_RATIOS, key=lambda k: abs(math.log(_RATIOS[k]) - want))


class ReplicateImages:
    name = "replicate"

    def __init__(self, token: str | None, *, model: str = DEFAULT_MODEL,
                 steps: int = 4, megapixels: str = "1", output_format: str = "png",
                 enabled: bool = True, timeout_s: float = 180.0, poll_s: float = 1.5,
                 max_images: int = 400):
        self.token = (token or "").strip()
        self.model = model
        self.steps = int(steps)
        self.megapixels = str(megapixels)
        self.output_format = output_format
        self.enabled = bool(enabled)
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        # A hard stop, for the same reason the video provider has one: a retry loop against
        # a metered API is a bill, not a crash.
        self.max_images = int(max_images)
        self.generated = 0
        self._no_credit = False

        if self.enabled and self.model not in MODELS:
            raise ProviderError(
                f"replicate image model '{self.model}' is not on the allowlist; add it to "
                f"media/images/replicate_images.MODELS with the licence its output carries",
                provider=self.name, retryable=False)
        if self.enabled and MODELS.get(self.model) not in PUBLISHABLE:
            raise ProviderError(
                f"replicate image model '{self.model}' carries licence "
                f"'{MODELS.get(self.model)}', which this pipeline will not publish; "
                f"pick one of {sorted(m for m, l in MODELS.items() if l in PUBLISHABLE)}",
                provider=self.name, retryable=False)

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.token and not self._no_credit
                    and self.generated < self.max_images)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"}

    def _raise_for(self, r: httpx.Response) -> None:
        if r.status_code < 400:
            return
        body = r.text[:200]
        if r.status_code == 402:
            # Latched: every later shot in this process would pay the same discovery cost
            # to be told the same thing. Same reasoning as the video provider's latch.
            self._no_credit = True
            raise QuotaExhausted("Replicate billing exhausted", provider=self.name)
        if r.status_code in (401, 403):
            raise AuthError(f"Replicate token rejected ({r.status_code}) for images")
        if r.status_code == 429:
            try:
                after = float(r.headers.get("retry-after") or 0.0)
            except ValueError:
                after = 0.0
            raise RateLimited("Replicate rate limited", provider=self.name,
                              retry_after_s=max(after, 10.0))
        raise ProviderError(f"replicate images HTTP {r.status_code}: {body}",
                            provider=self.name, retryable=r.status_code >= 500)

    def generate(self, prompt: str, out_path: Path, size: tuple[int, int],
                 negative: str = "", seed: int | None = None) -> GeneratedImage:
        if not self.enabled:
            raise ProviderError("replicate images disabled", provider=self.name,
                                retryable=False)
        if not self.token:
            raise ProviderError("replicate images unavailable (no REPLICATE_API_TOKEN)",
                                provider=self.name, retryable=False)
        if self._no_credit:
            raise QuotaExhausted("Replicate billing exhausted", provider=self.name)
        if self.generated >= self.max_images:
            raise QuotaExhausted(
                f"replicate image budget reached ({self.max_images} this process)",
                provider=self.name)

        # `negative` is in the key but not in the payload - see the module docstring.
        key = prompt_key(prompt, size, negative)
        payload = {"input": {
            "prompt": prompt,
            "aspect_ratio": nearest_aspect(size),
            "megapixels": self.megapixels,
            "num_inference_steps": self.steps,
            "output_format": self.output_format,
            "num_outputs": 1,
        }}
        if seed is not None:
            payload["input"]["seed"] = int(seed)

        t0 = time.time()
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{API}/models/{self.model}/predictions",
                            headers=self._headers(), json=payload)
            self._raise_for(r)
            pred = r.json()
            url = self._await(client, pred)
            img_bytes = self._download(client, url)

        img = Image.open(io.BytesIO(img_bytes))
        if img.size != tuple(size):
            # The model returns the enum's own pixel size, which is the right SHAPE but
            # rarely the exact pixels asked for. Resizing here keeps the frame-size contract
            # the compositor depends on.
            img = img.resize(size, Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out_path)
        self.generated += 1
        seconds = round(time.time() - t0, 2)
        log.info("replicate_image_ok", model=self.model, seconds=seconds,
                 size=list(size), aspect=payload["input"]["aspect_ratio"],
                 spent=self.generated)
        return GeneratedImage(
            path=out_path, provider=self.name, model_id=self.model, prompt_sha=key,
            seed=seed,
            meta={"seconds": seconds, "steps": self.steps,
                  "licence": MODELS[self.model],
                  "aspect_ratio": payload["input"]["aspect_ratio"]})

    def _await(self, client: httpx.Client, pred: dict) -> str:
        """Poll until the prediction settles. Returns the URL of the image."""
        deadline = time.time() + self.timeout_s
        pid = pred.get("id")
        while True:
            status = pred.get("status")
            if status == "succeeded":
                out = pred.get("output")
                url = out[0] if isinstance(out, list) and out else out
                if not isinstance(url, str) or not url:
                    raise ProviderError(
                        f"replicate image prediction {pid} succeeded with no output",
                        provider=self.name)
                return url
            if status in ("failed", "canceled"):
                detail = str(pred.get("error"))[:200]
                # Replicate's own infrastructure faults are transient and say so - "Director:
                # unexpected error handling prediction (E9828)" is their scheduler, not our
                # prompt. Retrying the SAME provider costs one more 4-second generation;
                # falling through costs a 10-17 minute local render for an identical picture.
                # Measured on job 11 at image 59 of 63.
                #
                # Deliberately narrow. A prediction rejected for its CONTENT also lands here
                # as `failed`, and retrying that is a loop that ends in the same refusal, so
                # only the infrastructure wording is treated as retryable.
                if any(w in detail.lower() for w in
                       ("unexpected error", "internal", "director:", "try again",
                        "temporarily")):
                    raise RateLimited(
                        f"replicate image prediction {pid} hit a transient fault: {detail}",
                        provider=self.name, retry_after_s=5.0)
                raise ProviderError(
                    f"replicate image prediction {pid} {status}: {detail}",
                    provider=self.name)
            if time.time() > deadline:
                # A prediction still QUEUED at the deadline has not gone wrong - it has not
                # been given a machine yet, which is what a throttled account looks like
                # from the outside. Raising a plain ProviderError here sent the chain to the
                # next provider, and on a CPU-only box that is local diffusion at 10-17
                # MINUTES an image. Measured on job 11: 42 stills came back from Replicate
                # in eleven minutes, then one sat in `starting`, and the art stage began
                # loading SSD-1B for the remaining eighteen.
                #
                # RateLimited instead, so `images/factory` waits and asks this provider
                # again rather than giving up on it. `processing` is left as a hard error:
                # that one really was handed a machine and really is stuck.
                if status in ("starting", "queued"):
                    raise RateLimited(
                        f"replicate image prediction {pid} still {status} after "
                        f"{self.timeout_s:.0f}s - the account is queueing",
                        provider=self.name, retry_after_s=15.0)
                raise ProviderError(
                    f"replicate image prediction {pid} still {status} after "
                    f"{self.timeout_s:.0f}s", provider=self.name)
            time.sleep(self.poll_s)
            r = client.get(f"{API}/predictions/{pid}", headers=self._headers())
            self._raise_for(r)
            pred = r.json()

    def _download(self, client: httpx.Client, url: str) -> bytes:
        r = client.get(url, follow_redirects=True)
        if r.status_code >= 400:
            raise ProviderError(f"replicate image download HTTP {r.status_code}",
                                provider=self.name, retryable=r.status_code >= 500)
        return r.content
