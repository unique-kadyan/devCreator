"""Pollinations.ai - keyless image generation.

DISABLED FOR PUBLICATION BY DEFAULT. Pollinations requires no key, which makes it a
tempting free tier, but its terms do not clearly grant commercial reuse of the outputs and
the underlying models are not disclosed per-request. The licence ledger therefore records
UNKNOWN for anything it produces, and `ledger.audit()` refuses to release a video
containing an UNKNOWN asset.

Enable it only if you have satisfied yourself about the terms for your use. It is wired in
so the decision is yours and explicit, not so it can be used by accident.
"""
from __future__ import annotations

import io
import urllib.parse
from pathlib import Path

import httpx
from PIL import Image

from ...core.errors import ProviderError, RateLimited
from ...core.logging import get_logger
from .base import GeneratedImage, prompt_key

log = get_logger("img_pollinations")
BASE = "https://image.pollinations.ai/prompt"


class PollinationsImages:
    name = "pollinations"

    def __init__(self, enabled: bool = False, timeout_s: float = 120.0,
                 model: str = "flux"):
        self.enabled = bool(enabled)
        self.timeout_s = timeout_s
        self.model = model

    @property
    def available(self) -> bool:
        return self.enabled

    def generate(self, prompt: str, out_path: Path, size: tuple[int, int],
                 negative: str = "", seed: int | None = None) -> GeneratedImage:
        if not self.enabled:
            raise ProviderError("pollinations disabled (licence terms unverified)",
                                provider=self.name, retryable=False)
        key = prompt_key(prompt, size, negative)
        params = {"width": size[0], "height": size[1], "nologo": "true",
                  "model": self.model, "safe": "true"}
        if seed is not None:
            params["seed"] = seed
        url = f"{BASE}/{urllib.parse.quote(prompt[:1400])}?{urllib.parse.urlencode(params)}"
        try:
            r = httpx.get(url, timeout=self.timeout_s,
                          follow_redirects=True)
        except httpx.HTTPError as e:
            raise ProviderError(f"pollinations network error: {e}",
                                provider=self.name) from e
        if r.status_code == 429:
            raise RateLimited("pollinations rate limited", provider=self.name,
                              retry_after_s=float(r.headers.get("retry-after", 20)))
        if r.status_code >= 400:
            raise ProviderError(f"pollinations HTTP {r.status_code}", provider=self.name)

        img = Image.open(io.BytesIO(r.content))
        if img.size != tuple(size):
            img = img.resize(size, Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out_path)
        return GeneratedImage(path=out_path, provider=self.name,
                              model_id=f"pollinations/{self.model}", prompt_sha=key,
                              seed=seed, meta={"licence": "UNVERIFIED"})
