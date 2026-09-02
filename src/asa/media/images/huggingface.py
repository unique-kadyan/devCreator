"""Hugging Face Inference Providers image generation.

Verified 2026-09-01 on this account: the legacy `hf-inference` provider returns HTTP 410
for FLUX.1-schnell and the OpenAI-style `/v1/images/generations` route returns 404. The
working path is `huggingface_hub.InferenceClient(provider="auto")`, which routes to
whichever third-party provider (nscale / fal-ai / wavespeed) currently serves the model.

Quota: HF bills a monthly *credit* window against a prepaid balance. `canPay: false` on
this account was confirmed via whoami-v2, which means exhaustion produces an error rather
than a charge. The exact allowance is not published by HF - treat the number in config as
an estimate and rely on the error path, not on the counter. NEEDS VERIFICATION.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.logging import get_logger
from .base import GeneratedImage, prompt_key

log = get_logger("img_hf")

# Only permissively licensed models belong here. A model whose weights forbid commercial
# use taints every frame it touches, and no amount of downstream editing cleans that.
DEFAULT_MODELS = [
    "black-forest-labs/FLUX.1-schnell",          # Apache-2.0
    "Tongyi-MAI/Z-Image-Turbo",
    "Qwen/Qwen-Image",                           # Apache-2.0
    "stabilityai/stable-diffusion-xl-base-1.0",  # OpenRAIL++-M
]


class HuggingFaceImages:
    name = "huggingface"

    def __init__(self, token: str | None, models: list[str] | None = None,
                 provider: str = "auto", negative_hints: str = "",
                 timeout_s: float = 180.0):
        self.token = (token or "").strip()
        self.models = models or DEFAULT_MODELS
        self.provider = provider
        self.negative_hints = negative_hints
        self.timeout_s = timeout_s
        self._client = None
        self._dead: dict[str, float] = {}      # model -> unix ts until which we skip it

    @property
    def available(self) -> bool:
        if not self.token or not self.models:
            return False
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return False
        return True

    def _client_for(self, model: str):
        from huggingface_hub import InferenceClient
        return InferenceClient(model=model, provider=self.provider, api_key=self.token,
                               timeout=self.timeout_s)

    def generate(self, prompt: str, out_path: Path, size: tuple[int, int],
                 negative: str = "", seed: int | None = None) -> GeneratedImage:
        if not self.available:
            raise ProviderError("huggingface image provider unavailable (no HF_TOKEN or "
                                "huggingface_hub not installed)", provider=self.name,
                                retryable=False)
        neg = ", ".join(x for x in (negative, self.negative_hints) if x.strip())
        key = prompt_key(prompt, size, neg)
        failures: list[str] = []
        now = time.time()

        for model in self.models:
            if self._dead.get(model, 0) > now:
                failures.append(f"{model}: cooling down")
                continue
            t0 = time.time()
            try:
                client = self._client_for(model)
                # negative_prompt is silently ignored by some routed providers; passing it
                # is still worth it for the ones that honour it.
                img = client.text_to_image(prompt, width=size[0], height=size[1],
                                           negative_prompt=neg or None, seed=seed)
            except Exception as e:                              # noqa: BLE001
                msg = str(e)
                low = msg.lower()
                if "401" in msg or "invalid" in low and "token" in low:
                    raise AuthError(f"HF token rejected: {msg[:200]}") from e
                if "402" in msg or "payment" in low or "credits" in low or "quota" in low:
                    raise QuotaExhausted(f"HF credits exhausted: {msg[:200]}",
                                         provider=self.name) from e
                # 410/404 mean this MODEL is unroutable right now - the account is fine.
                cool = 900 if ("410" in msg or "404" in msg or "not supported" in low) else 120
                self._dead[model] = time.time() + cool
                failures.append(f"{model}: {msg[:120]}")
                log.warning("hf_image_model_failed", model=model, error=msg[:160],
                            cooldown_s=cool)
                continue

            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img))
            if img.size != tuple(size):
                img = img.resize(size, Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(out_path)
            log.info("hf_image_ok", model=model, seconds=round(time.time() - t0, 2),
                     size=list(size))
            return GeneratedImage(path=out_path, provider=self.name, model_id=model,
                                  prompt_sha=key, seed=seed,
                                  meta={"routed_provider": self.provider,
                                        "seconds": round(time.time() - t0, 2)})

        raise ProviderError("every HF image model failed: " + " | ".join(failures),
                            provider=self.name)
