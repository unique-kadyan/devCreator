"""Diffusion on this machine's CPU: slow, free, and not subject to anyone's daily quota.

WHY THIS EXISTS. Measured on job_00009: of 82 generated stills, 78 were 6.5 KB procedural
placeholders and 4 were real images. The hosted free tier had run out four pictures into a
thirty-scene episode, and everything after that was a grey rectangle with a voice over it.
That is not a rendering problem or a prompt problem - the pipeline behaved correctly and
degraded exactly as designed - it is simply the ceiling of a free allowance, and no amount
of retry logic raises it.

This provider trades time for that ceiling. There is no GPU here (Intel UHD 620), so a
picture costs minutes rather than the five seconds the hosted path takes. That is a bad
deal for one image and a good one for a night: the art stage is resumable and the image
cache is keyed by prompt, so an episode's stills can be generated overnight and every
re-run afterwards is free. It sits AFTER `huggingface` in the chain for that reason - spend
the fast free allowance first, fall back to the slow free one, and only then to a
placeholder.

LICENCE IS ENFORCED, NOT ASSUMED. `base.LICENCE_BY_PROVIDER` maps a provider to one licence
code, but a local runner can load any checkpoint, and the popular fast ones are exactly the
ones that forbid this use: SDXL-Turbo and SD-Turbo are Stability Non-Commercial, which a
monetised channel may not publish. Rather than let the provider name promise something the
loaded weights do not honour, `MODELS` below is an allowlist of checkpoints whose licence
permits commercial use, and a slug absent from it is refused at construction. Adding a model
therefore forces the same explicit licence decision that adding a provider does.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from ...core.errors import ProviderError
from ...core.logging import get_logger
from .base import GeneratedImage, prompt_key

log = get_logger("img_local")

# Checkpoints whose licence permits commercial publication, with the code the ledger records.
# Verified against each model card. Anything not listed here is refused - see the module
# docstring for why that is deliberate rather than inconvenient.
MODELS: dict[str, str] = {
    # Apache-2.0. Distilled SDXL at ~1.3B UNet parameters, which is the largest thing that
    # finishes in tolerable time on a laptop CPU while still being an SDXL-family model -
    # and SDXL is the family the hosted prompts were written against, so the same prompt
    # produces a recognisably similar picture rather than needing a second prompt dialect.
    "segmind/SSD-1B": "APACHE-2.0",
    # Apache-2.0. The same model the hosted path prefers, so output matches what the four
    # real stills in job_00009 already look like. 12B parameters: it will not fit in this
    # machine's RAM at full precision and is here for when the weights or the machine change.
    "black-forest-labs/FLUX.1-schnell": "APACHE-2.0",
    # Apache-2.0, already first-choice on the hosted chain.
    "Tongyi-MAI/Z-Image-Turbo": "APACHE-2.0",
    # CreativeML OpenRAIL-M: commercial use permitted, subject to the use-based restrictions
    # in the licence. Slowest-but-smallest escape hatch at 860M, for a machine that cannot
    # hold anything above.
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "CREATIVEML-OPENRAIL-M",
}

DEFAULT_MODEL = "segmind/SSD-1B"


def _chunked_embeds(pipe, prompt: str, negative: str):
    """Encode a prompt longer than CLIP's 77-token window, or return None to use the
    pipeline's own path.

    THE DEFECT THIS EXISTS FOR, measured on the live scene-2 prompt: it tokenises to 169
    tokens, CLIP truncates at 77, and the 92 that fall off the end are the whole back half
    of every prompt this repo writes - the location clause, the region hint, and all of
    CINEMATIC_STYLE. The generation came back as a technically clean studio portrait of a
    fox on a seamless backdrop: no bakery, no stone oven, nothing Indian, and none of the
    cinematic grade. Silently, with the picture looking good enough to pass a glance.

    The hosted path never hit this because FLUX conditions on T5 with a 512-token window.
    It is specific to the CLIP-based models - which is every checkpoint small enough to run
    on this machine, so it is not an edge case here, it is the normal case.

    The fix is the standard one: tokenise without truncation, cut into 75-token chunks with
    the model's own BOS/EOS around each, encode them separately and concatenate along the
    sequence axis. The text encoder never sees more than it was trained on; the UNet cross-
    attends over all of it.

    Returns None rather than raising when the pipeline is not a shape this understands -
    a truncated prompt still makes a picture, and failing the whole art stage over prompt
    length would be worse than the defect.
    """
    import torch
    tokenizers = [t for t in (getattr(pipe, "tokenizer", None),
                              getattr(pipe, "tokenizer_2", None)) if t is not None]
    encoders = [e for e in (getattr(pipe, "text_encoder", None),
                            getattr(pipe, "text_encoder_2", None)) if e is not None]
    if not tokenizers or len(tokenizers) != len(encoders):
        return None

    def encode(text: str, want_chunks: int | None):
        """One text through every encoder this pipeline has. SDXL has two and concatenates
        their outputs on the feature axis; SD 1.5 has one."""
        per_encoder, pooled = [], None
        for i, (tok, enc) in enumerate(zip(tokenizers, encoders)):
            ids = tok(text, truncation=False, add_special_tokens=False).input_ids
            size = tok.model_max_length - 2
            chunks = [ids[j:j + size] for j in range(0, len(ids), size)] or [[]]
            if want_chunks is not None:
                # The negative must come out the same sequence length as the positive or
                # classifier-free guidance cannot pair them up.
                chunks = (chunks + [[]] * want_chunks)[:want_chunks]
            outs = []
            for chunk in chunks:
                padded = ([tok.bos_token_id] + chunk
                          + [tok.eos_token_id] * (size - len(chunk) + 1))
                t = torch.tensor([padded], device=enc.device)
                out = enc(t, output_hidden_states=True)
                if len(encoders) == 2:
                    # SDXL conditions on the penultimate hidden layer, and takes its pooled
                    # vector from the SECOND encoder only.
                    outs.append(out.hidden_states[-2])
                    if i == 1 and pooled is None:
                        pooled = out[0]
                else:
                    outs.append(out.hidden_states[-2])
            per_encoder.append(torch.cat(outs, dim=1))
        embeds = (torch.cat(per_encoder, dim=-1) if len(per_encoder) > 1
                  else per_encoder[0])
        return embeds, pooled, len(chunks)

    with torch.no_grad():
        pos, pos_pooled, n_chunks = encode(prompt, None)
        neg, neg_pooled, _ = encode(negative or "", n_chunks)
    if pos.shape != neg.shape:
        return None
    out = {"prompt_embeds": pos, "negative_prompt_embeds": neg}
    if pos_pooled is not None and neg_pooled is not None:
        out["pooled_prompt_embeds"] = pos_pooled
        out["negative_pooled_prompt_embeds"] = neg_pooled
    return out


class LocalDiffusionImages:
    name = "local"

    def __init__(self, model: str = DEFAULT_MODEL, steps: int = 20,
                 guidance: float = 7.0, threads: int | None = None,
                 enabled: bool = False, cache_dir: str | None = None):
        self.enabled = bool(enabled)
        self.model = model or DEFAULT_MODEL
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.threads = int(threads) if threads else (os.cpu_count() or 4)
        self.cache_dir = cache_dir
        self._pipe = None
        # The art stage may render scenes concurrently; a diffusion pipeline is neither
        # thread-safe nor cheap enough to hold two of. One lock, one model in memory.
        self._lock = threading.Lock()
        if self.enabled and self.model not in MODELS:
            raise ProviderError(
                f"local diffusion model {self.model!r} is not on the commercial-use "
                f"allowlist; add it to media/images/local.MODELS with its licence code "
                f"first (permitted: {', '.join(sorted(MODELS))})",
                provider=self.name, retryable=False)

    @property
    def licence_code(self) -> str:
        return MODELS.get(self.model, "UNKNOWN")

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        try:                                           # noqa: SIM105
            import diffusers  # noqa: F401
            import torch      # noqa: F401
        except ImportError:
            # Not an error worth failing a render for: the chain simply skips to the next
            # provider, exactly as it does for a missing API key.
            log.info("local_diffusion_unavailable", reason="diffusers/torch not installed")
            return False
        return True

    def _pipeline(self):
        """Load once and keep. Loading costs far more than a generation does, so a pipeline
        built per image would spend the whole night in `from_pretrained`."""
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import AutoPipelineForText2Image
        torch.set_num_threads(self.threads)
        t0 = time.time()
        log.info("local_diffusion_loading", model=self.model, threads=self.threads)
        # `dtype` since diffusers 0.31; `torch_dtype` is deprecated and removed in 1.0.
        # float32 rather than bf16: this CPU (Kaby Lake R) has no bf16 instructions, so a
        # half-precision load is emulated and slower, not faster.
        kwargs = {"dtype": torch.float32}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        pipe = AutoPipelineForText2Image.from_pretrained(self.model, **kwargs)
        pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)
        # Attention slicing trades a little speed for a much smaller peak, which is what
        # decides whether a 1024-wide generation completes at all on 31 GB shared with the
        # rest of the pipeline.
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        log.info("local_diffusion_loaded", model=self.model,
                 load_s=round(time.time() - t0, 1))
        self._pipe = pipe
        return pipe

    def generate(self, prompt: str, out_path: Path, size: tuple[int, int],
                 negative: str = "", seed: int | None = None) -> GeneratedImage:
        if not self.enabled:
            raise ProviderError("local diffusion disabled", provider=self.name,
                                retryable=False)
        import torch
        key = prompt_key(prompt, size, negative)
        # Diffusion needs both dimensions on an 8-pixel grid; the configured 1024x576 is
        # already there, but a hand-edited size that is not would fail deep inside the UNet
        # with a shape error rather than here.
        w, h = (max(8, (v // 8) * 8) for v in size)
        gen = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        t0 = time.time()
        with self._lock:
            pipe = self._pipeline()
            call = {"width": w, "height": h, "num_inference_steps": self.steps,
                    "guidance_scale": self.guidance, "generator": gen}
            embeds = None
            try:
                embeds = _chunked_embeds(pipe, prompt[:1400], negative[:600])
            except Exception as e:                                        # noqa: BLE001
                # A truncated prompt still draws something; a crashed art stage does not.
                log.warning("local_diffusion_long_prompt_failed", error=str(e)[:160])
            if embeds:
                call.update(embeds)
            else:
                call["prompt"] = prompt[:1400]
                call["negative_prompt"] = negative[:600] or None
            try:
                image = pipe(**call).images[0]
            except (RuntimeError, MemoryError, ValueError) as e:
                # Out of memory is the expected failure on this machine and it must advance
                # the chain rather than kill the episode.
                raise ProviderError(f"local diffusion failed: {e}", provider=self.name,
                                    retryable=False) from e
        if image.size != (size[0], size[1]):
            from PIL import Image as _Image
            image = image.resize((size[0], size[1]), _Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        elapsed = round(time.time() - t0, 1)
        log.info("local_diffusion_generated", model=self.model, seconds=elapsed,
                 size=[w, h], steps=self.steps)
        return GeneratedImage(path=out_path, provider=self.name, model_id=self.model,
                              prompt_sha=key, cached=False, seed=seed,
                              meta={"seconds": elapsed, "steps": self.steps,
                                    "licence": self.licence_code})
