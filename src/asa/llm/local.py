"""Terminal fallback: a local GGUF model via llama-cpp-python.

Slow on this CPU (~5-10 tok/s for a 4B Q4), so a full story takes minutes - but it has no
quota, no network and no rate limit, which is exactly what a terminal fallback needs to be.
Absent model file => the provider reports itself unavailable and the chain skips it.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..core.errors import ProviderError
from ..core.logging import get_logger
from .base import Completion

log = get_logger("local_llm")
_LLM = None


class LocalLlamaProvider:
    name = "local_llamacpp"

    def __init__(self, model_path: str | Path, threads: int = 8, ctx: int = 8192):
        self.model_path = Path(model_path)
        self.threads = threads
        self.ctx = ctx

    @property
    def available(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return True

    def _model(self):
        global _LLM
        if _LLM is None:
            from llama_cpp import Llama
            _LLM = Llama(model_path=str(self.model_path), n_threads=self.threads,
                         n_ctx=self.ctx, verbose=False)
        return _LLM

    def complete(self, system: str, user: str, role: str = "story",
                 max_tokens: int = 4096, temperature: float = 0.9,
                 structured: bool = False) -> Completion:
        if not self.available:
            raise ProviderError(
                f"local model not available at {self.model_path} "
                f"(install llama-cpp-python and download a GGUF)",
                provider=self.name, retryable=False)
        t0 = time.time()
        out = self._model().create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=min(max_tokens, 3072), temperature=temperature)
        text = out["choices"][0]["message"]["content"]
        log.info("local_completion", seconds=round(time.time() - t0, 1))
        return Completion(text=text, model_id=self.model_path.name, provider=self.name,
                          completion_tokens=out.get("usage", {}).get("completion_tokens", 0))
