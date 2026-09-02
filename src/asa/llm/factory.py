"""Assemble the LLM chain from config.

Every entry must be a real, no-card free tier. Providers whose key is absent are skipped
silently - the buffer degrades, it does not break.
"""
from __future__ import annotations

from pathlib import Path

from ..core.config import Config
from ..core.logging import get_logger
from ..core.quota import QuotaTracker
from .chain import LLMChain
from .local import LocalLlamaProvider
from .openai_compat import OpenAICompatProvider
from .openrouter import OpenRouterProvider
from .router import ModelRouter

log = get_logger("llm_factory")


def build_chain(cfg: Config, db: Path | None = None) -> LLMChain:
    db = db or cfg.path("paths.db", "data/asa.db")
    quota = QuotaTracker(db)
    providers: list = []
    order = cfg.get("providers.llm.chain", ["openrouter_free"])
    active, skipped = [], []

    for name in order:
        node = cfg.get(f"providers.llm.{name}", {}) or {}
        if not node.get("enabled", True):
            skipped.append(f"{name}(disabled)")
            continue
        try:
            p = _build_one(name, node, cfg, db, quota)
        except Exception as e:                                    # noqa: BLE001
            skipped.append(f"{name}({type(e).__name__})")
            continue
        if p is None:
            skipped.append(f"{name}(no key)")
            continue
        providers.append(p)
        active.append(name)

    log.info("llm_chain_built", active=active, skipped=skipped)
    if not providers:
        raise RuntimeError(
            "no LLM provider is usable. At minimum set OPENROUTER_API_KEY in config/.env")
    return LLMChain(providers)


def _secret(cfg: Config, name: str | None) -> str:
    if not name:
        return ""
    try:
        return cfg.secret(name, required=False)
    except Exception:                                             # noqa: BLE001
        return ""


def _build_one(name: str, node: dict, cfg: Config, db: Path, quota: QuotaTracker):
    if name == "openrouter_free":
        key = _secret(cfg, "OPENROUTER_API_KEY")
        if not key:
            return None
        router = ModelRouter(
            db, provider=name, api_key=key,
            pinned=node.get("models") or {},
            avoid_for_structured=tuple(node.get("avoid_for_structured_output") or ()))
        return OpenRouterProvider(
            api_key=key, router=router, quota=quota,
            rpm=node.get("rpm", 20), rpd=node.get("rpd", 50),
            app_name=_secret(cfg, "OPENROUTER_APP_NAME") or "asa",
            site_url=_secret(cfg, "OPENROUTER_SITE_URL"))

    if name == "local_llamacpp":
        p = LocalLlamaProvider(cfg.root / node.get("model_path", "models/model.gguf"),
                               threads=node.get("threads", 8), ctx=node.get("ctx", 8192))
        return p if p.available else None

    # everything else is an OpenAI-compatible free tier
    key = _secret(cfg, node.get("api_key_env"))
    models = node.get("models") or []
    if not key or not models:
        return None
    router = ModelRouter(db, provider=name, api_key=key)
    return OpenAICompatProvider(
        name=name, base_url=node["base_url"], api_key=key, models=models,
        router=router, quota=quota, rpm=node.get("rpm"), rpd=node.get("rpd"),
        extra_headers=node.get("headers") or {})
