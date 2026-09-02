"""Configuration loading: YAML for settings, .env for secrets. Never mix the two."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[3]


class Config:
    """Dot/slash-path access over the YAML tree, with secrets kept separate."""

    def __init__(self, data: dict[str, Any], secrets: dict[str, str], root: Path):
        self._data = data
        self._secrets = secrets
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def secret(self, name: str, required: bool = True) -> str:
        """Secrets come from the process env first, then config/.env."""
        val = os.environ.get(name) or self._secrets.get(name) or ""
        if required and not val:
            raise RuntimeError(
                f"Missing secret {name}. Add it to config/.env (chmod 600) or export it."
            )
        return val

    def path(self, path: str, default: str | None = None) -> Path:
        """Resolve a configured path relative to the project root."""
        raw = self.get(path, default)
        if raw is None:
            raise KeyError(path)
        p = Path(raw)
        return p if p.is_absolute() else self.root / p


@lru_cache(maxsize=1)
def load_config(config_file: str | None = None) -> Config:
    root = ROOT
    cfg_path = Path(config_file) if config_file else Path(
        os.environ.get("ASA_CONFIG", root / "config" / "config.yaml")
    )
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Copy config/config.example.yaml to config/config.yaml."
        )
    data = yaml.safe_load(cfg_path.read_text()) or {}
    env_path = root / "config" / ".env"
    secrets = dict(dotenv_values(env_path)) if env_path.exists() else {}
    return Config(data, {k: v for k, v in secrets.items() if v}, root)
