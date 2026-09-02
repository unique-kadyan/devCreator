"""Adaptive free-model router.

The problem this solves: free models are a shared, volatile resource. Measured 2026-09-01,
five of seven `:free` models returned 429 within one minute, one was 403-gated, and one
returned chain-of-thought instead of the requested JSON. A static model list is therefore not
a dependency you can build on.

So selection is learned. Every call updates per-model health in SQLite, and the next
selection scores models on what actually happened:

  score = success_rate * role_fitness * recency_penalty * speed_bonus

A 429 marks the model cold for a short window (the pool is busy, not you) and the router
advances immediately. Repeated schema failures demote a model for structured roles only - it
stays available for free-text work. The router only fails when EVERY free model has been
tried and none produced output.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from ..core.errors import AllProvidersExhausted, ProviderError
from ..core.logging import get_logger

log = get_logger("router")
CATALOG_TTL_S = 6 * 3600
COLD_ON_429_S = 90
COLD_ON_ERROR_S = 300
COLD_BACKOFF_MAX_S = 3600


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ModelStats:
    model_id: str
    calls: int = 0
    successes: int = 0
    rate_limits: int = 0
    errors: int = 0
    schema_failures: int = 0
    avg_latency_s: float = 0.0
    context_length: int = 0
    consecutive_failures: int = 0
    cold_until: datetime | None = None

    @property
    def success_rate(self) -> float:
        # Laplace smoothing: an untried model is optimistic but not reckless
        return (self.successes + 1.0) / (self.calls + 2.0)

    @property
    def schema_rate(self) -> float:
        return (self.schema_failures + 0.5) / (self.calls + 1.0)

    def is_cold(self) -> bool:
        return self.cold_until is not None and self.cold_until > _now()


class ModelRouter:
    def __init__(self, db: Path, provider: str = "openrouter_free",
                 api_key: str = "", pinned: dict[str, list[str]] | None = None,
                 avoid_for_structured: tuple[str, ...] = (), timeout: float = 30.0):
        self.db = Path(db)
        self.provider = provider
        self.api_key = api_key
        self.pinned = pinned or {}
        self.avoid_for_structured = set(avoid_for_structured)
        self.timeout = timeout

    # ---------------------------------------------------------------- storage

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def refresh_catalog(self, force: bool = False) -> list[str]:
        """Discover every free model currently on offer. Cached for CATALOG_TTL_S."""
        con = self._con()
        try:
            row = con.execute(
                "SELECT max(refreshed_at) AS t FROM model_catalog WHERE provider=?",
                (self.provider,)).fetchone()
            if not force and row and row["t"]:
                age = (_now() - datetime.fromisoformat(row["t"] + "+00:00")).total_seconds()
                if age < CATALOG_TTL_S:
                    return [r["model_id"] for r in con.execute(
                        "SELECT model_id FROM model_catalog WHERE provider=? AND is_free=1",
                        (self.provider,))]
        finally:
            con.close()

        try:
            r = httpx.get("https://openrouter.ai/api/v1/models", timeout=self.timeout)
            r.raise_for_status()
            models = [m for m in r.json().get("data", []) if m["id"].endswith(":free")]
        except Exception as e:                                   # noqa: BLE001
            log.warning("catalog_refresh_failed", error=str(e)[:120])
            con = self._con()
            try:
                return [r["model_id"] for r in con.execute(
                    "SELECT model_id FROM model_catalog WHERE provider=? AND is_free=1",
                    (self.provider,))]
            finally:
                con.close()

        con = self._con()
        try:
            for m in models:
                con.execute(
                    """INSERT INTO model_catalog (provider, model_id, context_length,
                                                  is_free, meta, refreshed_at)
                       VALUES (?,?,?,1,?,datetime('now'))
                       ON CONFLICT(provider, model_id) DO UPDATE SET
                         context_length=excluded.context_length, meta=excluded.meta,
                         is_free=1, refreshed_at=datetime('now')""",
                    (self.provider, m["id"], m.get("context_length") or 0,
                     json.dumps({"name": m.get("name", "")})))
            con.commit()
        finally:
            con.close()
        log.info("catalog_refreshed", models=len(models))
        return [m["id"] for m in models]

    def _stats(self, role: str) -> dict[str, ModelStats]:
        con = self._con()
        try:
            out: dict[str, ModelStats] = {}
            for r in con.execute(
                """SELECT h.*, c.context_length AS ctx FROM model_health h
                   LEFT JOIN model_catalog c
                     ON c.provider=h.provider AND c.model_id=h.model_id
                   WHERE h.provider=? AND h.role=?""", (self.provider, role)):
                out[r["model_id"]] = ModelStats(
                    model_id=r["model_id"], calls=r["calls"], successes=r["successes"],
                    rate_limits=r["rate_limits"], errors=r["errors"],
                    schema_failures=r["schema_failures"],
                    avg_latency_s=(r["total_latency_s"] / r["calls"]) if r["calls"] else 0.0,
                    context_length=r["ctx"] or r["context_length"] or 0,
                    consecutive_failures=r["consecutive_failures"],
                    cold_until=datetime.fromisoformat(r["cold_until"])
                    if r["cold_until"] else None)
            return out
        finally:
            con.close()

    # ---------------------------------------------------------------- selection

    def rank(self, role: str, structured: bool, min_context: int = 0) -> list[str]:
        """Every candidate, best first. Cold models go last rather than being dropped -
        if everything is cold we still try, because failing the job is worse."""
        catalog = self.refresh_catalog()
        pinned = [m for m in self.pinned.get(role, []) if m]
        pool = list(dict.fromkeys(pinned + catalog))
        if structured:
            pool = [m for m in pool if m not in self.avoid_for_structured] or pool
        stats = self._stats("structured" if structured else role)
        any_stats = self._stats("any")

        def score(mid: str) -> float:
            s = stats.get(mid) or any_stats.get(mid) or ModelStats(mid)
            v = s.success_rate
            if structured:
                v *= (1.0 - min(0.9, s.schema_rate))
            if s.avg_latency_s > 0:
                v *= 1.0 / (1.0 + s.avg_latency_s / 60.0)
            v *= 0.6 ** min(4, s.consecutive_failures)
            if min_context and s.context_length and s.context_length < min_context:
                v *= 0.25
            if mid in pinned:
                v *= 1.15                    # a human pinned it; nudge, don't override
            return v

        warm = [m for m in pool if not (stats.get(m) or ModelStats(m)).is_cold()]
        cold = [m for m in pool if m not in warm]
        return sorted(warm, key=score, reverse=True) + sorted(cold, key=score, reverse=True)

    # ---------------------------------------------------------------- feedback

    def _bump(self, model_id: str, role: str, **deltas) -> None:
        con = self._con()
        try:
            con.execute(
                "INSERT OR IGNORE INTO model_health (provider, model_id, role) VALUES (?,?,?)",
                (self.provider, model_id, role))
            sets = ", ".join(f"{k} = {k} + ?" for k in deltas if k != "_set")
            params = [v for k, v in deltas.items() if k != "_set"]
            extra = deltas.get("_set") or {}
            if extra:
                sets += (", " if sets else "") + ", ".join(f"{k} = ?" for k in extra)
                params += list(extra.values())
            con.execute(
                f"UPDATE model_health SET {sets}, updated_at = datetime('now') "
                f"WHERE provider=? AND model_id=? AND role=?",
                (*params, self.provider, model_id, role))
            con.commit()
        finally:
            con.close()

    def record_success(self, model_id: str, role: str, latency_s: float,
                       out_tokens: int = 0) -> None:
        for r in {role, "any"}:
            self._bump(r_id := model_id, r, calls=1, successes=1,
                       total_latency_s=latency_s, total_out_tokens=out_tokens,
                       _set={"consecutive_failures": 0, "cold_until": None,
                             "last_ok": _now().isoformat()})

    def record_rate_limit(self, model_id: str, role: str) -> None:
        cold = _now() + timedelta(seconds=COLD_ON_429_S)
        for r in {role, "any"}:
            self._bump(model_id, r, calls=1, rate_limits=1,
                       _set={"cold_until": cold.isoformat(),
                             "last_error": "429 upstream pool busy"})

    def record_error(self, model_id: str, role: str, message: str) -> None:
        stats = self._stats(role).get(model_id)
        n = (stats.consecutive_failures if stats else 0) + 1
        secs = min(COLD_BACKOFF_MAX_S, COLD_ON_ERROR_S * (2 ** min(4, n - 1)))
        cold = _now() + timedelta(seconds=secs)
        for r in {role, "any"}:
            self._bump(model_id, r, calls=1, errors=1, consecutive_failures=1,
                       _set={"cold_until": cold.isoformat(), "last_error": message[:200]})

    def record_schema_failure(self, model_id: str, role: str) -> None:
        """Parsed fine at HTTP level but the output did not satisfy the schema.
        Demotes the model for structured work only - it stays fine for prose."""
        for r in {"structured", role}:
            self._bump(model_id, r, calls=1, schema_failures=1, consecutive_failures=1)

    def record_empty(self, model_id: str, role: str) -> None:
        for r in {role, "any"}:
            self._bump(model_id, r, calls=1, empty_returns=1, consecutive_failures=1,
                       _set={"cold_until": (_now() + timedelta(
                           seconds=COLD_ON_ERROR_S)).isoformat()})

    # ---------------------------------------------------------------- reporting

    def report(self, role: str = "any", limit: int = 25) -> list[dict]:
        con = self._con()
        try:
            rows = con.execute(
                """SELECT model_id, calls, successes, rate_limits, errors, schema_failures,
                          consecutive_failures, cold_until,
                          CASE WHEN calls>0 THEN total_latency_s/calls ELSE 0 END AS avg_s
                   FROM model_health WHERE provider=? AND role=?
                   ORDER BY successes DESC, calls DESC LIMIT ?""",
                (self.provider, role, limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
