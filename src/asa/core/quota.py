"""Persisted token buckets.

In the DB, not memory: a crash at 3am must not reset the OpenRouter daily counter and burn
the next day's allowance. Windows are calendar-aligned (per-day / per-minute keys) because
that is how the upstream services actually reset.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .errors import QuotaExhausted


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(t: datetime) -> str:
    return t.strftime("%Y-%m-%d")


def _minute_key(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M")


@dataclass
class Limits:
    rpm: int | None = None
    rpd: int | None = None
    units_per_day: float | None = None


class QuotaTracker:
    def __init__(self, db: Path):
        self.db = Path(db)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _used(self, con, provider: str, window_key: str) -> sqlite3.Row | None:
        return con.execute(
            "SELECT requests, units_used, cold_until FROM provider_usage "
            "WHERE provider=? AND window_key=?", (provider, window_key)).fetchone()

    def check(self, provider: str, limits: Limits, units: float = 0.0) -> None:
        """Raise QuotaExhausted if this call would exceed a limit. Does not consume."""
        now = _now()
        con = self._con()
        try:
            cold = con.execute(
                "SELECT max(cold_until) AS c FROM provider_usage WHERE provider=?",
                (provider,)).fetchone()["c"]
            if cold and datetime.fromisoformat(cold) > now:
                raise QuotaExhausted(f"{provider} cold until {cold}", provider=provider,
                                     resets_at=datetime.fromisoformat(cold))
            if limits.rpd is not None:
                row = self._used(con, provider, _day_key(now))
                if row and row["requests"] >= limits.rpd:
                    reset = (now + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
                    raise QuotaExhausted(
                        f"{provider} daily cap {limits.rpd} reached", provider=provider,
                        resets_at=reset)
            if limits.rpm is not None:
                row = self._used(con, provider, _minute_key(now))
                if row and row["requests"] >= limits.rpm:
                    raise QuotaExhausted(f"{provider} per-minute cap {limits.rpm} reached",
                                         provider=provider,
                                         resets_at=now + timedelta(seconds=60))
            if limits.units_per_day is not None and units:
                row = self._used(con, provider, _day_key(now))
                spent = row["units_used"] if row else 0.0
                if spent + units > limits.units_per_day:
                    raise QuotaExhausted(
                        f"{provider} unit budget {limits.units_per_day} would be exceeded",
                        provider=provider)
        finally:
            con.close()

    def consume(self, provider: str, units: float = 0.0, requests: int = 1) -> None:
        now = _now()
        con = self._con()
        try:
            for key in (_day_key(now), _minute_key(now)):
                con.execute(
                    """INSERT INTO provider_usage (provider, window_key, units_used, requests)
                       VALUES (?,?,?,?)
                       ON CONFLICT(provider, window_key) DO UPDATE SET
                         units_used = units_used + excluded.units_used,
                         requests   = requests   + excluded.requests""",
                    (provider, key, units, requests))
            con.commit()
        finally:
            con.close()

    def mark_cold(self, provider: str, until: datetime) -> None:
        """Park a provider after a 429/quota response so the chain skips it."""
        con = self._con()
        try:
            con.execute(
                """INSERT INTO provider_usage (provider, window_key, cold_until)
                   VALUES (?,?,?)
                   ON CONFLICT(provider, window_key) DO UPDATE SET
                     cold_until = excluded.cold_until""",
                (provider, _day_key(_now()), until.isoformat()))
            con.commit()
        finally:
            con.close()

    def usage_today(self, provider: str) -> dict:
        con = self._con()
        try:
            row = self._used(con, provider, _day_key(_now()))
            return {"requests": row["requests"] if row else 0,
                    "units": row["units_used"] if row else 0.0}
        finally:
            con.close()
