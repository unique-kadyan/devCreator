"""Notifications. Deliberately small: the dashboard is the primary surface.

A pipeline that runs unattended needs exactly two kinds of message - "something needs you"
and "something broke" - and any driver that cannot be reached must never take the pipeline
down with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from .logging import get_logger

log = get_logger("notify")


class Notifier:
    def __init__(self, driver: str = "dashboard", url: str | None = None,
                 db: Path | None = None):
        self.driver = (driver or "none").lower()
        self.url = url
        self.db = db

    def send(self, title: str, message: str, level: str = "info",
             link: str | None = None) -> bool:
        try:
            if self.driver == "ntfy" and self.url:
                httpx.post(self.url, data=message.encode("utf-8"), timeout=10,
                           headers={"Title": title[:200],
                                    "Priority": {"error": "high", "warn": "default"}
                                    .get(level, "low"),
                                    **({"Click": link} if link else {})})
            elif self.driver == "webhook" and self.url:
                httpx.post(self.url, json={"title": title, "message": message,
                                           "level": level, "link": link}, timeout=10)
            # 'dashboard' and 'none' both fall through: the dashboard reads job state
            # straight from SQLite, so there is nothing to push.
        except httpx.HTTPError as e:
            # A failed notification must never fail a render.
            log.warning("notify_failed", driver=self.driver, error=str(e)[:140])
            return False
        log.info("notify", title=title[:80], level=level)
        return True

    def needs_human(self, job_id: int, reason: str) -> None:
        self.send(f"Job {job_id} needs review", reason, level="warn")

    def failed(self, job_id: int, stage: str, error: str) -> None:
        self.send(f"Job {job_id} failed at {stage}", error[:400], level="error")
