"""The job runner: advance one job through the state machine, or run it to completion.

Design points that earn their keep:

* **Leases.** A job is claimed with an expiring lease before work starts, so a second
  runner (a stray systemd timer, a manual `asa run` while the timer fires) cannot render
  the same episode twice. An expired lease is reclaimable, which means a hard kill does not
  wedge a job forever.
* **Resume is the default.** Every stage's output is durable before the state advances, so
  restarting after a crash re-enters at the failed stage, not at the beginning. Rendering
  an episode costs ~18 minutes of CPU; losing that to a dropped connection at the metadata
  stage would be absurd.
* **Errors are classified, not counted.** A quota error parks the job with a retry time. A
  policy violation stops it dead and asks for a human. Only genuinely transient failures
  consume an attempt.
* **Stages are bounded.** Every stage runs under the wall-clock budget its `timeout_key`
  names in `runtime.timeouts_s`. Provider-level timeouts are not enough on their own:
  httpx's is a per-read timeout, so a response that trickles bytes resets it forever and
  an unattended run can block on one socket indefinitely. Only a deadline above the
  provider catches that.
"""
from __future__ import annotations

import datetime as dt
import os
import signal
import socket
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .db import jdump, read, tx
from .errors import (AllProvidersExhausted, ASAError, AuthError, PolicyViolation,
                     QuotaExhausted, RateLimited, RenderError, ValidationError)
from .logging import get_logger
from .stages import REGISTRY, Stage, by_state

log = get_logger("runner")

TERMINAL = {"UPLOADED", "PUBLISHED", "ANALYZED", "REJECTED", "FAILED"}
PAUSED = {"AWAITING_APPROVAL", "QUOTA_BLOCKED"}
MAX_ATTEMPTS = 3
LEASE_S = 3 * 3600


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class StageTimeout(BaseException):
    """Deadline expiry, deliberately outside the `Exception` hierarchy.

    This started as a plain TimeoutError and silently did nothing. TimeoutError is an
    OSError, so httpx's transport mapped it to ConnectTimeout like any socket failure, the
    OpenRouter client turned that into ProviderError, and the model loop read it as "this
    model failed" and advanced to the next one. Because signal.alarm() is one-shot, the
    deadline then no longer existed and the stage ran on unbounded - 788 seconds against a
    300 second budget, with nothing in the logs to say the timeout had been swallowed.

    Deriving from BaseException is the trick KeyboardInterrupt uses, for exactly this
    reason: a deadline must not be catchable by ordinary `except Exception` error handling
    that is trying to be helpful about a different problem.
    """


@contextmanager
def _deadline(seconds: int, label: str):
    """Hard wall-clock bound on a block of work.

    SIGALRM rather than a worker thread because a stage is not cancellable co-operatively -
    it is usually parked in a blocking socket read inside a provider SDK, and only a signal
    interrupts that. The guards matter: signal handlers can only be installed on the main
    thread, so the dashboard and the tests (which drive stages off-thread) fall through to
    running unbounded rather than raising ValueError.
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _fire(signum, frame):
        raise StageTimeout(f"stage {label} exceeded its {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class StageResult:
    stage: str
    ok: bool
    state: str
    detail: dict
    seconds: float
    error: str = ""


class Runner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.db: Path = ctx.db

    # ------------------------------------------------------------------ jobs

    def create_job(self, topic_id: int | None = None, fmt: str = "long") -> int:
        with tx(self.db) as con:
            cur = con.execute(
                "INSERT INTO jobs (topic_id, state, format) VALUES (?, 'RESEARCHED', ?)",
                (topic_id, fmt))
            job_id = cur.lastrowid
        paths = self.ctx.paths_for(job_id)
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET work_dir = ?, out_dir = ? WHERE id = ?",
                        (str(paths.work), str(paths.out), job_id))
        log.info("job_created", job=job_id, topic_id=topic_id)
        return job_id

    def get(self, job_id: int) -> dict:
        with read(self.db) as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job {job_id} not found")
        return dict(row)

    def claim(self, job_id: int) -> bool:
        """Atomic: only succeeds if nobody holds a live lease."""
        expires = (_now() + dt.timedelta(seconds=LEASE_S)).isoformat(timespec="seconds")
        with tx(self.db) as con:
            cur = con.execute("""
                UPDATE jobs SET lease_owner = ?, lease_expires = ?,
                                updated_at = datetime('now')
                WHERE id = ?
                  AND (lease_owner IS NULL OR lease_expires IS NULL
                       OR lease_expires < ?)
            """, (_owner(), expires, job_id, _now().isoformat(timespec="seconds")))
            return cur.rowcount == 1

    def release(self, job_id: int) -> None:
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET lease_owner = NULL, lease_expires = NULL "
                        "WHERE id = ?", (job_id,))

    def ready_jobs(self) -> list[dict]:
        now = _now().isoformat(timespec="seconds")
        with read(self.db) as con:
            rows = con.execute("""
                SELECT * FROM jobs
                WHERE state NOT IN ('UPLOADED','PUBLISHED','ANALYZED','REJECTED','FAILED',
                                    'AWAITING_APPROVAL')
                  AND (retry_after IS NULL OR retry_after <= ?)
                  AND (lease_owner IS NULL OR lease_expires IS NULL OR lease_expires < ?)
                ORDER BY id
            """, (now, now)).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- stages

    def step(self, job_id: int) -> StageResult:
        """Run exactly one stage. The unit the CLI, the timer and the tests all share."""
        job = self.get(job_id)
        state = job["state"]
        if state in TERMINAL:
            return StageResult("-", True, state, {"note": "terminal"}, 0.0)
        if state in PAUSED:
            return StageResult("-", True, state, {"note": "waiting for a human"}, 0.0)

        stage = by_state(state)
        if stage is None:
            return StageResult("-", False, state,
                               {"note": f"no stage consumes state {state!r}"}, 0.0,
                               error=f"unroutable state {state!r}")
        if not self.claim(job_id):
            return StageResult(stage.name, False, state, {"note": "leased elsewhere"}, 0.0,
                               error="job is leased by another runner")

        self._begin(job_id, stage)
        t0 = time.time()
        # getattr rather than self.ctx.cfg: tests drive step() with a minimal context stub
        # that carries only what a stage touches, and requiring config there would make
        # every future fake carry a settings object to exercise unrelated behaviour. A
        # context without config simply has no configured budget, which is the same as
        # leaving the key unset.
        cfg = getattr(self.ctx, "cfg", None)
        budget = int((cfg.get(f"runtime.timeouts_s.{stage.timeout_key}", 0) if cfg else 0) or 0)
        # Logged because a deadline that silently fails to arm is indistinguishable from one
        # that never needed to fire, and the difference only shows up as a wedged unattended
        # run hours later.
        log.info("stage_deadline", stage=stage.name, key=stage.timeout_key, budget_s=budget)
        try:
            with _deadline(budget, stage.name):
                detail = stage.fn(self.ctx, job) or {}
        except BaseException as e:                             # noqa: BLE001
            elapsed = time.time() - t0
            return self._handle_error(job_id, stage, e, elapsed)
        finally:
            self.release(job_id)

        elapsed = time.time() - t0
        # A stage may have set its own terminal-ish state (approval auto-approves).
        current = self.get(job_id)["state"]
        new_state = current if current not in (state,) else stage.to_state
        self._finish(job_id, stage, new_state, detail, elapsed)
        # A stage's own detail may legitimately contain keys this call already binds
        # (`animate` reports its own `seconds`). Dropping the collisions is right: the
        # runner's timing is the authoritative one, and a TypeError here would abort the
        # job AFTER the work succeeded, which is the worst possible place to fail.
        reserved = {"job", "stage", "state", "seconds", "event", "level", "timestamp"}
        extra = {k: v for k, v in detail.items()
                 if k not in reserved and isinstance(v, (int, float, str, bool))}
        log.info("stage_done", job=job_id, stage=stage.name, state=new_state,
                 seconds=round(elapsed, 1), **extra)
        return StageResult(stage.name, True, new_state, detail, elapsed)

    def run(self, job_id: int, max_stages: int = 40) -> list[StageResult]:
        """Advance until the job finishes, pauses for a human, or fails."""
        results: list[StageResult] = []
        for _ in range(max_stages):
            r = self.step(job_id)
            results.append(r)
            if not r.ok:
                break
            if r.state in TERMINAL or r.state in PAUSED:
                break
            if r.detail.get("note") in ("terminal", "waiting for a human"):
                break
        return results

    def retry(self, job_id: int) -> str:
        """Put a FAILED job back at the entry of the stage that failed.

        Everything before that stage is already durable, so this re-runs only the work that
        actually broke - which for a job that died at `metadata` means not re-rendering
        eighteen minutes of video.
        """
        with read(self.db) as con:
            row = con.execute(
                "SELECT stage FROM job_stages WHERE job_id = ? AND status = 'failed' "
                "ORDER BY id DESC LIMIT 1", (job_id,)).fetchone()
        stage = next((s for s in REGISTRY if row and s.name == row["stage"]), None)
        target = stage.from_state if stage else "RESEARCHED"
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET state=?, needs_human=0, retry_after=NULL, "
                        "lease_owner=NULL, lease_expires=NULL, finished_at=NULL, "
                        "updated_at=datetime('now') WHERE id=?", (target, job_id))
            if stage:
                con.execute("UPDATE job_stages SET status='pending', attempts=0 "
                            "WHERE job_id=? AND stage=?", (job_id, stage.name))
        log.info("job_retry", job=job_id, from_stage=stage.name if stage else "start",
                 state=target)
        return target

    def approve(self, job_id: int, who: str = "human") -> None:
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET state = 'APPROVED', needs_human = 0, "
                        "retry_after = NULL, updated_at = datetime('now') WHERE id = ?",
                        (job_id,))
            con.execute("UPDATE youtube_uploads SET approved_by = ?, "
                        "approved_at = datetime('now') WHERE job_id = ?", (who, job_id))
        log.info("job_approved", job=job_id, by=who)

    def reject(self, job_id: int, reason: str) -> None:
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET state = 'REJECTED', needs_human = 0, "
                        "finished_at = datetime('now') WHERE id = ?", (job_id,))
            con.execute("INSERT INTO errors (job_id, stage, kind, message) "
                        "VALUES (?,'approval','policy',?)", (job_id, reason[:800]))
        log.info("job_rejected", job=job_id, reason=reason[:120])

    # ---------------------------------------------------------------- private

    def _begin(self, job_id: int, stage: Stage) -> None:
        with tx(self.db) as con:
            con.execute("""
                INSERT INTO job_stages (job_id, stage, status, attempts, started_at)
                VALUES (?,?, 'running', 1, datetime('now'))
                ON CONFLICT(job_id, stage) DO UPDATE SET
                    status='running', attempts = job_stages.attempts + 1,
                    started_at = datetime('now')
            """, (job_id, stage.name))

    def _finish(self, job_id: int, stage: Stage, new_state: str, detail: dict,
                seconds: float) -> None:
        with tx(self.db) as con:
            con.execute("""
                UPDATE job_stages SET status='done', finished_at=datetime('now'),
                    duration_s=?, output_ref=? WHERE job_id=? AND stage=?
            """, (round(seconds, 2), jdump(detail)[:2000], job_id, stage.name))
            con.execute("""
                UPDATE jobs SET state=?, retry_after=NULL, updated_at=datetime('now'),
                    finished_at = CASE WHEN ? IN ('UPLOADED','PUBLISHED','ANALYZED')
                                       THEN datetime('now') ELSE finished_at END
                WHERE id=?
            """, (new_state, new_state, job_id))

    def _handle_error(self, job_id: int, stage: Stage, exc: BaseException,
                      seconds: float) -> StageResult:
        kind = getattr(exc, "kind", "unknown")
        message = str(exc)[:900]
        tb = traceback.format_exc()[-4000:]

        with tx(self.db) as con:
            con.execute("INSERT INTO errors (job_id, stage, kind, message, traceback) "
                        "VALUES (?,?,?,?,?)", (job_id, stage.name, kind, message, tb))
            con.execute("UPDATE job_stages SET status='failed', finished_at=datetime('now'),"
                        " duration_s=? WHERE job_id=? AND stage=?",
                        (round(seconds, 2), job_id, stage.name))
            attempts = con.execute(
                "SELECT attempts FROM job_stages WHERE job_id=? AND stage=?",
                (job_id, stage.name)).fetchone()["attempts"]

        if isinstance(exc, (QuotaExhausted, AllProvidersExhausted)):
            # Not a failure of the job - a failure of the moment. Park and come back.
            resets = getattr(exc, "resets_at", None)
            retry = (resets or (_now() + dt.timedelta(hours=1))).isoformat(timespec="seconds")
            self._park(job_id, "QUOTA_BLOCKED", retry)
            log.warning("job_quota_blocked", job=job_id, stage=stage.name, retry_after=retry)
            self.ctx.notifier.send(f"Job {job_id} paused on quota",
                                   f"{stage.name}: {message}", level="warn")
            return StageResult(stage.name, False, "QUOTA_BLOCKED", {}, seconds, message)

        if isinstance(exc, RateLimited):
            retry = (_now() + dt.timedelta(
                seconds=getattr(exc, "retry_after_s", 60))).isoformat(timespec="seconds")
            self._park(job_id, self.get(job_id)["state"], retry)
            return StageResult(stage.name, False, "RATE_LIMITED", {}, seconds, message)

        if isinstance(exc, (PolicyViolation, AuthError)):
            # Never retried. A policy failure retried is a policy failure twice.
            self._fail(job_id, needs_human=True)
            log.error("job_needs_human", job=job_id, stage=stage.name, kind=kind,
                      error=message)
            self.ctx.notifier.failed(job_id, stage.name, message)
            return StageResult(stage.name, False, "FAILED", {}, seconds, message)

        if not stage.retryable or attempts >= MAX_ATTEMPTS:
            self._fail(job_id, needs_human=True)
            log.error("job_failed", job=job_id, stage=stage.name, attempts=attempts,
                      kind=kind, error=message)
            self.ctx.notifier.failed(job_id, stage.name, message)
            return StageResult(stage.name, False, "FAILED", {}, seconds, message)

        delay = min(900, 30 * 2 ** (attempts - 1))
        retry = (_now() + dt.timedelta(seconds=delay)).isoformat(timespec="seconds")
        self._park(job_id, self.get(job_id)["state"], retry)
        log.warning("stage_retry_scheduled", job=job_id, stage=stage.name,
                    attempt=attempts, in_s=delay, error=message[:160])
        return StageResult(stage.name, False, self.get(job_id)["state"], {}, seconds,
                           message)

    def _park(self, job_id: int, state: str, retry_after: str) -> None:
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET state=?, retry_after=?, lease_owner=NULL, "
                        "lease_expires=NULL, attempts=attempts+1, "
                        "updated_at=datetime('now') WHERE id=?",
                        (state, retry_after, job_id))

    def _fail(self, job_id: int, needs_human: bool = True) -> None:
        with tx(self.db) as con:
            con.execute("UPDATE jobs SET state='FAILED', needs_human=?, lease_owner=NULL, "
                        "lease_expires=NULL, finished_at=datetime('now'), "
                        "updated_at=datetime('now') WHERE id=?",
                        (int(needs_human), job_id))
