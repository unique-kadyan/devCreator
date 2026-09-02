"""Turn performance data into a bias on future topic selection.

The statistics matter here. With five episodes, one video doing 4x the others is noise, and
a system that chases it will lock onto that animal forever. So every feature score is
Bayesian-shrunk toward the channel mean by its sample size:

    shrunk = (n * mean_feature + k * mean_channel) / (n + k)

with k = 4. A feature seen once barely moves; a feature seen twenty times moves a lot. The
result is a `prefer` / `neutral` / `avoid` verdict per feature value, which the topic
scorer applies as a bounded nudge (+/-0.12), never as a veto.

What this deliberately does NOT do: change the story rules, shorten episodes to chase
retention, or narrow the channel to one format. Those are editorial decisions.
"""
from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass
from pathlib import Path

from ..core.db import read, tx
from ..core.logging import get_logger

log = get_logger("feedback")

SHRINK_K = 4.0
MIN_VIDEOS = 5           # below this the channel has no signal worth acting on
PREFER_MARGIN = 0.08     # how far above the mean a value must sit to earn 'prefer'


@dataclass
class FeatureScore:
    feature: str
    value: str
    n: int
    raw_mean: float
    shrunk: float
    verdict: str


def _performance_rows(db: Path, min_days: int = 7) -> list[dict]:
    """One row per video, using its most recent snapshot at least `min_days` old.

    Comparing a two-day-old video with a two-month-old one on raw views would rank by age,
    not by quality.
    """
    with read(db) as con:
        rows = con.execute("""
            SELECT a.video_id, a.views, a.avg_view_percentage, a.likes, a.comments,
                   a.subscribers_gained, a.days_since_publish,
                   s.archetype, s.id AS story_id, v.duration_s,
                   u.title
            FROM analytics a
            JOIN youtube_uploads u ON u.video_id = a.video_id
            JOIN videos v ON v.job_id = u.job_id
            JOIN stories s ON s.id = v.story_id
            WHERE a.days_since_publish >= ?
              AND a.snapshot_date = (SELECT MAX(snapshot_date) FROM analytics a2
                                     WHERE a2.video_id = a.video_id)
        """, (min_days,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["animals"] = [x[0] for x in con.execute(
                "SELECT DISTINCT c.species FROM story_cast sc "
                "JOIN characters c ON c.id = sc.character_id WHERE sc.story_id = ?",
                (d["story_id"],))]
            out.append(d)
    return out


def _quality(row: dict) -> float:
    """A single 0..1 number per video.

    Retention is weighted highest because it is the metric least polluted by how the video
    was surfaced, and engagement per view is used rather than raw counts so a video that
    happened to get promoted does not dominate.
    """
    retention = min(1.0, (row.get("avg_view_percentage") or 0) / 55.0)
    views = row.get("views") or 0
    engagement = min(1.0, ((row.get("likes") or 0) + 2 * (row.get("comments") or 0))
                     / max(1.0, views) * 40)
    subs = min(1.0, (row.get("subscribers_gained") or 0) / max(1.0, views) * 300)
    return round(0.60 * retention + 0.25 * engagement + 0.15 * subs, 4)


def _bucket_duration(seconds: float | None) -> str:
    if not seconds:
        return "unknown"
    m = seconds / 60
    for hi, name in ((4, "under_4m"), (7, "4_7m"), (10, "7_10m"), (15, "10_15m")):
        if m < hi:
            return name
    return "over_15m"


def compute(db: Path, min_days: int = 7) -> list[FeatureScore]:
    rows = _performance_rows(db, min_days)
    if len(rows) < MIN_VIDEOS:
        log.info("feedback_insufficient_data", videos=len(rows), needed=MIN_VIDEOS)
        return []

    scored = [(r, _quality(r)) for r in rows]
    channel_mean = statistics.fmean(q for _, q in scored)

    buckets: dict[tuple[str, str], list[float]] = {}
    for r, q in scored:
        buckets.setdefault(("archetype", r["archetype"] or "unknown"), []).append(q)
        buckets.setdefault(("duration_bucket", _bucket_duration(r["duration_s"])), []).append(q)
        for a in r["animals"]:
            buckets.setdefault(("animal", a), []).append(q)

    out: list[FeatureScore] = []
    for (feature, value), qs in sorted(buckets.items()):
        n = len(qs)
        raw = statistics.fmean(qs)
        shrunk = (n * raw + SHRINK_K * channel_mean) / (n + SHRINK_K)
        if shrunk > channel_mean + PREFER_MARGIN:
            verdict = "prefer"
        elif shrunk < channel_mean - PREFER_MARGIN:
            verdict = "avoid"
        else:
            verdict = "neutral"
        out.append(FeatureScore(feature, value, n, round(raw, 4), round(shrunk, 4), verdict))

    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    with tx(db) as con:
        con.executemany("""
            INSERT INTO strategy (computed_at, feature, value, n, raw_mean, shrunk, verdict)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(computed_at, feature, value) DO UPDATE SET
                n=excluded.n, raw_mean=excluded.raw_mean, shrunk=excluded.shrunk,
                verdict=excluded.verdict
        """, [(stamp, f.feature, f.value, f.n, f.raw_mean, f.shrunk, f.verdict)
              for f in out])
    log.info("feedback_computed", features=len(out), channel_mean=round(channel_mean, 4),
             prefer=sum(1 for f in out if f.verdict == "prefer"),
             avoid=sum(1 for f in out if f.verdict == "avoid"))
    return out


def current_strategy(db: Path) -> dict[str, float]:
    """`{"animal:fox": 0.62, "archetype:underdog": 0.41}` from the newest computation."""
    with read(db) as con:
        stamp = con.execute("SELECT MAX(computed_at) FROM strategy").fetchone()[0]
        if not stamp:
            return {}
        rows = con.execute(
            "SELECT feature, value, shrunk FROM strategy WHERE computed_at = ?",
            (stamp,)).fetchall()
    return {f"{r['feature']}:{r['value']}": float(r["shrunk"]) for r in rows}


def prompt_hints(db: Path) -> tuple[str, str]:
    """Human-readable prefer/avoid lines for the story prompt. Advisory only."""
    with read(db) as con:
        stamp = con.execute("SELECT MAX(computed_at) FROM strategy").fetchone()[0]
        if not stamp:
            return "", ""
        rows = [dict(r) for r in con.execute(
            "SELECT feature, value, verdict, n FROM strategy WHERE computed_at = ? "
            "AND verdict <> 'neutral' ORDER BY shrunk DESC", (stamp,))]
    prefer = ", ".join(f"{r['feature']} {r['value']}" for r in rows
                       if r["verdict"] == "prefer")
    avoid = ", ".join(f"{r['feature']} {r['value']}" for r in rows
                      if r["verdict"] == "avoid")
    return prefer, avoid
