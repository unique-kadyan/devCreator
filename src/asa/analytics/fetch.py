"""Pull performance data for published videos.

Two different APIs are involved and they are not interchangeable. The Data API's
videos.list gives public counters (views, likes, comments) for 1 unit. The YouTube
Analytics API gives what actually matters - retention, average view duration, impressions,
CTR, traffic sources - and it only covers videos on a channel you own, with its own
authorisation scope.

Analytics data lags. A video published in the last 48 hours will return partial or empty
rows; that is normal and is not treated as an error.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..core.db import jdump, read, tx
from ..core.errors import AuthError, ProviderError, QuotaExhausted
from ..core.logging import get_logger

log = get_logger("analytics")

METRICS = ("views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,"
           "likes,dislikes,comments,shares,subscribersGained,subscribersLost")


def published_videos(db: Path, max_age_days: int = 120) -> list[dict]:
    with read(db) as con:
        rows = con.execute("""
            SELECT u.video_id, u.job_id, u.uploaded_at, v.duration_s, s.id AS story_id
            FROM youtube_uploads u
            JOIN videos v ON v.job_id = u.job_id
            JOIN stories s ON s.id = v.story_id
            WHERE u.video_id IS NOT NULL AND u.upload_status = 'uploaded'
              AND julianday('now') - julianday(u.uploaded_at) <= ?
        """, (max_age_days,)).fetchall()
    return [dict(r) for r in rows]


def fetch_counters(client, video_ids: list[str]) -> dict[str, dict]:
    """videos.list, 1 unit per call, up to 50 ids per call."""
    from googleapiclient.errors import HttpError
    out: dict[str, dict] = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        try:
            client._spend("videos.list")
            resp = client.service().videos().list(
                part="statistics,contentDetails", id=",".join(batch)).execute()
        except HttpError as e:
            if getattr(e.resp, "status", 0) == 403 and "quota" in str(e).lower():
                raise QuotaExhausted(f"analytics counters: {e}", provider="youtube") from e
            raise ProviderError(f"videos.list failed: {e}", provider="youtube") from e
        for item in resp.get("items", []):
            st = item.get("statistics", {})
            out[item["id"]] = {
                "views": int(st.get("viewCount", 0)),
                "likes": int(st.get("likeCount", 0)),
                "comments": int(st.get("commentCount", 0)),
            }
    return out


def fetch_analytics(client, video_id: str, start: dt.date, end: dt.date) -> dict:
    """YouTube Analytics API. Requires yt-analytics.readonly and channel ownership."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    try:
        svc = build("youtubeAnalytics", "v2", credentials=client._credentials(),
                    cache_discovery=False)
        resp = svc.reports().query(
            ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(),
            metrics=METRICS, filters=f"video=={video_id}").execute()
    except HttpError as e:
        status = getattr(e.resp, "status", 0)
        if status == 403:
            raise AuthError(
                "YouTube Analytics refused the request. The token needs the "
                "yt-analytics.readonly scope and the channel must be yours; re-run "
                "`asa youtube auth` after adding the scope.") from e
        raise ProviderError(f"analytics query failed: {e}", provider="youtube") from e

    rows = resp.get("rows") or []
    if not rows:
        return {}
    headers = [h["name"] for h in resp.get("columnHeaders", [])]
    return dict(zip(headers, rows[0]))


def fetch_retention(client, video_id: str, start: dt.date, end: dt.date) -> list[list[float]]:
    """Relative retention curve. Empty for videos below YouTube's reporting threshold -
    that is a data-availability fact, not a failure."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    try:
        svc = build("youtubeAnalytics", "v2", credentials=client._credentials(),
                    cache_discovery=False)
        resp = svc.reports().query(
            ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(),
            metrics="audienceWatchRatio", dimensions="elapsedVideoTimeRatio",
            filters=f"video=={video_id}").execute()
    except HttpError as e:
        log.info("retention_unavailable", video=video_id, error=str(e)[:120])
        return []
    return [[float(a), float(b)] for a, b in (resp.get("rows") or [])]


def snapshot(db: Path, client, max_age_days: int = 120) -> int:
    """One row per (video, day). Re-running the same day updates rather than duplicates."""
    vids = published_videos(db, max_age_days)
    if not vids:
        log.info("analytics_no_published_videos")
        return 0
    today = dt.date.today()
    counters = fetch_counters(client, [v["video_id"] for v in vids])
    written = 0
    for v in vids:
        vid = v["video_id"]
        published = dt.date.fromisoformat((v["uploaded_at"] or "")[:10] or today.isoformat())
        days = (today - published).days
        a = fetch_analytics(client, vid, published, today)
        curve = fetch_retention(client, vid, published, today) if days >= 2 else []
        c = counters.get(vid, {})
        impressions = a.get("impressions")
        with tx(db) as con:
            con.execute("""
                INSERT INTO analytics (video_id, snapshot_date, days_since_publish, views,
                    minutes_watched, avg_view_duration_s, avg_view_percentage, impressions,
                    ctr, likes, dislikes, comments, shares, subscribers_gained,
                    subscribers_lost, traffic_sources, retention_curve)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(video_id, snapshot_date) DO UPDATE SET
                    views=excluded.views, minutes_watched=excluded.minutes_watched,
                    avg_view_duration_s=excluded.avg_view_duration_s,
                    avg_view_percentage=excluded.avg_view_percentage,
                    likes=excluded.likes, comments=excluded.comments,
                    retention_curve=excluded.retention_curve
            """, (vid, today.isoformat(), days,
                  int(a.get("views", c.get("views", 0)) or 0),
                  float(a.get("estimatedMinutesWatched", 0) or 0),
                  float(a.get("averageViewDuration", 0) or 0) or None,
                  float(a.get("averageViewPercentage", 0) or 0) or None,
                  int(impressions) if impressions is not None else None,
                  None,
                  int(a.get("likes", c.get("likes", 0)) or 0),
                  int(a.get("dislikes", 0) or 0) or None,
                  int(a.get("comments", c.get("comments", 0)) or 0),
                  int(a.get("shares", 0) or 0),
                  int(a.get("subscribersGained", 0) or 0),
                  int(a.get("subscribersLost", 0) or 0),
                  jdump({}), jdump(curve) if curve else None))
        written += 1
    log.info("analytics_snapshot", videos=written)
    return written
