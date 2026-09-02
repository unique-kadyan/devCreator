"""The approval dashboard: watch the episode, read the QC report, approve or reject.

This is the human gate the whole design rests on, so it is deliberately boring: server-
rendered HTML, no build step, no JavaScript framework, no auth beyond binding to loopback.
It reads the same SQLite the runner writes, which is why the DB uses WAL - a page refresh
during a render must not block the render.

Binding: 127.0.0.1 by default. If you expose this beyond loopback you are publishing an
unauthenticated approve/reject/delete surface onto your network. Put it behind a reverse
proxy with authentication first.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..core.db import read
from ..core.logging import get_logger

log = get_logger("dashboard")

CSS = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#14120f;color:#eee8dd}
a{color:#ffb765;text-decoration:none}a:hover{text-decoration:underline}
header{padding:18px 26px;border-bottom:1px solid #2e2822;display:flex;gap:20px;
       align-items:baseline;background:#1b1813;position:sticky;top:0;z-index:5}
header h1{font-size:17px;margin:0;letter-spacing:.02em}
header nav{display:flex;gap:16px;font-size:14px}
main{padding:26px;max-width:1180px;margin:0 auto}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:#9a9086;font-weight:600;padding:9px 10px;border-bottom:1px solid #2e2822}
td{padding:9px 10px;border-bottom:1px solid #221e19;vertical-align:top}
tr:hover td{background:#1b1813}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
.s-ok{background:#1d3a24;color:#8ee6a1}.s-wait{background:#3a3418;color:#f0d97a}
.s-bad{background:#3d1e1e;color:#f2a0a0}.s-run{background:#1e2f3d;color:#8fc9f2}
.card{background:#1b1813;border:1px solid #2e2822;border-radius:10px;padding:20px;
      margin-bottom:20px}
.grid{display:grid;grid-template-columns:1.35fr .95fr;gap:22px}
video,img.thumb{width:100%;border-radius:8px;background:#000;display:block}
button{font:inherit;font-weight:600;padding:10px 18px;border-radius:8px;border:0;
       cursor:pointer}
.approve{background:#2f7d47;color:#fff}.reject{background:#8c3535;color:#fff}
input[type=text]{width:100%;padding:9px 11px;border-radius:8px;border:1px solid #3a332b;
                 background:#14120f;color:#eee8dd;font:inherit;margin:8px 0 12px}
.k{color:#9a9086;font-size:13px}.v{font-weight:600}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;margin:0;font-size:14px}
dt{color:#9a9086}dd{margin:0}
.finding{padding:8px 11px;border-radius:7px;margin-bottom:7px;font-size:13.5px}
.f-fail{background:#3d1e1e}.f-warn{background:#3a3418}.f-info{background:#1e2229}
pre{white-space:pre-wrap;font-size:13px;color:#c9c0b4;background:#14120f;padding:12px;
    border-radius:7px;overflow-x:auto}
.empty{color:#9a9086;padding:40px 0;text-align:center}
"""

STATE_CLASS = {
    "AWAITING_APPROVAL": "s-wait", "QC_PASSED": "s-ok", "APPROVED": "s-ok",
    "UPLOADED": "s-ok", "PUBLISHED": "s-ok", "FAILED": "s-bad", "REJECTED": "s-bad",
    "QUOTA_BLOCKED": "s-wait",
}


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<header><h1>Animal Story Automation</h1><nav>
<a href="/">Queue</a><a href="/all">All jobs</a><a href="/topics">Topics</a>
<a href="/models">Models</a></nav></header><main>{body}</main></body></html>""")


def _pill(state: str) -> str:
    return (f'<span class="pill {STATE_CLASS.get(state, "s-run")}">'
            f'{html.escape(state)}</span>')


def create_app(ctx) -> FastAPI:
    app = FastAPI(title="asa dashboard", docs_url=None, redoc_url=None)
    db = ctx.db

    # ------------------------------------------------------------------ queue

    @app.get("/", response_class=HTMLResponse)
    def queue() -> HTMLResponse:
        with read(db) as con:
            rows = con.execute("""
                SELECT j.id, j.state, j.needs_human, s.title, v.duration_s, v.lufs,
                       v.thumbnail_path, v.qc_report
                FROM jobs j
                LEFT JOIN stories s ON s.id = j.story_id
                LEFT JOIN videos v ON v.job_id = j.id
                WHERE j.state IN ('AWAITING_APPROVAL','QC_PASSED','FAILED','QUOTA_BLOCKED')
                ORDER BY j.needs_human DESC, j.id DESC
            """).fetchall()
        if not rows:
            return _page("Queue", '<div class="empty">Nothing is waiting for you.<br>'
                                  'Queue an episode with <code>asa job new</code>.</div>')
        cells = []
        for r in rows:
            qc = json.loads(r["qc_report"] or "{}")
            warn = qc.get("warn", 0)
            cells.append(f"""<tr>
<td><a href="/job/{r['id']}">#{r['id']}</a></td>
<td>{html.escape(r['title'] or '(no story yet)')}</td>
<td>{_pill(r['state'])}</td>
<td>{(r['duration_s'] or 0) / 60:.1f} min</td>
<td>{f"{r['lufs']:.1f} LUFS" if r['lufs'] else '-'}</td>
<td>{warn} warning{'' if warn == 1 else 's'}</td></tr>""")
        return _page("Queue", f"""<div class="card"><h2 style="margin-top:0">Waiting on you</h2>
<table><tr><th>Job</th><th>Title</th><th>State</th><th>Length</th><th>Loudness</th>
<th>QC</th></tr>{''.join(cells)}</table></div>""")

    # ------------------------------------------------------------------ job

    @app.get("/job/{job_id}", response_class=HTMLResponse)
    def job(job_id: int) -> HTMLResponse:
        with read(db) as con:
            j = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if j is None:
                return _page("Not found", '<div class="empty">No such job.</div>')
            v = con.execute("SELECT * FROM videos WHERE job_id = ?", (job_id,)).fetchone()
            s = con.execute("SELECT * FROM stories WHERE id = ?",
                            (j["story_id"],)).fetchone() if j["story_id"] else None
            titles = con.execute(
                "SELECT value, score, chosen FROM metadata_candidates WHERE job_id = ? "
                "ORDER BY chosen DESC, score DESC", (job_id,)).fetchall()
            thumbs = con.execute(
                "SELECT variant, path, score, chosen FROM thumbnails WHERE job_id = ? "
                "ORDER BY score DESC", (job_id,)).fetchall()
            stages = con.execute(
                "SELECT stage, status, attempts, duration_s FROM job_stages "
                "WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
            errs = con.execute(
                "SELECT stage, kind, message, created_at FROM errors WHERE job_id = ? "
                "ORDER BY id DESC LIMIT 5", (job_id,)).fetchall()

        qc = json.loads(v["qc_report"] or "{}") if v else {}
        findings = "".join(
            f'<div class="finding f-{f["severity"]}"><b>{html.escape(f["check"])}</b> '
            f'&mdash; {html.escape(f["message"])}</div>'
            for f in qc.get("findings", []) if f["severity"] != "info") or \
            '<div class="k">No findings.</div>'

        # The `info` findings are the measurements QC actually took. They are filtered out
        # of the findings list above (which is for things needing attention), but they are
        # exactly what a reviewer wants to confirm before approving.
        facts: dict = {}
        for f in qc.get("findings", []):
            if f["severity"] == "info":
                facts.update(f.get("detail") or {})
        keep = [("duration", "Length", lambda v: f"{float(v) / 60:.2f} min"),
                ("lufs", "Loudness", lambda v: f"{float(v):.1f} LUFS"),
                ("true_peak_db", "True peak", lambda v: f"{float(v):.1f} dBTP"),
                ("width", "Frame", lambda v: f"{v}x{facts.get('height', '?')}"),
                ("fps", "Rate", lambda v: f"{float(v):.0f} fps"),
                ("vcodec", "Video", str), ("acodec", "Audio", str),
                ("sample_rate", "Sample rate", lambda v: f"{int(v) / 1000:.0f} kHz"),
                ("audio_start", "A/V offset",
                 lambda v: f"{abs(float(v) - float(facts.get('video_start', 0))):.3f} s")]
        measured = "".join(
            f"<dt>{label}</dt><dd>{fmt(facts[key])}</dd>"
            for key, label, fmt in keep if key in facts) or (
            '<dd class="k">not measured yet</dd>')

        player = (f'<video controls preload="metadata" src="/media/video/{job_id}"></video>'
                  if v and Path(v["path"]).exists() else
                  '<div class="empty">No video rendered yet.</div>')
        thumb_html = "".join(
            f'<div style="margin-bottom:10px"><img class="thumb" '
            f'src="/media/thumb/{job_id}/{t["variant"]}">'
            f'<div class="k">variant {t["variant"]} &middot; {t["score"]:.3f}'
            f'{" &middot; chosen" if t["chosen"] else ""}</div></div>'
            for t in thumbs[:3])

        title_rows = "".join(
            f'<tr><td>{"&#9679;" if t["chosen"] else ""}</td>'
            f'<td>{html.escape(t["value"])}</td><td>{t["score"]:.3f}</td></tr>'
            for t in titles)
        stage_rows = "".join(
            f'<tr><td>{html.escape(st["stage"])}</td><td>{_pill(st["status"].upper())}</td>'
            f'<td>{st["attempts"]}</td><td>{(st["duration_s"] or 0):.1f}s</td></tr>'
            for st in stages)
        error_html = "".join(
            f'<div class="finding f-fail"><b>{html.escape(e["stage"] or "?")}</b> '
            f'({html.escape(e["kind"])}) {html.escape(e["message"][:400])}</div>'
            for e in errs)

        actions = ""
        if j["state"] in ("AWAITING_APPROVAL", "QC_PASSED"):
            actions = f"""<form method="post" action="/job/{job_id}/approve"
 style="display:inline">
<input type="text" name="who" placeholder="your name" value="human">
<button class="approve" type="submit">Approve &amp; upload</button></form>
<form method="post" action="/job/{job_id}/reject" style="display:inline;margin-left:10px">
<input type="text" name="reason" placeholder="why">
<button class="reject" type="submit">Reject</button></form>"""

        story_html = ""
        if s:
            story_html = f"""<dl>
<dt>Logline</dt><dd>{html.escape(s['logline'])}</dd>
<dt>Hook</dt><dd>{html.escape(s['hook'])}</dd>
<dt>Archetype</dt><dd>{html.escape(s['archetype'])}</dd>
<dt>Moral</dt><dd>{html.escape(s['moral'])}</dd>
<dt>Skeleton</dt><dd><code>{html.escape(s['beat_signature'])}</code></dd></dl>"""

        return _page(f"Job {job_id}", f"""
<h2 style="margin-top:0">#{job_id} {html.escape(s['title'] if s else 'untitled')}
 {_pill(j['state'])}</h2>
<div class="grid">
  <div>
    <div class="card">{player}
      <div style="margin-top:16px">{actions}</div></div>
    <div class="card"><h3 style="margin-top:0">QC</h3>{findings}
      <h4 style="margin:16px 0 8px;color:#9a9086;font-size:13px;
                 text-transform:uppercase;letter-spacing:.06em">Measured</h4>
      <dl>{measured}</dl></div>
    <div class="card"><h3 style="margin-top:0">Story</h3>{story_html}</div>
  </div>
  <div>
    <div class="card"><h3 style="margin-top:0">Thumbnails</h3>{thumb_html or
      '<div class="k">none</div>'}</div>
    <div class="card"><h3 style="margin-top:0">Titles</h3>
      <table>{title_rows or '<tr><td class="k">none</td></tr>'}</table></div>
    <div class="card"><h3 style="margin-top:0">Stages</h3>
      <table><tr><th>Stage</th><th>Status</th><th>Try</th><th>Time</th></tr>
      {stage_rows}</table></div>
    {f'<div class="card"><h3 style="margin-top:0">Recent errors</h3>{error_html}</div>'
     if error_html else ''}
  </div>
</div>""")

    @app.post("/job/{job_id}/approve")
    def approve(job_id: int, who: str = Form("human")) -> RedirectResponse:
        from ..core.runner import Runner
        Runner(ctx).approve(job_id, who=who or "human")
        return RedirectResponse(f"/job/{job_id}", status_code=303)

    @app.post("/job/{job_id}/reject")
    def reject(job_id: int, reason: str = Form("")) -> RedirectResponse:
        from ..core.runner import Runner
        Runner(ctx).reject(job_id, reason or "rejected in dashboard")
        return RedirectResponse(f"/job/{job_id}", status_code=303)

    # ------------------------------------------------------------------ media

    @app.get("/media/video/{job_id}")
    def media_video(job_id: int):
        with read(db) as con:
            row = con.execute("SELECT path FROM videos WHERE job_id = ?",
                              (job_id,)).fetchone()
        if row is None or not Path(row["path"]).exists():
            return HTMLResponse("not found", status_code=404)
        return FileResponse(row["path"], media_type="video/mp4")

    @app.get("/media/thumb/{job_id}/{variant}")
    def media_thumb(job_id: int, variant: int):
        with read(db) as con:
            row = con.execute(
                "SELECT path FROM thumbnails WHERE job_id = ? AND variant = ?",
                (job_id, variant)).fetchone()
        if row is None or not Path(row["path"]).exists():
            return HTMLResponse("not found", status_code=404)
        return FileResponse(row["path"], media_type="image/jpeg")

    # ------------------------------------------------------------------ lists

    @app.get("/all", response_class=HTMLResponse)
    def all_jobs() -> HTMLResponse:
        with read(db) as con:
            rows = con.execute("""
                SELECT j.id, j.state, j.updated_at, s.title, v.duration_s
                FROM jobs j LEFT JOIN stories s ON s.id = j.story_id
                LEFT JOIN videos v ON v.job_id = j.id
                ORDER BY j.id DESC LIMIT 80""").fetchall()
        body = "".join(
            f'<tr><td><a href="/job/{r["id"]}">#{r["id"]}</a></td>'
            f'<td>{html.escape(r["title"] or "-")}</td><td>{_pill(r["state"])}</td>'
            f'<td>{(r["duration_s"] or 0)/60:.1f} min</td><td class="k">'
            f'{r["updated_at"]}</td></tr>' for r in rows)
        return _page("All jobs", f'<div class="card"><table><tr><th>Job</th><th>Title</th>'
                                 f'<th>State</th><th>Length</th><th>Updated</th></tr>'
                                 f'{body}</table></div>')

    @app.get("/topics", response_class=HTMLResponse)
    def topics() -> HTMLResponse:
        with read(db) as con:
            rows = con.execute(
                "SELECT id, topic, source, primary_animal, overall_score, status "
                "FROM research_topics ORDER BY overall_score DESC LIMIT 60").fetchall()
        body = "".join(
            f'<tr><td>{r["overall_score"]:.3f}</td><td>{html.escape(r["topic"][:110])}</td>'
            f'<td class="k">{html.escape(r["source"])}</td>'
            f'<td class="k">{html.escape(r["primary_animal"] or "-")}</td>'
            f'<td>{_pill(r["status"].upper())}</td></tr>' for r in rows)
        return _page("Topics", f'<div class="card"><table><tr><th>Score</th><th>Topic</th>'
                               f'<th>Source</th><th>Animal</th><th>Status</th></tr>'
                               f'{body}</table></div>')

    @app.get("/models", response_class=HTMLResponse)
    def models() -> HTMLResponse:
        with read(db) as con:
            rows = con.execute("""
                SELECT provider, model_id, role, calls, successes, rate_limits,
                       schema_failures, cold_until, last_ok
                FROM model_health WHERE calls > 0
                ORDER BY (CAST(successes AS REAL) + 1) / (calls + 2) DESC, calls DESC
                LIMIT 60""").fetchall()
        if not rows:
            return _page("Models", '<div class="empty">No model has been called yet.</div>')
        body = "".join(
            f'<tr><td class="k">{html.escape(r["provider"])}</td>'
            f'<td>{html.escape(r["model_id"])}</td><td class="k">{r["role"]}</td>'
            f'<td>{(r["successes"] + 1) / (r["calls"] + 2):.2f}</td>'
            f'<td>{r["calls"]}</td><td>{r["rate_limits"]}</td>'
            f'<td>{r["schema_failures"]}</td>'
            f'<td class="k">{html.escape(r["cold_until"] or "-")}</td></tr>' for r in rows)
        return _page("Models", f'<div class="card"><table><tr><th>Provider</th>'
                               f'<th>Model</th><th>Role</th><th>Score</th><th>Calls</th>'
                               f'<th>429s</th><th>Schema</th><th>Cold until</th></tr>'
                               f'{body}</table></div>')

    return app
