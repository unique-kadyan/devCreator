"""asa - command line entry point."""
from __future__ import annotations

import json
import argparse
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


def _p(status: str, label: str, detail: str = "") -> bool:
    colour = {"OK  ": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}[status]
    print(f"  {colour}{status}\033[0m  {label:<34} {detail}")
    return status != BAD


# ---------------------------------------------------------------- doctor

def cmd_doctor(args) -> int:
    from .config import load_config
    from ..media.tts.kokoro_tts import KokoroTTS  # noqa: F401  (import cost is the test)

    ok = True
    print("\nsystem")
    ok &= _p(OK if sys.version_info[:2] == (3, 12) else WARN, "python",
             f"{sys.version.split()[0]} (3.12 expected)")
    for tool in ("ffmpeg", "ffprobe", "sqlite3", "espeak-ng"):
        path = shutil.which(tool)
        ok &= _p(OK if path else BAD, tool, path or "not on PATH")
    try:
        import PIL, numpy, soundfile, torch  # noqa: F401
        ok &= _p(OK, "python deps", f"torch {torch.__version__}")
        if torch.version.cuda:
            _p(WARN, "torch build", "CUDA wheel on a machine with no GPU - wasted disk")
    except ImportError as e:
        ok &= _p(BAD, "python deps", str(e))

    print("\nconfig")
    try:
        cfg = load_config()
        ok &= _p(OK, "config.yaml", f"channel={cfg.get('channel.name')!r}")
        mfk = cfg.get("channel.made_for_kids")
        ok &= _p(OK if isinstance(mfk, bool) else BAD, "made_for_kids",
                 f"{mfk} (must be set explicitly - see docs/05 §2)")
        placeholders = [k for k in ("providers.tts.kokoro_local.narrator_voice",
                                    "subtitles.font")
                        if "REPLACE" in str(cfg.get(k, ""))]
        _p(OK if not placeholders else WARN, "placeholders",
           "none" if not placeholders else ", ".join(placeholders))
    except Exception as e:
        return 1 if not _p(BAD, "config.yaml", str(e)) else 1

    print("\nsecrets")
    env = ROOT / "config/.env"
    if not env.exists():
        ok &= _p(BAD, "config/.env", "missing - copy config/.env.example")
    else:
        mode = stat.S_IMODE(env.stat().st_mode)
        ok &= _p(OK if mode == 0o600 else BAD, "config/.env perms",
                 f"{oct(mode)} (must be 0o600)")
        expect = {"OPENROUTER_API_KEY": ("sk-or-v1-", True),
                  "GROQ_API_KEY": ("", False),
                  "GEMINI_API_KEY": ("", False),
                  "CLOUDFLARE_API_TOKEN": ("", False),
                  "HF_TOKEN": ("hf_", True),
                  "FREESOUND_API_KEY": ("", False),
                  "YT_CLIENT_ID": (".apps.googleusercontent.com", False),
                  "YT_CLIENT_SECRET": ("", False)}
        for name, (marker, required) in expect.items():
            try:
                val = cfg.secret(name, required=False)
            except Exception:
                val = ""
            if not val:
                ok &= _p(BAD if required else WARN, name, "empty")
            elif marker and not (val.startswith(marker) or val.endswith(marker)):
                ok &= _p(BAD, name, f"set but does not look like {marker!r}")
            else:
                _p(OK, name, f"set ({len(val)} chars)")

    print("\nllm buffer")
    try:
        from ..llm.factory import build_chain
        chain = build_chain(cfg)
        names = [p.name for p in chain.providers]
        _p(OK, "providers active", ", ".join(names))
        try:
            from ..llm.router import ModelRouter
            r = ModelRouter(ROOT / "data/asa.db")
            free = r.refresh_catalog()
            _p(OK if len(free) >= 3 else WARN, "openrouter free models",
               f"{len(free)} discovered")
        except Exception as e:                                    # noqa: BLE001
            _p(WARN, "openrouter catalogue", str(e)[:60])
        if len(names) == 1:
            _p(WARN, "buffer depth", "only 1 provider - add GROQ_API_KEY / GEMINI_API_KEY "
                                     "for headroom when free pools are busy")
    except Exception as e:                                        # noqa: BLE001
        ok &= _p(BAD, "llm chain", str(e)[:80])

    print("\ngit safety")
    gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
    for pat in ("config/.env", "data/", "logs/"):
        ok &= _p(OK if pat in gi else BAD, f"gitignore {pat}",
                 "covered" if pat in gi else "NOT IGNORED")

    print("\ndata")
    db = ROOT / "data/asa.db"
    if db.exists():
        n = sqlite3.connect(db).execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        ok &= _p(OK if n >= 20 else WARN, "database", f"{db.name}, {n} tables")
    else:
        ok &= _p(BAD, "database", "missing - sqlite3 data/asa.db < migrations/001_initial.sql")
    for label, path, hint in (
        ("characters", ROOT / "assets/characters", "asa character new"),
        ("backgrounds", ROOT / "assets/backgrounds", "generated per scene"),
        ("music library", ROOT / "assets/music", "docs/07 §8 - MANUAL, required before publish"),
        ("sfx library", ROOT / "assets/sfx", "freesound or manual"),
    ):
        files = [p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"]
        empty = not files
        crit = BAD if (empty and label == "music library") else (WARN if empty else OK)
        _p(crit, label, f"{len(files)} files" + (f" - {hint}" if empty else ""))
        if crit == BAD:
            ok = False
    print()
    # ------------------------------------------------------------------ pipeline
    print("\npipeline")
    try:
        con = sqlite3.connect(str(db))
        con.row_factory = sqlite3.Row
        states = {r["state"]: r["n"] for r in con.execute(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")}
        waiting = states.get("AWAITING_APPROVAL", 0)
        failed = states.get("FAILED", 0)
        blocked = states.get("QUOTA_BLOCKED", 0)
        if not states:
            _p(WARN, "jobs", "none queued - try: asa job new --topic '...'")
        else:
            _p(OK, "jobs", ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
        if waiting:
            _p(WARN, "awaiting approval", f"{waiting} - review with: asa dashboard")
        if failed:
            row = con.execute(
                "SELECT stage, message FROM errors ORDER BY id DESC LIMIT 1").fetchone()
            _p(BAD, "failed jobs",
               f"{failed}; last error at {row['stage'] if row else '?'}: "
               f"{(row['message'][:70] if row else '')}")
        if blocked:
            _p(WARN, "quota blocked", f"{blocked} - will resume automatically")
        health = con.execute(
            "SELECT COUNT(*) FROM model_health WHERE calls > 0").fetchone()[0]
        best = con.execute(
            "SELECT model_id, (successes + 1.0) / (calls + 2) AS s FROM model_health "
            "WHERE calls > 2 ORDER BY s DESC LIMIT 1").fetchone()
        if health:
            _p(OK, "model health",
               f"{health} model(s) tracked" + (f"; best {best['model_id']} "
                                               f"({best['s']:.0%})" if best else ""))
        else:
            _p(WARN, "model health", "no model has been called yet")
        con.close()
    except sqlite3.Error as e:
        _p(WARN, "pipeline", f"could not read: {e}")

    return 0 if ok else 1


# ---------------------------------------------------------------- bench

def cmd_bench(args) -> int:
    """Measure this machine so the docs' numbers can be re-derived, not trusted."""
    from PIL import Image
    from ..characters.procedural_puppet import FoxPuppet, MILO_PALETTE
    from ..characters.rig import Rig
    from ..media.animation.camera import Camera
    from ..media.animation.compositor import CharacterInstance, SceneRenderer
    from ..media.animation.parallax import multiplane

    print(f"\n  cpu threads: {os.cpu_count()}   frames: {args.frames}\n")
    cdir = ROOT / "assets/characters/milo_fox"
    if not (cdir / "rig.json").exists():
        FoxPuppet("milo_fox", MILO_PALETTE).build(cdir)
    rig = Rig.load(cdir / "rig.json")
    plate_path = ROOT / "assets/backgrounds/forest_village/plate.png"
    plate = Image.open(plate_path) if plate_path.exists() else Image.new(
        "RGB", (1536, 864), (140, 110, 80))

    world, frame = (2688, 1512), (1920, 1080)
    for move in ("static", "push_in"):
        r = SceneRenderer(world, frame, multiplane(plate, world),
                          [CharacterInstance(rig=rig, base_dir=cdir, blink_seed=1)],
                          Camera(world, frame, move=move), duration=args.frames / 24, fps=24)
        r.render_frame(0)
        t0 = time.time()
        for i in range(args.frames):
            r.render_frame(i)
        dt = (time.time() - t0) / args.frames
        print(f"  composite  {move:<10} {dt*1000:6.1f} ms/frame   "
              f"{1/dt:5.1f} fps single-threaded")

    if args.tts:
        from ..media.tts.kokoro_tts import KokoroTTS
        tts = KokoroTTS()
        text = ("Milo had walked this path a hundred times, and never once been inside "
                "the little shuttered bakery on the corner.")
        out = ROOT / "data/work/bench_tts.wav"
        t0 = time.time()
        tts.synthesize(text, "bm_fable", out)   # includes ~24s init on a cold process
        t1 = time.time()
        u = tts.synthesize(text + " Not once.", "bm_fable", out)
        dt = time.time() - t1
        print(f"\n  kokoro     cold init  {t1-t0:5.1f} s")
        print(f"  kokoro     synth      {u.duration_s/dt:5.2f}x realtime "
              f"({dt:.2f}s for {u.duration_s:.2f}s)")
    print()
    return 0


# ---------------------------------------------------------------- character / assets

def cmd_character_new(args) -> int:
    from ..characters.procedural_puppet import FoxPuppet, MILO_PALETTE
    out = ROOT / "assets/characters" / args.character_id
    rig = FoxPuppet(args.character_id, MILO_PALETTE).build(out)
    print(f"  built {len(rig.layers)} layers -> {out}")
    print(f"  style_hash {rig.style_hash}")
    return 0


def cmd_assets_add(args) -> int:
    from .ledger import add_asset
    row = add_asset(ROOT / "data/asa.db", Path(args.path), kind=args.kind,
                    source=args.source, license_code=args.license,
                    attribution=args.attribution, source_ref=args.url)
    print(f"  registered #{row} {args.path} as {args.license}")
    return 0


def cmd_assets_audit(args) -> int:
    from .ledger import audit
    problems = audit(ROOT / "data/asa.db")
    if not problems:
        print("  all registered assets are cleared for commercial use")
        return 0
    for p in problems:
        print(f"  BLOCKED  {p['path']}  ({p['reason']})")
    return 1


def cmd_assets_scan(args) -> int:
    from .ledger import unregistered
    dirs = [ROOT / "assets/music", ROOT / "assets/sfx"]
    on_disk = [p for d in dirs for p in d.rglob("*")
               if p.is_file() and p.suffix.lower() in
               (".wav", ".mp3", ".flac", ".ogg", ".m4a")]
    if not on_disk:
        print("  assets/music and assets/sfx contain NO audio files.")
        print("  This is not a pass - nothing can be published without a music bed.")
        print("  See docs/07 §8 to build the licensed library.")
        return 1
    missing = unregistered(ROOT / "data/asa.db", dirs)
    if not missing:
        print(f"  all {len(on_disk)} audio file(s) in assets/music and assets/sfx "
              f"have a licence row")
        return 0
    print(f"  {len(missing)} file(s) on disk with NO licence row - these cannot be published:")
    for m in missing:
        print(f"    {m}")
    print("\n  register each with:  asa assets add <path> --kind music --source "
          "yt_audio_library --license YT-AUDIO-LIB")
    return 1



# ---------------------------------------------------------------- pipeline

def _context(args):
    from .config import Config, load_config
    cfg = load_config()
    overrides = getattr(args, "set", None) or []
    if overrides:
        # `--set production.target_minutes=2` - for smoke tests and one-off runs. Kept out
        # of the config file so a temporary tweak cannot silently become permanent.
        import copy
        data = copy.deepcopy(cfg._data)
        for item in overrides:
            key, _, raw = item.partition("=")
            node = data
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            node[parts[-1]] = value
        cfg = Config(data, cfg._secrets, cfg.root)
    from .context import Context
    return Context(cfg)


def cmd_research(args) -> int:
    from ..research.collectors import collect_all
    from ..research.scoring import ingest, select_next
    from ..analytics.feedback import current_strategy
    ctx = _context(args)
    cands = collect_all(ctx.cfg, quota=ctx.quota)
    n = ingest(ctx.db, cands, weights=ctx.cfg.get("research.weights"),
               strategy=current_strategy(ctx.db))
    top = select_next(ctx.db)
    print(f"  collected {len(cands)} candidate(s), stored {n}")
    if top:
        print(f"  next up ({top['overall_score']:.3f}): {top['topic'][:100]}")
    else:
        print("  nothing scores above the selection threshold yet")
    return 0


def cmd_job_new(args) -> int:
    from .runner import Runner
    from ..research.scoring import ingest
    from ..research.collectors import Candidate, _animal_in, _keywords
    ctx = _context(args)
    topic_id = args.topic_id
    if args.topic:
        from .db import read, tx
        ingest(ctx.db, [Candidate(topic=args.topic, keywords=_keywords(args.topic),
                                  primary_animal=_animal_in(args.topic), source="manual")])
        with read(ctx.db) as con:
            topic_id = con.execute(
                "SELECT id FROM research_topics ORDER BY id DESC LIMIT 1").fetchone()[0]
        with tx(ctx.db) as con:
            con.execute("UPDATE research_topics SET status='used' WHERE id=?", (topic_id,))
    job_id = Runner(ctx).create_job(topic_id=topic_id, fmt=args.format)
    print(f"  job {job_id} created (topic_id={topic_id})")
    if topic_id:
        from .db import tx
        with tx(ctx.db) as con:
            con.execute("UPDATE jobs SET state='TOPIC_SELECTED' WHERE id=?", (job_id,))
        print("  state: TOPIC_SELECTED")
    return 0


def cmd_job_retry(args) -> int:
    from .runner import Runner
    state = Runner(_context(args)).retry(args.job_id)
    print(f"  job {args.job_id} reset to {state}")
    return 0


def cmd_run(args) -> int:
    from .runner import Runner
    ctx = _context(args)
    r = Runner(ctx)
    job_ids = [args.job_id] if args.job_id else [j["id"] for j in r.ready_jobs()]
    if not job_ids:
        print("  no jobs are ready")
        return 0
    rc = 0
    for jid in job_ids[: args.max_jobs]:
        print(f"\n  === job {jid} ===")
        results = [r.step(jid)] if args.once else r.run(jid)
        for res in results:
            mark = "ok  " if res.ok else "FAIL"
            print(f"  {mark} {res.stage:<14} -> {res.state:<18} {res.seconds:6.1f}s "
                  f"{res.error[:80]}")
            if not res.ok:
                rc = 1
    return rc


def cmd_jobs(args) -> int:
    from .db import read
    ctx = _context(args)
    with read(ctx.db) as con:
        rows = con.execute(
            "SELECT id, state, needs_human, story_id, retry_after, updated_at "
            "FROM jobs ORDER BY id DESC LIMIT ?", (args.limit,)).fetchall()
    if not rows:
        print("  no jobs")
        return 0
    print(f"  {'id':>4}  {'state':<18} {'story':>6}  {'human':<6} updated")
    for r in rows:
        print(f"  {r['id']:>4}  {r['state']:<18} {str(r['story_id'] or '-'):>6}  "
              f"{'YES' if r['needs_human'] else '-':<6} {r['updated_at']}"
              + (f"  retry@{r['retry_after']}" if r["retry_after"] else ""))
    return 0


def cmd_approve(args) -> int:
    from .runner import Runner
    Runner(_context(args)).approve(args.job_id, who=args.who)
    print(f"  job {args.job_id} approved by {args.who}")
    return 0


def cmd_reject(args) -> int:
    from .runner import Runner
    Runner(_context(args)).reject(args.job_id, args.reason)
    print(f"  job {args.job_id} rejected")
    return 0


def cmd_youtube_auth(args) -> int:
    ctx = _context(args)
    if not ctx.youtube.configured:
        print("  YT_CLIENT_ID / YT_CLIENT_SECRET are not set in config/.env")
        return 1
    path = ctx.youtube.authorise(port=args.port)
    print(f"  token stored at {path} (mode {oct(path.stat().st_mode & 0o777)})")
    print("  NOTE: until your API project passes Google's audit, every API upload is")
    print("        locked to private. Apply once you have uploads to show.")
    return 0


def cmd_analytics(args) -> int:
    from ..analytics.fetch import snapshot
    from ..analytics.feedback import compute
    ctx = _context(args)
    n = snapshot(ctx.db, ctx.youtube)
    print(f"  snapshotted {n} video(s)")
    features = compute(ctx.db)
    if not features:
        print("  not enough published history to compute a strategy yet")
        return 0
    for f in sorted(features, key=lambda x: -x.shrunk)[:12]:
        print(f"  {f.verdict:<7} {f.feature:<16} {f.value:<14} n={f.n:<3} "
              f"shrunk={f.shrunk:.3f}")
    return 0


def cmd_dashboard(args) -> int:
    import uvicorn
    from ..dashboard.app import create_app
    ctx = _context(args)
    host = args.host or ctx.cfg.get("dashboard.host", "127.0.0.1")
    port = args.port or int(ctx.cfg.get("dashboard.port", 8420))
    print(f"  dashboard on http://{host}:{port}")
    uvicorn.run(create_app(ctx), host=host, port=port, log_level="warning")
    return 0


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    from .logging import setup_logging

    ap = argparse.ArgumentParser(prog="asa", description="Animal Story Automation")
    ap.add_argument("--log-level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check the environment is ready").set_defaults(fn=cmd_doctor)

    b = sub.add_parser("bench", help="measure render and TTS speed on this machine")
    b.add_argument("--frames", type=int, default=24)
    b.add_argument("--tts", action="store_true")
    b.set_defaults(fn=cmd_bench)

    c = sub.add_parser("character", help="character tools")
    csub = c.add_subparsers(dest="sub", required=True)
    cn = csub.add_parser("new", help="build a procedural puppet")
    cn.add_argument("character_id")
    cn.set_defaults(fn=cmd_character_new)

    a = sub.add_parser("assets", help="licence ledger")
    asub = a.add_subparsers(dest="sub", required=True)
    aa = asub.add_parser("add", help="register an asset with its licence")
    aa.add_argument("path")
    aa.add_argument("--kind", required=True,
                    choices=["music", "sfx", "font", "background", "character_layer"])
    aa.add_argument("--source", required=True)
    aa.add_argument("--license", required=True)
    aa.add_argument("--attribution", default=None)
    aa.add_argument("--url", default=None)
    aa.set_defaults(fn=cmd_assets_add)
    asub.add_parser("audit", help="fail if any asset is not cleared").set_defaults(fn=cmd_assets_audit)
    asub.add_parser("scan", help="find files on disk with no licence row").set_defaults(fn=cmd_assets_scan)

    r = sub.add_parser("research", help="collect and score topics")
    r.set_defaults(fn=cmd_research)

    j = sub.add_parser("job", help="job management")
    jsub = j.add_subparsers(dest="sub", required=True)
    jn = jsub.add_parser("new", help="queue a new episode")
    jn.add_argument("--topic", default=None, help="free-text seed; skips topic selection")
    jn.add_argument("--topic-id", type=int, default=None)
    jn.add_argument("--format", default="long", choices=["long", "short"])
    jn.set_defaults(fn=cmd_job_new)
    jr = jsub.add_parser("retry", help="reset a failed job to the stage that broke")
    jr.add_argument("job_id", type=int)
    jr.set_defaults(fn=cmd_job_retry)

    rn = sub.add_parser("run", help="advance jobs through the pipeline")
    rn.add_argument("job_id", nargs="?", type=int, default=None)
    rn.add_argument("--once", action="store_true", help="run a single stage")
    rn.add_argument("--max-jobs", type=int, default=1)
    rn.set_defaults(fn=cmd_run)

    js = sub.add_parser("jobs", help="list jobs")
    js.add_argument("--limit", type=int, default=20)
    js.set_defaults(fn=cmd_jobs)

    ap_ = sub.add_parser("approve", help="approve a job for upload")
    ap_.add_argument("job_id", type=int)
    ap_.add_argument("--who", default="human")
    ap_.set_defaults(fn=cmd_approve)

    rj = sub.add_parser("reject", help="reject a job")
    rj.add_argument("job_id", type=int)
    rj.add_argument("--reason", default="rejected by reviewer")
    rj.set_defaults(fn=cmd_reject)

    y = sub.add_parser("youtube", help="YouTube account")
    ysub = y.add_subparsers(dest="sub", required=True)
    ya = ysub.add_parser("auth", help="one-time OAuth (needs a browser)")
    ya.add_argument("--port", type=int, default=0)
    ya.set_defaults(fn=cmd_youtube_auth)

    an = sub.add_parser("analytics", help="pull performance data and recompute strategy")
    an.set_defaults(fn=cmd_analytics)

    d = sub.add_parser("dashboard", help="run the approval dashboard")
    d.add_argument("--host", default=None)
    d.add_argument("--port", type=int, default=None)
    d.set_defaults(fn=cmd_dashboard)

    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="temporary config override, e.g. --set production.target_minutes=2")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
