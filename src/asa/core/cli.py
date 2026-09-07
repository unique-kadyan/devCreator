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


# ---------------------------------------------------------------- video preflight

def _credit_verdict(probe) -> tuple[str, str, str]:
    """Read a creation probe for what it says about billing.

    422 is the GOOD answer: the route accepted us and rejected only the empty input we sent
    on purpose. 402 is the account being out of credit. 429 is ambiguous and worth saying so
    - an uncredited account is throttled to a trickle, so a throttle here is more often a
    billing problem wearing a rate limit's clothes than it is real congestion.
    """
    if probe.status_code == 402:
        return (BAD, "credit", "insufficient credit - buy some at "
                               "replicate.com/account/billing, then wait a few minutes")
    if probe.status_code == 429:
        return (WARN, "credit", "throttled, could not verify - usually means the account "
                                "has no credit")
    return (OK, "credit", "the account can create predictions")


def _preflight_replicate(cfg) -> bool:
    """Check the configured Replicate models against the live catalogue.

    Worth its own command because the failure it prevents is expensive and late. A model
    slug that has been renamed, or an input mapping that no longer matches the model's
    schema, does not show up until a render is already underway - and by then the episode
    has spent its image budget and part of its video budget getting there.

    It only reads: GET /account and GET /models/{slug}. No prediction is created, so this
    costs nothing to run. Every finding here was a real defect the first time it was run:
    `duration` turned out to be an enum of two values that rejects everything else, and
    SadTalker's `preprocess` defaulted to cropping the composed shot down to a bare head.
    """
    import httpx

    base = "providers.video.replicate"
    token = cfg.secret("REPLICATE_API_TOKEN", required=False)
    print("\nreplicate")
    if not token:
        _p(WARN, "REPLICATE_API_TOKEN", "not set - this provider is skipped")
        return True
    api = "https://api.replicate.com/v1"
    head = {"Authorization": f"Bearer {token}"}
    ok = True
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(f"{api}/account", headers=head)
            if r.status_code != 200:
                return _p(BAD, "auth", f"HTTP {r.status_code} {r.text[:80]}")
            ok &= _p(OK, "auth", f"{r.json().get('type')} {r.json().get('username')!r}")

            # Every model in every chain, not just the primaries. A fallback exists to be
            # used on the day the primary breaks, which is the worst possible day to find
            # out that its mapping was wrong all along - and its mapping is its own, so
            # nothing about the primary passing says anything about it.
            credit_checked = False
            for role, mkey, ikey, xkey, fkey in (
                    ("lipsync", "lipsync_model", "lipsync_inputs", "lipsync_extra",
                     "lipsync_fallbacks"),
                    ("i2v", "i2v_model", "i2v_inputs", "i2v_extra", "i2v_fallbacks")):
                models = []
                if cfg.get(f"{base}.{mkey}"):
                    models.append((role, cfg.get(f"{base}.{mkey}"),
                                   cfg.get(f"{base}.{ikey}") or {},
                                   cfg.get(f"{base}.{xkey}") or {}))
                for n, entry in enumerate(cfg.get(f"{base}.{fkey}") or [], 1):
                    if not (entry or {}).get("model"):
                        ok &= _p(BAD, f"{role} fallback {n}", "no `model` - it is ignored")
                        continue
                    models.append((f"{role} fallback {n}", entry["model"],
                                   entry.get("inputs") or {}, entry.get("extra") or {}))

                for label, slug, sent, extra in models:
                    sent, extra = dict(sent), dict(extra)
                    r = c.get(f"{api}/models/{slug.split(':')[0]}", headers=head)
                    if r.status_code != 200:
                        ok &= _p(BAD, f"{label} model", f"{slug}: HTTP {r.status_code}")
                        continue
                    body = r.json()
                    version = (body.get("latest_version") or {}).get("id") or ""
                    schema = ((body.get("latest_version") or {}).get("openapi_schema") or {})
                    inp = (schema.get("components", {}).get("schemas", {}).get("Input", {}))
                    props = inp.get("properties") or {}
                    # "The slug resolves" and "the slug runs" are different questions, and
                    # only the second one matters. Replicate serves the model route for
                    # OFFICIAL models and 404s it for community ones, which must be run by
                    # version id - so a preflight that stops at GET /models goes green on a
                    # model that fails every prediction. Probe the route with a deliberately
                    # empty input: 404 means community (we resolve the version at render
                    # time), 422 means the route works and only the input was wrong, 429
                    # means it works and we are merely throttled.
                    probe = c.post(f"{api}/models/{slug.split(':')[0]}/predictions",
                                   headers=head, json={"input": {}})
                    route = "model route"
                    if probe.status_code == 404 and version:
                        route = "version id"
                        probe = c.post(f"{api}/predictions", headers=head,
                                       json={"version": version, "input": {}})
                    if probe.status_code in (401, 403):
                        ok &= _p(BAD, f"{label} model",
                                 f"{slug}: not runnable by this account")
                        continue
                    ok &= _p(OK, f"{label} model", f"{slug}  [{route}, v{version[:8]}]")
                    # Credit, checked here because it is invisible everywhere else until a
                    # render is already underway. An account with no credit answers 429 to a
                    # creation attempt and only shows the real 402 once you are under the
                    # rate limit - so a run burns its retries on what looks like transient
                    # throttling and quietly falls back to the local renderer. Asking
                    # directly is the difference between knowing now and finding out after
                    # the art stage; once is enough, since it is a property of the account
                    # and not of any model in these lists.
                    if not credit_checked:
                        credit_checked = True
                        ok &= _p(*_credit_verdict(probe))
                    bad = [f"{k}->{v}" for k, v in sent.items() if v not in props]
                    ok &= _p(OK if not bad else BAD, f"{label} inputs",
                             "all mapped" if not bad else "no such input: " + ", ".join(bad))
                    bad_x = [k for k in extra if k not in props]
                    ok &= _p(OK if not bad_x else BAD, f"{label} extras",
                             "ok" if not bad_x else "no such input: " + ", ".join(bad_x))
                    missing = [k for k in (inp.get("required") or [])
                               if k not in set(sent.values()) | set(extra)]
                    ok &= _p(OK if not missing else BAD, f"{label} required",
                             "all supplied" if not missing
                             else "never sent: " + ", ".join(missing))
    except httpx.HTTPError as e:
        return _p(BAD, "replicate", str(e)[:120])
    return ok


def _preflight_service(cfg, service) -> bool:
    """Check one profile-driven service by probing the route it will really submit to.

    Deliberately NOT modelled on the Replicate checker, because these services do not serve
    a model's input schema over the API and pretending otherwise is how a preflight goes
    green on something that cannot run. What can be established without spending anything is
    the pair that actually breaks: does the key authenticate, and does this slug exist.

    The probe is a submit with an EMPTY input, and 422/400 is the answer we want - it means
    the route accepted our key and rejected only the input we deliberately left blank. The
    validation body that comes back usually names the required fields, which is the closest
    thing to a schema these services offer, so it is printed rather than swallowed: it is
    what tells you `image_url` should have been `image` before a real shot finds out.

    Nothing billable is created. In the unlikely event a service queues an empty job anyway,
    it is cancelled and reported.
    """
    import httpx

    from ..media.video.hosted import clip_url, dig

    base = f"providers.video.{service.name}"
    key = cfg.secret(service.env, required=False)
    print(f"\n{service.name}")
    if not key:
        _p(WARN, service.env, "not set - this provider is skipped")
        return True

    models = []
    for role, mkey, ikey, xkey, fkey in (
            ("lipsync", "lipsync_model", "lipsync_inputs", "lipsync_extra",
             "lipsync_fallbacks"),
            ("i2v", "i2v_model", "i2v_inputs", "i2v_extra", "i2v_fallbacks")):
        if cfg.get(f"{base}.{mkey}"):
            models.append((role, cfg.get(f"{base}.{mkey}"),
                           cfg.get(f"{base}.{ikey}") or {}, cfg.get(f"{base}.{xkey}") or {}))
        for n, entry in enumerate(cfg.get(f"{base}.{fkey}") or [], 1):
            if not (entry or {}).get("model"):
                _p(BAD, f"{role} fallback {n}", "no `model` - it is ignored")
                continue
            models.append((f"{role} fallback {n}", entry["model"],
                           entry.get("inputs") or {}, entry.get("extra") or {}))
    if not models:
        _p(WARN, "models", f"a key is set but {base}.lipsync_model is empty")
        return True

    head = {**service.auth_headers(key), "Content-Type": "application/json"}
    ok = True
    try:
        with httpx.Client(timeout=30.0) as c:
            for label, slug, sent, extra in models:
                # The route and the envelope this service will really be submitted to,
                # built by the profile rather than assembled a second time here. Probing
                # `{base}/{slug}` was right while every service named its model in the path
                # and wrong the moment one named it in the body: DashScope would have been
                # probed at a URL it does not serve and reported as a dead slug.
                r = c.post(service.submit_url(slug), headers=head,
                           json=service.wire_body(slug, {}))
                code = r.status_code
                billing = code == 402 or any(
                    w in r.text.lower() for w in ("insufficient", "credit", "balance",
                                                  "quota", "payment", "locked"))
                if code in (401, 403):
                    # One line, then stop: every remaining slug fails identically and
                    # thirty repetitions of it hides the one thing worth reading.
                    #
                    # But read the BODY before blaming the key. fal answers a spent account
                    # with 403 "User is locked. Reason: Exhausted balance", and this branch
                    # used to report that as "key rejected - check FAL_KEY", which sends you
                    # to verify a credential that was correct all along. Measured 2026-09-05
                    # on a real fal account: the key was valid and the balance was zero.
                    if billing:
                        return _p(BAD, "credit", f"account has no balance (HTTP {code}) - "
                                                 f"{' '.join(r.text.split())[:100]}")
                    return _p(BAD, "auth", f"key rejected (HTTP {code}) - check "
                                           f"{service.env} in config/.env")
                if billing and code >= 400:
                    ok &= _p(BAD, "credit", f"{slug}: allowance spent - {r.text[:90]}")
                    continue
                if code == 404:
                    ok &= _p(BAD, f"{label} model", f"{slug}: no such model on this service")
                    continue
                if code in (400, 422):
                    detail = " ".join(r.text.split())[:110]
                    low = detail.lower()
                    # A service with ONE submit route cannot answer a bad slug with a 404 -
                    # the route exists, the model named in the body does not - so an
                    # unknown DashScope model comes back as a 400 that would otherwise be
                    # reported as "route accepts this key", which is the opposite of true.
                    if "model" in low and any(w in low for w in (
                            "not exist", "not found", "invalid model", "unsupported",
                            "not supported", "unknown model")):
                        ok &= _p(BAD, f"{label} model", f"{slug}: {detail}")
                        continue
                    ok &= _p(OK, f"{label} model", f"{slug}  [route accepts this key]")
                    # The validation complaint IS the schema, as far as these services
                    # expose one. Truncated, because some of them return every field.
                    _p(WARN, f"{label} required", detail or "(no detail returned)")
                    continue
                if code < 300:
                    ident = dig(r.json(), service.id_path) or clip_url(r.json())
                    ok &= _p(WARN, f"{label} model",
                             f"{slug}: the service QUEUED an empty job ({str(ident)[:24]}) "
                             f"- cancel it in the dashboard if it is billable")
                    continue
                ok &= _p(BAD, f"{label} model", f"{slug}: HTTP {code} {r.text[:80]}")
    except httpx.HTTPError as e:
        return _p(BAD, service.name, str(e)[:120])

    # Said once, and said plainly: passing here does NOT mean the field names are right.
    _p(WARN, "input names", "unverified - this service publishes no schema; a wrong name "
                            "shows up as a rejected prediction on the first real shot")
    return ok


def cmd_video_preflight(args) -> int:
    """Check every hosted video provider in the chain, without spending anything.

    Worth its own command because the failure it prevents is expensive and late. A model
    slug that has been renamed, or an input mapping that no longer matches the model's
    schema, does not show up until a render is already underway - and by then the episode
    has spent its image budget and part of its video budget getting there.

    Every provider in `providers.video.chain` is checked, in the order it will be tried, so
    the report reads the way a real episode will: the first service with credit renders, and
    the ones below it are what happens when that runs out. A provider with no key set is a
    WARN rather than a failure - an unconfigured allowance is a choice, not a defect.
    """
    from ..media.video.hosted import SERVICES
    from .config import load_config

    cfg = load_config()
    chain = cfg.get("providers.video.chain", []) or []
    ok = True
    checked = False
    for name in chain:
        if name == "replicate":
            ok &= _preflight_replicate(cfg)
            checked = True
        elif name in SERVICES:
            ok &= _preflight_service(cfg, SERVICES[name])
            checked = True
    if not checked:
        print("\nvideo")
        _p(WARN, "chain", f"no hosted provider in {chain} - every shot renders locally, "
                          f"which means no lip-sync and no gesture")
    return 0 if ok else 1


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
        behind = _pending_migrations(db)
        ok &= _p(OK if not behind else BAD, "migrations",
                 "up to date" if not behind
                 else f"{len(behind)} unapplied ({behind[0]}...) - run `asa db migrate`")
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


# ---------------------------------------------------------------- one command
#
# `asa auto` is the whole pipeline behind a single verb. Everything it does was already
# possible - research, job new, run, approve, run - and the reason it exists anyway is that
# the four-command sequence hides two decisions that are easy to get wrong unattended:
# whether there is a topic worth using (research must run FIRST or `select_topic` fails
# with nothing to select), and what to do when a stage stops for a human rather than
# failing (`asa run` prints "ok approval -> AWAITING_APPROVAL" and exits 0, which reads
# like a finished episode and is not one).
#
# The approval gate is NOT removed here. It is the one thing in this pipeline that stands
# between it and being a content farm, so `auto` stops at it by default and prints the file
# to watch. `--upload` is how an operator states, per run, that they are approving - the
# flag is the approval, and it is recorded under their name.


def _newest_topic_id(db) -> int | None:
    from .db import read
    with read(db) as con:
        row = con.execute("SELECT id FROM research_topics ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def _ensure_topic(ctx, seed: str | None, subject_only: bool = False
                  ) -> tuple[int | None, str]:
    """A topic to make an episode from, collecting some if the table has nothing usable.

    Returns (topic_id, how). A seed typed by a person is stored with source='manual', which
    `stages.generate_story` reads as a binding brief rather than as a suggestion - so
    `asa auto --topic "..."` makes the episode the operator asked for, not one loosely
    inspired by it.
    """
    from ..analytics.feedback import current_strategy
    from ..research.collectors import Candidate, _animal_in, _keywords, collect_all
    from ..research.scoring import (DEFAULT_MIN_SCORE, ingest, near_miss, select_next,
                                    select_subject)
    from .db import tx

    floor = float(ctx.cfg.get("research.min_score", DEFAULT_MIN_SCORE))

    if seed:
        ingest(ctx.db, [Candidate(topic=seed, keywords=_keywords(seed),
                                  primary_animal=_animal_in(seed), source="manual")])
        topic_id = _newest_topic_id(ctx.db)
        with tx(ctx.db) as con:
            con.execute("UPDATE research_topics SET status='used' WHERE id=?", (topic_id,))
        return topic_id, "your brief"

    # A factual episode is an editorial choice, so it gets its own door rather than a thumb
    # on the general scale. MEASURED: even with SUBJECT_BONUS a subject candidate lands
    # around 0.30-0.35 while a strong wildlife story from RSS reaches 0.40, so on a mixed
    # pool subjects compete honestly and honestly lose most days. Rigging the scorer until
    # they win would decide, in arithmetic nobody reads, that this is a science channel now.
    if subject_only:
        if select_subject(ctx.db) is None:
            cands = collect_all(ctx.cfg, quota=ctx.quota)
            stored = ingest(ctx.db, cands, weights=ctx.cfg.get("research.weights"),
                            strategy=current_strategy(ctx.db))
            print(f"  research: collected {len(cands)}, stored {stored}")
        top = select_subject(ctx.db)
        if top is None:
            print("  no unused subject topic. Enable the `subjects` collector in "
                  "research.collectors, or pass --topic.")
            return None, ""
        with tx(ctx.db) as con:
            con.execute("UPDATE research_topics SET status='used' WHERE id=?", (top["id"],))
        return top["id"], f"{top['topic'][:90]} ({top['overall_score']:.3f})"

    if select_next(ctx.db, min_score=floor) is None:
        # Nothing scores above the selection threshold, which is the normal state of a
        # fresh install. Collect before giving up - a first run should produce an episode,
        # not an instruction to run a different command first.
        cands = collect_all(ctx.cfg, quota=ctx.quota)
        stored = ingest(ctx.db, cands, weights=ctx.cfg.get("research.weights"),
                        strategy=current_strategy(ctx.db))
        print(f"  research: collected {len(cands)}, stored {stored}")
    top = select_next(ctx.db, min_score=floor)
    if top is None:
        best = near_miss(ctx.db, min_score=floor)
        if best:
            print(f"  the floor (research.min_score = {floor:.2f}) rejected every one of "
                  f"them.")
            print(f"  best rejected, at {best['overall_score']:.3f}: "
                  f"{best['topic'][:88]}")
        return None, ""
    return None, top["topic"]        # the stage selects and marks it; don't race it here


def _auto_report(ctx, job_id: int) -> None:
    """What was made, where it is, and what is wrong with it."""
    import json as _json

    from .db import read
    from .stages import video_parts

    job = {"id": job_id}
    try:
        meta = ctx.load_metadata(job_id)
    except FileNotFoundError:
        meta = None
    parts = video_parts(ctx, job)

    if meta:
        print(f"\n  title       {meta.title}")
        print(f"  tags        {len(meta.tags)}   hashtags {len(meta.hashtags)}")
    for v in parts:
        label = f"part {v['part']}" if len(parts) > 1 else "video"
        print(f"\n  {label:<11} {v['path']}")
        print(f"  {'duration':<11} {(v['duration_s'] or 0) / 60:.1f} min"
              + (f"   {v['lufs']:.1f} LUFS" if v["lufs"] else ""))
        if v["thumbnail_path"]:
            print(f"  {'thumbnail':<11} {v['thumbnail_path']}")
        if v["srt_path"]:
            print(f"  {'subtitles':<11} {v['srt_path']}")
        report = _json.loads(v["qc_report"]) if v["qc_report"] else {}
        # "warn", not "warning" - qc.checks.Finding.severity is fail | warn | info.
        warns = [f for f in report.get("findings", []) if f.get("severity") == "warn"]
        for f in warns[:6]:
            print(f"  {'qc warn':<11} {f['check']}: {f['message'][:90]}")

    with read(ctx.db) as con:
        row = con.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
    print(f"\n  state       {row['state'] if row else '?'}")


def _pending_migrations(db: Path) -> list[str]:
    """Migration files this database has not applied yet.

    Nothing in the pipeline calls `db.migrate()` - the README applies 001 by hand and every
    later file has been applied the same way - so a schema change lands on disk long before
    it lands in anybody's database. That is survivable when a migration adds a table nobody
    reads yet, and not survivable when it adds a column a stage WRITES: the job fails at
    the story stage with `no such column`, after the research and topic selection it just
    paid for.
    """
    import sqlite3
    from .db import MIGRATIONS
    if not Path(db).exists():
        return [p.name for p in sorted(MIGRATIONS.glob("*.sql"))]
    con = sqlite3.connect(str(db))
    try:
        try:
            done = {r[0] for r in con.execute("SELECT name FROM schema_migrations")}
        except sqlite3.OperationalError:
            done = set()                      # ledger itself predates this; treat as none
    finally:
        con.close()
    return [p.name for p in sorted(MIGRATIONS.glob("*.sql")) if p.name not in done]


def cmd_db_migrate(args) -> int:
    from .db import migrate
    ctx = _context(args)
    pending = _pending_migrations(ctx.db)
    if not pending:
        print("  database is up to date")
        return 0
    print(f"  applying {len(pending)} migration(s) to {ctx.db}")
    for name in migrate(ctx.db):
        print(f"    ok  {name}")
    return 0


def _lipsync_preflight(ctx) -> str:
    """Whether anything in this run can actually move a mouth.

    Worth a check before spending twenty minutes of CPU rather than after, because the
    failure is silent by design: `providers.video` ends in `local`, so a chain with no
    funded account does not error - it renders every shot as a camera move over a still and
    reports success. The episode comes back looking exactly like the pre-hosted output, and
    nothing in the logs says the word "lip-sync" at all.

    No network: this reads keys and configured slugs only, so it costs nothing and cannot
    itself be the thing that fails an unattended run. It therefore cannot know whether an
    allowance is spent - only whether there is a provider to spend one.
    """
    from ..media.video.factory import build_video_chain
    try:
        chain = build_video_chain(ctx.cfg, crf=int(ctx.cfg.get("production.crf", 20)))
    except Exception as e:                                   # noqa: BLE001
        return f"could not build the video chain ({str(e)[:80]})"
    ready = [p.name for p in chain.providers
             if getattr(p, "available", False) and getattr(p, "lipsync_model", "")]
    if ready:
        print(f"  lip-sync:   {', '.join(ready)}")
        return ""
    return ("no hosted video provider in `providers.video.chain` has both a key and a "
            "lipsync_model, so every speaking shot falls to the local renderer - which "
            "moves a camera, not a mouth")


def cmd_auto(args) -> int:
    from .runner import Runner
    ctx = _context(args)
    r = Runner(ctx)

    pending = _pending_migrations(ctx.db)
    if pending:
        # Before anything is spent, not after. A column a stage writes is missing until the
        # migration is applied, and the job would die at `story` having already paid for
        # research and topic selection.
        print(f"  {len(pending)} unapplied migration(s): {', '.join(pending)}")
        print("  run `asa db migrate` first.")
        return 1

    print(f"  style:      {ctx.cfg.get('channel.art_style') or 'photoreal'} / "
          f"{ctx.cfg.get('production.render_mode')}")
    warning = _lipsync_preflight(ctx)
    if warning:
        print(f"  WARNING:    {warning}.")
        print("              `asa video preflight` checks the chain for real.")
        if not args.ignore_no_lipsync:
            print("              Re-run with --ignore-no-lipsync to render anyway.")
            return 1

    topic_id, how = _ensure_topic(ctx, args.topic, subject_only=args.subject)
    if topic_id is None and not how:
        print("  nothing is selectable. Either collect more, lower research.min_score, or")
        print("  give it a seed:  asa auto --topic \"a clever fox opens a village bakery\"")
        return 1
    print(f"  topic:      {how[:100]}")

    job_id = r.create_job(topic_id=topic_id, fmt=args.format)
    if topic_id:
        from .db import tx
        with tx(ctx.db) as con:
            con.execute("UPDATE jobs SET state='TOPIC_SELECTED' WHERE id=?", (job_id,))
    print(f"  job {job_id} created\n")

    def _advance() -> bool:
        ok = True
        for res in r.run(job_id):
            mark = "ok  " if res.ok else "FAIL"
            print(f"  {mark} {res.stage:<14} -> {res.state:<18} {res.seconds:6.1f}s "
                  f"{res.error[:80]}")
            ok = ok and res.ok
        return ok

    if not _advance():
        print(f"\n  job {job_id} stopped. `asa job retry {job_id}` re-enters at the stage "
              f"that broke.")
        _auto_report(ctx, job_id)
        return 1

    _auto_report(ctx, job_id)

    state = r.get(job_id)["state"]
    if state != "AWAITING_APPROVAL":
        # Already terminal (auto_publish was on and the threshold met), or parked on a
        # quota. Either way there is nothing here for --upload to approve.
        return 0 if state in ("UPLOADED", "PUBLISHED") else 1

    if not args.upload:
        print("\n  Watch it, then either:")
        print(f"    asa approve {job_id} --who <you>   &&  asa run {job_id}")
        print(f"    asa reject  {job_id} --reason \"...\"")
        print("  or review it in the dashboard:  asa dashboard")
        return 0

    print(f"\n  --upload given: approving as {args.who!r} and uploading "
          f"({ctx.cfg.get('production.privacy_on_upload', 'private')}).")
    r.approve(job_id, who=args.who)
    if not _advance():
        return 1
    _auto_report(ctx, job_id)
    return 0


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

    v = sub.add_parser("video", help="hosted video providers")
    vsub = v.add_subparsers(dest="sub", required=True)
    vsub.add_parser("preflight",
                    help="check the configured hosted video models and input mappings"
                    ).set_defaults(fn=cmd_video_preflight)

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

    db_ = sub.add_parser("db", help="database maintenance")
    dbsub = db_.add_subparsers(dest="sub", required=True)
    dbsub.add_parser("migrate", help="apply pending schema migrations"
                     ).set_defaults(fn=cmd_db_migrate)

    au = sub.add_parser("auto",
                        help="one command: topic -> story -> video -> QC -> review")
    au.add_argument("--topic", default=None,
                    help="a binding brief for this episode; omit to use researched topics")
    au.add_argument("--subject", action="store_true",
                    help="make this episode a factual one - pick the best topic from the "
                         "`subjects` collector (a discovery, invention, ancient technology "
                         "or text) rather than the highest-scoring topic overall")
    au.add_argument("--format", default="long", choices=["long", "short"])
    au.add_argument("--upload", action="store_true",
                    help="approve as --who and upload (privacy from config; private by "
                         "default). Without this it stops at the approval gate.")
    au.add_argument("--who", default="operator",
                    help="who is approving, recorded against the upload")
    au.add_argument("--ignore-no-lipsync", action="store_true",
                    help="render even when no hosted video provider can lip-sync; every "
                         "speaking shot then gets a camera move over a still")
    au.set_defaults(fn=cmd_auto)

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
