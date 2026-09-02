# 00 — System Architecture

_Last verified: 2026-09-01. Every external quota/licence claim in these docs carries a
verification date and, where I could not confirm it from a primary source, an explicit
**⚠ Needs verification** marker._

---

## 0. The constraint that drives every decision

This design is written for the machine it will run on:

| Property | Value | Consequence |
|---|---|---|
| CPU | Intel i7-8550U, 4 cores / 8 threads, 1.8 GHz base | ~2017 ultrabook class. Slow. |
| GPU | Intel UHD 620 (integrated), **no CUDA device** | **Local diffusion is not viable.** |
| RAM | 31 GiB | Plenty. Not the bottleneck. |
| Disk | 278 GiB free | Fine. Budget ~2 GiB per finished video incl. intermediates. |
| ffmpeg | 8.1.2 (system) | Good — modern filters available. |
| Python | 3.14.6 (system) | **Too new for the ML stack.** See §6. |

### What "no GPU" actually costs

Rough, order-of-magnitude figures for this CPU (measure them yourself with
`scripts/bench.py` before trusting them):

| Workload | On a T4/3060 | On this box | Verdict |
|---|---|---|---|
| SDXL 1024², 30 steps | ~4 s | **~8–20 min** | Unusable per-scene |
| SD 1.5 512², 20 steps | ~1 s | ~1.5–4 min | Batch-only |
| AnimateDiff, 16 frames | ~30 s | **~1–3 h** | Not viable |
| Stable Video Diffusion, 25 frames | ~40 s | **hours** | Not viable |
| Kokoro-82M TTS | ~30× realtime | **~1–3× realtime** | ✅ Viable |
| whisper.cpp `base` transcription | ~20× RT | ~2–4× RT | ✅ Viable |
| x264 1080p24 encode, veryfast | — | ~1.5–3× RT | ✅ Viable |
| Pillow layer compositing, 9 layers + 3 parallax planes @1080p | — | **~100 ms/frame (6 workers, measured)** | ✅ Viable |

So the architecture splits along a hard line:

- **Anything diffusion-shaped → free cloud API, or a Kaggle GPU batch, and cached forever.**
- **Anything deterministic (audio, compositing, encoding, alignment, DB, scoring) → local CPU.**

---

## 1. The central design idea: characters are *assets*, not *prompts*

The hardest requirement in the brief is **visual character consistency across scenes and
across videos**. The usual answers — LoRA training, IP-Adapter, InstantID, reference-image
conditioning — all need a GPU on the hot path and all of them still drift.

This design sidesteps the problem entirely:

> **Generate each character exactly once, as a layered 2D puppet. Then never generate that
> character again — composite the same PNG layers into every scene of every video.**

```
                 ONE TIME, per character (on Kaggle GPU or free image API)
                 ┌──────────────────────────────────────────────┐
   character  →  │ turnaround sheet → background removal →      │  →  puppet/
   spec (DB)     │ layer cutting → rig definition (JSON)        │      milo_fox/
                 └──────────────────────────────────────────────┘      ├ body.png
                                                                       ├ head.png
                                                                       ├ eyes_{open,half,closed}.png
                                                                       ├ mouth_{A,E,I,O,U,M,rest}.png
                                                                       ├ arm_l.png / arm_r.png
                                                                       └ rig.json
                 EVERY SCENE, forever after (local CPU, deterministic)
                 ┌──────────────────────────────────────────────┐
   scene plan →  │ background (generated per scene) +           │  →  scene_007.mp4
                 │ puppet layers transformed per frame          │
                 └──────────────────────────────────────────────┘
```

Consequences — all of them good:

- **Consistency becomes 100%, by construction.** Milo's face is literally the same pixels in
  episode 1 scene 3 and episode 40 scene 12. No model can beat file reuse.
- **Marginal cost per scene ≈ 1 background image.** Backgrounds have no consistency
  requirement, so any free image API is fine, and a location can be cached and reused.
- **It runs on a CPU.** Compositing is Pillow. Measured: **0.39× realtime** at 1080p24 on this laptop, i.e. ~18 min to render a 7-minute episode.
- **It degrades gracefully.** If every image API is down, you still ship — you reuse cached
  backgrounds, or fall back to a procedurally-drawn flat-colour environment.
- **It is a real animation technique**, not a hack. Cutout/puppet animation is how *South
  Park*, *Paper Mario* cutscenes, and most Toon Boom Harmony TV work is built.

The trade-off you accept: characters cannot do arbitrary poses. You get a shot vocabulary
(idle, talk, walk-cycle, react, point, run, sit) rather than freeform action. For dialogue-
and-narration-driven moral stories — exactly the brief — that is enough. §5 of
[02-PIPELINE.md](02-PIPELINE.md) covers how far this stretches and where it breaks.

---

## 2. Layer diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATION            systemd timers → asa.core.runner (DB state machine)  │
└────────────────────────────────────────────────────────────────────────────────┘
        │                                                              ▲
        ▼                                                              │
┌────────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE STAGES (each: idempotent, resumable, own retry/timeout policy)       │
│                                                                                │
│  research → select → story → cast → scenes → art → voice → sound → animate →   │
│  assemble → subtitle → thumbnail → metadata → qc → APPROVE → publish → learn   │
└────────────────────────────────────────────────────────────────────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────────────┐
│ PROVIDER LAYER   │  │ ASSET LAYER         │  │ STATE LAYER                  │
│ (swappable)      │  │                     │  │                              │
│ • LLMProvider    │  │ assets/characters/  │  │ SQLite (WAL)                 │
│ • ImageProvider  │  │ assets/backgrounds/ │  │ • jobs + job_stages          │
│ • TTSProvider    │  │ assets/sfx/         │  │ • characters / stories /     │
│ • MusicProvider  │  │ assets/music/       │  │   scenes / assets / licenses │
│ • SFXProvider    │  │ data/cache/  (hash- │  │ • research_topics            │
│ Each: quota-     │  │   keyed, permanent) │  │ • youtube_uploads / analytics│
│ aware, retrying, │  │ data/work/<job>/    │  │ • prompts / errors           │
│ falling back     │  │   (scratch)         │  │                              │
└──────────────────┘  └─────────────────────┘  └──────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  HUMAN GATE      FastAPI + HTMX dashboard on 127.0.0.1:8420                    │
│                  preview · script · thumbnail · metadata · flags · Approve     │
└────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  PUBLISH         YouTube Data API v3 (upload/thumbnail/captions)              │
│  LEARN           YouTube Analytics API → feature attribution → strategy table │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Orchestration: what I recommend and why

**Recommendation: plain Python + SQLite-backed state machine, triggered by systemd timers.
No n8n, no Airflow, no Celery, no Redis.**

The job model is: *one video = one row in `jobs`, which advances through an ordered list of
stages, each of which is a pure function of (job state, assets on disk) and is safe to
re-run.* Resume-after-crash is then "look up `jobs.stage`, run the next one" — about 200
lines in [`src/asa/core/runner.py`](../src/asa/core/runner.py).

| Option | Verdict for this project |
|---|---|
| **Python + SQLite state machine + systemd timer** | ✅ **Recommended.** Zero infra, trivially resumable, everything in git, debuggable with `sqlite3` and a stack trace. Handles multi-minute steps naturally. |
| n8n (self-hosted) | ❌ Adds a container + Postgres/SQLite + a GUI. Workflow logic lives in JSON blobs that diff badly and can't be unit-tested. Its retry/branch model is weak for 40-minute render steps. Good for webhook glue; wrong for a media pipeline. |
| Apache Airflow / Prefect | ❌ Massively over-scaled. Airflow's scheduler alone would eat a meaningful share of 4 cores. |
| Celery + Redis/RabbitMQ | ❌ Buys distributed workers you don't have and don't want. SQLite + a `FOR UPDATE`-style claim is enough for concurrency 1–2. |
| Bare cron + shell | ❌ No state, no resume, no retry accounting. This is what you're replacing. |
| **Temporal / restate** | ❌ Correct semantics, absurd operational weight for one laptop. |

**Concurrency policy:** exactly **one** render-heavy job at a time (`jobs.lease`), because
x264 and Pillow will saturate all 8 threads. Network-bound stages (research, image fetch,
metadata) may run at concurrency 3–4 and are gated per-provider by a token-bucket in
`core/quota.py`.

**Docker:** optional and *not* on the critical path. Use it for (a) pinning Python 3.12 +
the ML wheels reproducibly, and (b) running the dashboard. Do **not** containerise ffmpeg —
the system build is already current and container overhead on this CPU is wasted.

---

## 4. Provider abstraction — OpenRouter + Hugging Face, free tiers only

Every external capability sits behind a small protocol so a provider can be replaced without
touching pipeline code. Providers are declared in `config/config.yaml` as an ordered
**fallback chain**; the runtime tries them in order, skipping any whose quota is exhausted.

```python
# src/asa/media/images/base.py  (sketch)
class ImageProvider(Protocol):
    name: str
    def cost_units(self, req: ImageRequest) -> int: ...
    def generate(self, req: ImageRequest) -> ImageResult: ...   # raises QuotaExhausted, ProviderError
```

### Chosen chains

```yaml
llm:
  chain: [openrouter_free, huggingface, local_llamacpp]
image:
  chain: [huggingface, pollinations, procedural]
tts:
  chain: [kokoro_local, piper_local]          # local CPU — no API, no quota
music:
  chain: [local_library]                       # pre-vetted licensed files on disk
sfx:
  chain: [freesound, local_library]            # free API key, CC0/CC-BY only
```

Only **two paid-capable services** are used, and both are used strictly on their free tiers:

**OpenRouter** — all story, scene, metadata and QC reasoning. Use only model IDs ending in
`:free`. Verified limits (2026-09-01): **20 requests/minute on every `:free` model**, and a
daily cap of **50 free-model requests/day** for accounts with under $10 lifetime credit
purchases, rising to **1,000/day** after a single one-time $10 purchase — and it stays
elevated afterwards.

> The 50 RPD floor is the single tightest constraint in the whole system. A full video needs
> ~10–14 LLM calls (§3 of [02-PIPELINE.md](02-PIPELINE.md) budgets them). That is **~3–4
> videos/day** on a never-paid account — comfortably more than this hardware can render
> anyway. So a $0 account is genuinely sufficient. The optional one-time $10 is the single
> highest-leverage upgrade available if you ever want headroom, and it is not a subscription.

**Hugging Face Inference Providers** — all image generation (character turnaround sheets,
per-scene backgrounds, thumbnail plates). Free accounts get a **monthly inference-credit
allowance**, after which calls fail rather than bill. ⚠ **Needs verification:** HF does not
publish the free allowance as a fixed number and it has changed repeatedly; check your own
usage page and set `providers.huggingface.monthly_credit_budget` in config to what you
actually observe. Design accordingly: the image cache is permanent and content-addressed, so
credits are only ever spent on genuinely new art.

### Terminal fallbacks (never let the pipeline hard-stop)

| Capability | Terminal fallback | Quality |
|---|---|---|
| LLM | local `llama.cpp` + Qwen3-4B-Instruct GGUF Q4_K_M | Usable; ~5–10 tok/s on this CPU, so a full story takes ~5–8 min |
| Image | `procedural.py` — Pillow gradient/vector environment plates | Stylised but shippable; characters are puppets so they never depend on this |
| TTS | Piper + a bundled permissively-licensed voice | Slightly flatter than Kokoro, ~instant |
| Music / SFX | `assets/music`, `assets/sfx` local vetted library | Always available; already the primary for music |

### Why these two and not the alternatives

[01-TOOL-COMPARISON.md](01-TOOL-COMPARISON.md) scores the full field. Short version:
OpenRouter is one endpoint in front of many free models, so a model being deprecated is a
one-line config change rather than a rewrite; Hugging Face is the only free image endpoint
that is unambiguously accountable (real ToS, real account, real licence metadata per model),
which matters because these images go into monetised video. Keyless services like
Pollinations stay in the chain as an unreliable free overflow, never as the primary.

## 5. Project structure

I've changed the structure you proposed in one significant way: **I dropped the
`agents/` vs `generators/` split.** That split cuts each domain in half — story generation
would live in `agents/story/` while its prompts, similarity check and schema live elsewhere —
which makes every change a multi-directory edit. Grouping by *domain* instead keeps a feature
in one place.

```
youtube-automation/
├── README.md
├── pyproject.toml                  # installable package; pins Python 3.12
├── config/
│   ├── config.example.yaml         # committed, no secrets
│   ├── config.yaml                 # gitignored, real settings
│   └── .env.example                # committed; real .env is gitignored + chmod 600
├── migrations/
│   └── 001_initial.sql             # full schema (see docs/03-DATABASE.md)
├── prompts/                        # versioned prompt templates, hashed into DB
│   ├── story/ scene/ metadata/ thumbnail/ qc/
├── src/asa/
│   ├── core/          config.py  db.py  runner.py  stages.py  retry.py
│   │                  quota.py   logging.py  errors.py  notify.py
│   ├── research/      collectors/{youtube,trends,reddit,wikipedia,rss}.py
│   │                  scoring.py  dedupe.py
│   ├── story/         generator.py  schema.py  similarity.py  library.py
│   ├── characters/    registry.py  puppet_builder.py  rig.py  casting.py
│   ├── scenes/        planner.py  shot_grammar.py
│   ├── media/
│   │   ├── images/    base.py  cloudflare.py  pollinations.py  huggingface.py
│   │   │              local_sd.py  procedural.py  cache.py
│   │   ├── animation/ compositor.py  camera.py  parallax.py  lipsync.py
│   │   │              shots.py  transitions.py
│   │   ├── tts/       base.py  kokoro.py  piper.py  voices.py
│   │   ├── audio/     sfx.py  music.py  mixer.py  ducking.py  loudness.py
│   │   └── subtitles/ builder.py  align.py  styles.py
│   ├── assemble/      timeline.py  render.py  thumbnail.py  intro_outro.py
│   ├── qc/            safety.py  similarity.py  technical.py  policy.py
│   ├── publish/       metadata.py  oauth.py  youtube.py  scheduler.py
│   ├── analytics/     collect.py  analyze.py  feedback.py
│   └── dashboard/     app.py  templates/  static/
├── assets/
│   ├── characters/<char_id>/       # the puppets — the crown jewels, back these up
│   ├── backgrounds/<location_id>/  # reusable, cached
│   ├── sfx/  music/  fonts/  brand/
├── data/
│   ├── asa.db                      # SQLite (WAL)
│   ├── cache/<sha256>.png          # content-addressed generation cache, permanent
│   ├── work/<job_id>/              # scratch, deletable
│   └── out/<job_id>/               # final mp4 + thumb + srt + metadata.json
├── docker/  Dockerfile  compose.yaml
├── scripts/ bench.py  init_db.py  new_character.py  doctor.py
├── tests/
└── docs/    00-ARCHITECTURE.md … 06-MVP-PLAN.md
```

---

## 6. Python version — a real, immediate blocker

The system Python here is **3.14.6**. As of the last check, the ML/audio stack this project
needs (`torch`, `onnxruntime`, `kokoro`/`misaki`, `sentence-transformers`, `numpy` pinned
builds) does **not** reliably publish 3.14 wheels; you would be building from source on a
4-core laptop.

**Do this first, before anything else:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv: fast, handles interpreters
uv python install 3.12
uv venv --python 3.12 .venv
```

Everything in these docs assumes Python 3.12 in `.venv`. ⚠ Re-check 3.13/3.14 wheel
availability before pinning — this moves fast.

---

## 7. Where the risk actually is

Ranked by how likely it is to kill the project, not by how interesting it is:

1. **YouTube API audit gate.** Uploads from an unaudited API project are locked to *private*.
   This is a manual, human-reviewed application. See [05-COMPLIANCE.md](05-COMPLIANCE.md) §1.
2. **"Made for Kids" classification.** Talking-animal cartoons with moral lessons are very
   likely MFK, which disables personalised ads (the bulk of revenue), comments, end screens
   and notifications. This is a *strategy* decision, not a technical one. §2 of the same doc.
3. **Inauthentic / mass-produced content policy.** The monetisation bar explicitly targets
   templated, repetitive, low-effort output. The whole point of §1's puppet system is that it
   raises quality per video rather than volume.
4. **Free-tier drift.** Every quota in [01-TOOL-COMPARISON.md](01-TOOL-COMPARISON.md) can
   change without notice. The provider chain + `core/quota.py` exist so that a single
   provider dying is a config edit, not an outage.
5. **Render throughput.** ~1 finished 6–8 minute video per overnight run on this hardware.
   Plan around 2–4 videos/week, not 2/day. That is also the right *editorial* pace.


---

## 12. Cast and species (added during the Phase 4 build)

29 species share **one rig**. Identical layer keys, identical anchors, identical viseme and
eye-state names — so `compositor.py` contains no species logic at all and never will. What
varies is silhouette, proportion, palette and a handful of signature features (trunk,
tusks, mane, spines, stripes, spots, four distinct horn shapes, beak-versus-muzzle mouths).

Three consequences worth stating, because each one removes a class of bug:

**The species enum is generated from the profile table.** `story/schema.py` builds its
`Species` literal from `SPECIES_LIST`, so a species the writer is allowed to name is by
construction a species the puppet builder can draw. There is no code path to an
unrenderable cast, and adding an animal is one dictionary entry.

**Scale is physical, and separate from staging.** Each profile carries a `build_scale`
(mouse 0.62 → elephant 1.58). The writer's `staging.scale` is a *dramatic* choice — how
close, how looming — and the two multiply. A mouse pushed forward for emphasis is still
smaller than the elephant behind it, which is the behaviour you want and not the behaviour
you get if you let one number mean both things.

**Casting is suitability-ranked, not free-form.** Every profile declares traits, the roles
it suits, and its typical habitats. `species.suggest()` ranks candidates for a role, and
the outline prompt carries that sheet. This is what stops the writer defaulting to a fox
for everything: a rhino is offered as the immovable gatekeeper, a panther as the silent
threat, an elephant or an owl as the mentor, a mouse as the underdog.

An unknown species falls back to fox geometry rather than raising. A story that names an
animal with no profile is still a story worth making; it just gets generic proportions and
the palette carries the identity.
