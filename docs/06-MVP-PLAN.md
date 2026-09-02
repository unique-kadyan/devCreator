# 06 — The Recommended MVP

**Constraint restated:** one person, one 2017-era laptop with no GPU, $0/month, and only
OpenRouter + Hugging Face free tiers for hosted inference.

---

## 1. The MVP in one paragraph

A Python package driven by a SQLite-backed state machine. Research collectors (free RSS,
Wikipedia pageviews, a rationed 4 YouTube searches/day) fill a scored topic table. One
OpenRouter free-tier model writes the story in three structured calls. Characters are **layered
PNG puppets built once and reused forever**, so character consistency is a file-reuse property
rather than a model property. Hugging Face generates only *background plates*, which are cached
by content hash and reused across episodes. Kokoro-82M speaks every line locally on CPU. A
NumPy/Pillow compositor animates the puppets over parallaxed backgrounds at 24 fps, ffmpeg
encodes, and a local FastAPI dashboard shows you the finished video, thumbnail and metadata with
an Approve button. Approved videos upload to YouTube as private via the Data API; you flip them
public until the API audit clears. Analytics come back daily and re-weight the topic scorer.

**Cost: $0/month.** The only optional spend is a **one-time $10** OpenRouter credit purchase that
permanently raises the free-model cap from 50 to 1,000 requests/day — and even that is not needed
at this hardware's throughput.

---

## 2. Why this is the right MVP and not a compromise

The instinct with "AI video automation" is Stable Diffusion → AnimateDiff → SVD. On this machine
that path produces roughly **one scene per several hours** and still drifts character designs
between shots. The puppet approach is not the budget version of that plan — for
dialogue-driven character stories it is **strictly better on three axes at once**:

| | Diffusion-per-scene | Puppet compositing |
|---|---|---|
| Character consistency | Drifts; needs LoRA/IP-Adapter, still imperfect | **Identical pixels, always** |
| Speed on this CPU | ~1–3 h per animated scene | **~100 ms per frame, measured** |
| Cost per scene | 1+ image credits, unbounded | **~0 after the first background** |
| Re-render one line of dialogue | Regenerate everything | Re-synthesise one clip, recomposite |
| Art direction control | Prompt roulette | Deterministic; you own every layer |

It gives up freeform character posing. For moral/comedy/friendship stories built on dialogue and
reaction — your entire brief — that is not the constraint that matters.

---

## 3. Build order (realistic solo timeline)

Each phase ends with something you can actually run. Do not build ahead of the phase you're in.

| Phase | What you build | Done when | Est. |
|---|---|---|---|
| **0. Foundation** | `uv` + Python 3.12 venv, package skeleton, config loader, SQLite migration, `structlog`, `scripts/doctor.py` | `asa doctor` prints all-green | 1 day |
| **1. Vertical slice — one hardcoded story** | Hand-write one story JSON + one hand-cut puppet + one background. Build compositor → ffmpeg → 30-second mp4. | **A 30-second animated clip plays.** This is the moment the project becomes real. | 3–5 days |
| **2. Voice + timeline** | Kokoro integration, per-line synthesis, duration-driven timeline, mixer, loudness normalisation | Clip has narration + dialogue, correctly timed, at −14 LUFS | 2–3 days |
| **3. Story generation** | OpenRouter provider, 3-call story chain, Pydantic schemas, repair loop | A full 7-minute video renders from a topic string | 3–4 days |
| **4. Character factory** | HF turnaround generation, `rembg`, `puppet_builder`, `rig.json`, dashboard rig editor | You can add a new character in <15 min | 4–6 days |
| **5. Backgrounds + cache** | HF image provider, content-addressed cache, parallax slicing, procedural fallback | New locations appear without manual work | 2 days |
| **6. Runner + resume** | State machine, `job_stages`, retry/timeout/quota, leases, error table | Kill the process mid-render; it resumes correctly | 2–3 days |
| **7. Subtitles + thumbnails + metadata** | SRT/VTT, 6 scored thumbnail variants, title scoring with accuracy gate | Full `data/out/<job>/` bundle | 3 days |
| **8. QC + dashboard** | Safety checks, similarity, licence gate, FastAPI+HTMX approval UI | You approve a video from a browser | 3–4 days |
| **9. YouTube publish** | OAuth, resumable upload, thumbnail, captions | First private upload succeeds | 2 days |
| **10. Research** | Collectors + scorer + dedupe + cooldowns | Topics arrive without you typing them | 3–4 days |
| **11. Analytics loop** | Analytics API, snapshots, shrinkage, strategy table | Weights change based on real data | 2–3 days |
| **12. Hardening** | Tests, systemd timers, backups, runbook | Runs unattended for a week | 3 days |

**Total ≈ 6–8 weeks part-time.** Phases 0–3 (~2 weeks) already give you a working
story-to-video generator; everything after that is quality and automation.

**Start the YouTube API audit application during Phase 0.** It is the longest-lead item and it
is entirely out of your control.

---

## 4. Deployment

No Kubernetes, no cloud. This runs on the laptop.

```ini
# ~/.config/systemd/user/asa-pipeline.service
[Unit]
Description=Animal Story Automation — pipeline tick
[Service]
Type=oneshot
WorkingDirectory=%h/Projects/youtube-automation
ExecStart=%h/Projects/youtube-automation/.venv/bin/python -m asa.core.runner tick
Nice=10
IOSchedulingClass=idle
```

```ini
# ~/.config/systemd/user/asa-pipeline.timer
[Unit]
Description=Run the pipeline every 15 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
Persistent=true
[Install]
WantedBy=timers.target
```

```bash
systemctl --user enable --now asa-pipeline.timer asa-research.timer asa-analytics.timer asa-dashboard.service
loginctl enable-linger "$USER"     # so timers run when you're not logged in
```

`Nice=10` + `IOSchedulingClass=idle` matter: this is your daily-driver laptop, and rendering
must yield to you.

**Docker** is provided (`docker/`) for reproducible Python 3.12 + wheels, and for the dashboard.
It is optional; the systemd path is simpler and faster on this hardware.

**Backups** — one line, and it covers everything irreplaceable:
```bash
restic backup data/asa.db assets/characters assets/backgrounds config/config.yaml
```
The puppets and the DB are the crown jewels. Renders are reproducible; puppets are not.

---

## 5. Testing strategy

| Level | What | Tool |
|---|---|---|
| Unit | Scoring maths, timeline arithmetic, viseme mapping, camera easing, quota buckets, licence gate SQL | `pytest` |
| Schema | Every LLM response schema round-trips; malformed JSON triggers exactly one repair | `pytest` + recorded fixtures |
| Provider contract | Each provider satisfies its Protocol; fakes raise `QuotaExhausted`, `ProviderError`, 429 | `pytest` |
| Golden-frame | Composite frame 0/mid/last of a fixed scene, compare to stored PNG within tolerance | `pytest` + `Pillow` |
| Media | Rendered mp4 has expected duration ±100 ms, correct codec/fps, LUFS within ±1 | `ffprobe` in test |
| Resume | Kill the runner at each stage boundary; assert it resumes at exactly the right stage with no duplicate work | integration |
| Policy | Known-bad scripts (trademark, violence, misleading title) must be blocked; known-good must pass | fixture corpus |
| **No network in CI** | All providers replaced by recorded fixtures. Live-API tests are a separate, manually-run marker. | `pytest -m live` |

The golden-frame and resume tests are the two that will actually save you. Write them early.

---

## 6. Cost analysis

### Tier 1 — completely free, local, no account
Kokoro TTS · Piper · ffmpeg · Pillow/NumPy compositor · faster-whisper · sentence-transformers ·
SQLite · FastAPI dashboard · procedural fallback · local music/SFX library · llama.cpp fallback ·
Wikipedia Pageviews · YouTube RSS · systemd

### Tier 2 — free APIs, $0, free account required
| Service | Free allowance (verified 2026-09-01) | Used for | Per-video use |
|---|---|---|---|
| OpenRouter `:free` | 20 RPM; 50/day (<$10 lifetime) or 1,000/day | All reasoning | ~8–9 calls |
| Hugging Face Inference | Monthly credit allowance ⚠ **needs verification** | Backgrounds, character sheets | ~3–6 images |
| Freesound | 60/min, 2,000/day | SFX | ~5 (mostly cache hits) |
| YouTube Data API | 100 searches + 100 uploads/day, 10,000 units | Research + publish | ~2,454 units/day total |
| YouTube Analytics API | Separate quota | Feedback loop | ~5 calls/day |
| Kaggle | ~30 GPU-h/week, manual notebooks | Character-sheet batches | Occasional |

**Monthly cost at 3 videos/week: $0.00.** Electricity for ~10 min of render per video is the
only real cost.

### Tier 3 — optional paid, ranked by value
1. **$10 one-time, OpenRouter** → 50 → 1,000 free requests/day, permanently. Not a subscription.
2. **~$200 used RTX 3060 12 GB** → local FLUX/SDXL, character LoRAs, AnimateDiff, Stable Audio Open. Collapses Tier 2 back into Tier 1 and removes every quota. **The single best upgrade if this works out.**
3. ~$9/mo Hugging Face PRO → bigger inference credit pool.
4. Hosted TTS — only if Kokoro's prosody becomes your quality ceiling.
5. Hosted video-generation APIs — **not recommended**; per-second pricing makes the unit economics of a small channel impossible.

---

## 7. Honest limitations

1. **Throughput: ~18 min of render per 7-minute episode** (measured, 6 workers), plus story/art/voice stages. Comfortably 1 video per evening; plan 2–4/week, which is also the correct editorial pace for original content.
2. **No freeform character action.** Puppets do the gestures in the rig. Complex physical comedy is out of reach without hand-animating a new gesture.
3. **Rigging is manual.** ~10 minutes per new character. This is the deliberate human-in-the-loop point, and it is why the cast is small and recurring.
4. **Kokoro has no emotion control.** Emotion comes from writing, pacing, pauses and facial expressions. It will not match a voice actor.
5. **Free-tier drift is permanent.** Every quota in these docs can change without notice. The provider-chain design is the mitigation, not a guarantee.
6. **HF's free image allowance is unpublished.** ⚠ This is the largest unverified number in the design. Measure it in week one and set `monthly_credit_budget` accordingly.
7. **You cannot publish publicly until the API audit clears.** Weeks, not days. Interim: upload private, flip manually.
8. **MFK likely caps revenue** unless you commit to Option B in [05 §2](05-COMPLIANCE.md).
9. **Music is a curated local library, not generated.** Every good open music model has a non-commercial weights licence. Building the library is a one-off manual afternoon.
10. **Analytics need ~20 videos** before the feedback loop says anything that isn't noise. Bayesian shrinkage keeps it honest until then; it does not create information.
11. **This will not make money quickly, if at all.** The system is designed to make *good* videos cheaply and compliantly. Monetisation depends on YPP thresholds, MFK status, and whether the stories are actually good — which is on the writing, not the pipeline.

---

## 8. Upgrade path

| Trigger | Upgrade |
|---|---|
| You buy a GPU | Local FLUX/SDXL via ComfyUI → drop HF from the image chain; add per-character LoRAs for freeform poses; add Stable Audio Open for music; add AnimateDiff for occasional hero shots |
| Audit clears | Flip `auto_publish` on for videos with zero safety flags; keep the human gate for anything flagged |
| >20 videos published | Analytics loop becomes meaningful; start thumbnail A/B using the stored variants |
| Puppet limits bite | Add gestures to the rig vocabulary, or move a character to a Blender 2.5D rig (same architecture, new compositor backend) |
| You want Shorts | Add a `format: short` path — 9:16 canvas, re-stage, reuse everything else. The compositor is resolution-agnostic by design |
| Second machine | Swap SQLite → Postgres, split the runner into a queue worker. The stage protocol doesn't change |
| Multiple languages | Kokoro has multilingual voices; the scene contract is language-agnostic; subtitles regenerate free |
