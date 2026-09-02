# 01 — Free / Open-Source Tool Comparison

_All figures verified 2026-09-01 against the sources listed at the bottom. Anything I could
not confirm from a primary source is marked **⚠ Needs verification**. Nothing here is called
"free" if it is a time-limited trial or an expiring credit grant._

**Selected stack is marked ✅. Per your instruction the runtime uses only OpenRouter and
Hugging Face for hosted inference, both on free tiers; everything else is local or a free
asset API.**

---

## 1. LLM — story, scenes, metadata, QC reasoning

| Tool | Free? | Free quota (verified 2026-09-01) | API key? | Local? | Commercial use | Licence | Limitations | Rec. |
|---|---|---|---|---|---|---|---|---|
| **OpenRouter `:free` models** | Yes, genuinely | **20 RPM** on any `:free` model; **50 req/day** if lifetime credit purchases < $10, **1,000/day** after a one-time $10 purchase | Yes (free acct) | No | Depends on the underlying model's licence — check per model | Service ToS + per-model | Free model roster churns; a model can vanish. Prompts may be logged. | ✅ **Primary** |
| **Hugging Face Inference Providers (text)** | Yes | Monthly **inference credit** allowance, then hard stop ⚠ **Needs verification** — HF does not publish a fixed number and has changed it repeatedly | Yes (free acct) | No | Per-model licence | Per-model | Cold starts; credit ceiling hits image work first | ✅ **Secondary** |
| **llama.cpp + Qwen3-4B-Instruct GGUF** | Yes | Unlimited | No | **Yes** | Yes (Apache-2.0 model) | MIT engine / Apache-2.0 weights | **~5–10 tok/s on this CPU** → 5–8 min per story | ✅ **Terminal fallback** |
| Google Gemini API free tier | Yes, no card | Conflicting public figures (100–1,500 RPD depending on model and source); AI Studio shows *your* real limits ⚠ **Needs verification** | Yes | No | Generally yes ⚠ verify | Google ToS | Free-tier inputs may be used to improve products | Not used (per your instruction) |
| Groq free tier | Yes, no card | 30 RPM / 6,000 TPM / **14,400 req/day** | Yes | No | Yes ⚠ verify | Service ToS | 6k TPM is the real binder for long outputs | Alternative |
| Cloudflare Workers AI | Yes | **10,000 Neurons/day**, shared across all models | Yes | No | Yes ⚠ verify | Service ToS | Neuron cost varies wildly per model | Alternative |
| Ollama + local model | Yes | Unlimited | No | Yes | Per model | MIT engine | Same CPU ceiling as llama.cpp | Alternative to llama.cpp |

**Model choice on OpenRouter — measured, not assumed.** A live test of 7 `:free` models on
2026-09-01 returned:

| Result | Count | Detail |
|---|---|---|
| HTTP 429 "temporarily rate-limited upstream" | **5 of 7** | Free models sit on shared upstream pools (e.g. Google AI Studio). Availability is minute-to-minute. |
| HTTP 403 | 1 | Model gated despite being listed as free |
| Clean structured JSON in 1.5 s | 1 | `minimax/minimax-m3:free` |

Two design consequences, both now in `config.yaml`:

1. **Fallback must be model-level, not just provider-level.** A single pinned `:free` model id is
   not a dependency you can build on — it is a coin flip. Configure an ordered list per role and
   advance on 429 immediately (a 429 here means "this shared pool is busy", not "you are over
   your quota" — do not burn backoff time on it).
2. **Screen models for structured-output discipline.** `nvidia/nemotron-3.5-lightning:free`
   returned its reasoning preamble instead of the requested bare JSON, which would fail schema
   validation every time. Keep an `avoid_for_structured_output` list; such models are still fine
   for free-text stages.

Pin exact model IDs and treat them as versioned inputs — record the ID in `prompts.model_id` so
you can attribute quality regressions later.

---

## 2. Image generation — character sheets, backgrounds, thumbnail plates

| Tool | Free? | Free quota | API key? | Local? | Commercial | Licence | Limitations | Rec. |
|---|---|---|---|---|---|---|---|---|
| **HF Inference Providers → FLUX.1-schnell** | Yes | Monthly credit window (reset date confirmed via `whoami-v2.periodEnd`); **exact allowance still unpublished** ⚠ | Yes | No | **Yes** | Model: **Apache-2.0** | **`hf-inference` no longer serves this model** — routes via third-party providers (`nscale`, `fal-ai`, `wavespeed`). Free accounts show `canPay:false`, so calls **hard-stop rather than bill** when credits run out | ✅ **Primary — live-tested 5.7 s for 1024×576** |
| HF Inference → Qwen-Image | Yes | same pool | Yes | No | **Yes** | **Apache-2.0** | Heavier, more credits/image | ✅ Alt for text-in-image |
| HF Inference → SDXL 1.0 | Yes | same pool | Yes | No | Yes, with use restrictions | **CreativeML OpenRAIL++-M** | RAIL use-restrictions clause; cheaper per image than FLUX | ✅ Budget option |
| **Pollinations.ai** | Yes, keyless | No published quota; best-effort | **No key** | No | ⚠ **Needs verification** — no clear commercial-rights statement | Unclear | No SLA, no support, rights ambiguity → **never for character sheets, only overflow backgrounds** | ✅ **Overflow only** |
| **`procedural.py` (Pillow)** | Yes | Unlimited | No | **Yes** | Yes | Your code | Stylised gradients/vector shapes, not photoreal | ✅ **Terminal fallback** |
| Local SDXL / FLUX via ComfyUI | Yes | Unlimited | No | Yes | Per model | Apache-2.0 / OpenRAIL++ | **8–20 min per 1024² image on this CPU** → unusable per-scene | ❌ on this box |
| Kaggle Notebooks (P100 / 2×T4) | Yes | **~30 GPU-hours/week**, 12 h/session, 20 GB persistent storage | Account | Cloud GPU | Yes | Kaggle ToS | Explicitly *not for production workloads*; interactive/batch notebooks | ✅ **Offline batches only** (see below) |
| Google Colab free | Yes | Unmetered-but-throttled T4, aggressive idle disconnects | Account | Cloud GPU | Yes | Colab ToS restricts long-running/unattended use | Unreliable for automation | ❌ for the hot path |

**Verified live on 2026-09-01 (this account):** `black-forest-labs/FLUX.1-schnell` via
`InferenceClient(provider="auto")` produced a 1024×576 storybook-style plate in **5.7 s**.
Two things the docs did not predict:

1. **The `hf-inference` provider has deprecated FLUX.1-schnell** (HTTP 410) and the
   OpenAI-compatible `/v1/images/generations` route 404s. Image calls must go through
   `huggingface_hub.InferenceClient`, which routes to a live third-party provider. Direct
   `httpx` calls against a hardcoded router URL will break.
2. **Free accounts report `canPay: false`**, so exceeding the credit window returns an error
   rather than silently billing you. That is the safe failure mode, and it means the provider
   chain's `procedural` fallback is what actually keeps the pipeline running.
3. FLUX.1-schnell renders **garbled text on in-scene signage** — a known weakness. Prompt
   `no signage text` and composite any needed lettering locally with Pillow.

**How Kaggle fits without abusing it:** it is *not* in the runtime chain. You use it manually,
occasionally, in a notebook you sit with, to batch-generate character turnaround sheets when
you introduce new cast members (say, 10 characters at a time, ~20 minutes of GPU). Those PNGs
land in `assets/characters/` and are then reused forever by the local compositor. That is
interactive, bounded, notebook-shaped use — which is what Kaggle is for.

---

## 3. Character consistency techniques

| Technique | Needs GPU? | Consistency | Verdict here |
|---|---|---|---|
| **Layered puppet re-composited from fixed PNGs** | **No** | **Perfect (identical pixels)** | ✅ **The design.** See [00-ARCHITECTURE.md §1](00-ARCHITECTURE.md) |
| Character LoRA (trained per character) | Yes, to train | High but drifts | Upgrade path only |
| IP-Adapter / reference conditioning | Yes, per image | Medium; face drifts across poses | Upgrade path only |
| InstantID / PuLID | Yes | High for *human* faces; weak for stylised animals | ❌ wrong domain |
| Fixed seed + verbose prompt | Yes | Low — the classic failure | ❌ |

---

## 4. Text-to-speech

| Tool | Free? | Quota | Local? | Commercial | Licence | Limitations | Rec. |
|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | Yes | Unlimited | **Yes** | **Yes** | **Apache-2.0** | Preset voices, no cloning; **measured 1.20× realtime on this CPU**, ~24 s one-off init. Needs `espeak-ng` and pulls spaCy `en_core_web_sm` on first run | ✅ **Primary — live-tested** |
| **Piper** | Yes | Unlimited | **Yes** | Engine yes | Engine **GPL-3.0** (`OHF-Voice/piper1-gpl`; the archived `rhasspy/piper` was MIT, read-only since Oct 2025). **Voice licences vary per voice — check each `MODEL_CARD`** | Flatter prosody; blazing fast | ✅ **Fallback** |
| Coqui XTTS-v2 | Weights are not free for this | — | Yes | **No** | **CPML — non-commercial** | Voice cloning is great, licence forbids monetised use | ❌ **Excluded** |
| **edge-tts** | "Free" as in unbilled | — | No | **Likely not** | Wraps Microsoft's Edge Read-Aloud backend; Microsoft's position is that commercial use without an Azure subscription violates their ToS | Great quality, wrong licence posture for a monetised channel | ❌ **Excluded** |
| Chatterbox (Resemble) | Yes | Unlimited | Yes | Yes | MIT ⚠ verify | GPU-preferred; slow on this CPU | Upgrade path |
| espeak-ng | Yes | Unlimited | Yes | Yes | GPL-3.0 | Robotic | Emergency only |

**Voice casting:** Kokoro's 54 preset voices are assigned per character in
`characters.voice_id` and are then immutable — a character's voice is part of its identity,
exactly like its fur colour. Pitch/rate/timbre offsets per character are applied
deterministically post-TTS (`media/tts/voices.py`) to widen the cast beyond 54.

---

## 5. Music and sound effects

| Source | Free? | Quota | Commercial | Licence | Limitations | Rec. |
|---|---|---|---|---|---|---|
| **YouTube Audio Library** | Yes | n/a | Yes | YouTube's own terms; **explicitly Content-ID-safe** | **No API — manual download.** Build the library once. | ✅ **Primary music** |
| **Pixabay music** | Yes | **No music/audio API endpoint** — the Pixabay API serves images and videos only (verified 2026-09-01). Manual download. | Yes | Pixabay Content License — no attribution required | Manual only; ~100 req/60s applies to the image/video API you won't need | ✅ **Secondary music, manual** |
| **Freesound API v2** | Yes | **60 req/min, 2,000 req/day** (30/min, 500/day for write ops) | Depends on each sound | Per-sound: CC0 / CC-BY / CC-BY-NC / Sampling+ | **Must filter to CC0 + CC-BY only** and record attribution | ✅ **Primary SFX** |
| Incompetech (Kevin MacLeod) | Yes | n/a | Yes | **CC-BY** — attribution mandatory | No API; manual | ✅ Supplemental |
| Free Music Archive | Yes | ⚠ API status uncertain | Per track | Mixed CC | Verify each track | Supplemental |
| **MusicGen / AudioCraft** | Weights not free for this | — | **No** | Code MIT, **weights CC-BY-NC 4.0** | Non-commercial weights | ❌ **Excluded** |
| Stable Audio Open | Conditionally | — | Yes if revenue < $1M ⚠ verify current terms | Stability Community License | GPU-hungry; unusable on this CPU anyway | ❌ on this box |

**Rule enforced in code:** nothing enters `assets/music` or `assets/sfx` without a row in the
`licenses` table (`source`, `licence`, `attribution_required`, `attribution_text`, `url`,
`download_date`, `usage_allowed`). `qc/policy.py` fails the job if any asset referenced by the
timeline lacks a licence row or is `CC-BY-NC`/unknown. See [05-COMPLIANCE.md](05-COMPLIANCE.md).

---

## 6. Research data sources

| Source | Free? | Quota | Key? | Limitations | Rec. |
|---|---|---|---|---|---|
| **YouTube Data API — `search.list`** | Yes | **Unaudited projects: 100 `search.list` calls/day**, 100 units each, inside the shared 10,000-unit/day pool | OAuth/API key | Expensive; budget it deliberately | ✅ but rationed |
| **YouTube channel RSS** (`/feeds/videos.xml?channel_id=`) | Yes | No quota | **No** | Titles + publish dates only, no view counts | ✅ **Primary competitor watch** |
| **Wikipedia Pageviews API** | Yes | Generous, courtesy rate limit | **No** | Proxy for topic interest, not search volume | ✅ Strong free trend signal |
| **Reddit API** | Yes | 100 QPM per OAuth client, averaged over a rolling 10-min window — **but self-service registration closed in late 2025**; every new client needs manual approval, reported 2–4 week queues with silent rejections | Yes + approval | Theme mining from r/AnimalsBeingBros, r/aww etc. | ⚠ **Optional — apply, don't depend on it** |
| **pytrends** | Yes | — | No | **Archived 2025-04-17, unmaintained**; breaks on Google backend changes | ❌ **Do not depend on** |
| `trendspy` (community fork) | Yes | — | No | Community-maintained, same fragility class ⚠ verify | ⚠ Optional, must be non-blocking |
| Official Google Trends API | — | — | Yes | Announced 2025-07-24, still **application-gated alpha**; most applicants never get in | ❌ Not available |

**Design consequence:** no research collector may block the pipeline. Each runs on its own
schedule, writes into `research_topics`, and the topic-selection stage works from whatever is
in the table. A dead collector degrades scoring quality; it does not stop production.

---

## 7. Video, audio and subtitle toolchain (all local, all free)

| Tool | Licence | Role | Notes |
|---|---|---|---|
| **ffmpeg 8.1.2** (already installed) | LGPL/GPL depending on build; x264 makes it GPL | Encode, concat, filtergraphs, loudness | GPL only matters if you *redistribute* binaries; irrelevant for internal use |
| **Pillow + NumPy** | MIT-CMU / BSD | Per-frame puppet compositing | The animation engine |
| **MoviePy** | MIT | High-level edits where convenient | Slower than raw ffmpeg — use sparingly |
| **faster-whisper** (CTranslate2) | MIT | Word-level caption timing | `base`/`small` on CPU; only needed for karaoke captions |
| **pysubs2 / srt** | MIT | SRT/VTT/ASS generation | |
| **rembg** (u2net) | MIT (model weights: check per-model) | Background removal when cutting puppets | One-time, offline |
| **sentence-transformers** `all-MiniLM-L6-v2` | Apache-2.0 | Story similarity / dedupe | Fast on CPU |
| **SQLite 3.53** (installed) | Public domain | State | WAL mode |
| **FastAPI + Jinja2 + HTMX** | MIT / BSD / BSD | Approval dashboard | No JS build step |
| **Blender** | GPL | *Not used in MVP* | Only if you later move to 3D |

---

## 8. Tiering

### Tier 1 — Completely free / local, no account, no quota
Kokoro-82M TTS · Piper · ffmpeg · Pillow/NumPy compositor · faster-whisper · sentence-transformers ·
SQLite · FastAPI dashboard · procedural background fallback · local music/SFX library ·
llama.cpp fallback LLM · Wikipedia Pageviews · YouTube RSS

### Tier 2 — Free APIs (free account required, real quotas, $0)
**OpenRouter `:free`** (20 RPM; 50 or 1,000 RPD) · **Hugging Face Inference** (monthly credit
allowance ⚠) · **Freesound** (60/min, 2,000/day) · **YouTube Data API** (100 uploads/day and
100 searches/day unaudited, 10,000-unit pool) · **YouTube Analytics API** · Pixabay ·
Pollinations (keyless overflow) · Kaggle GPU (manual batches)

### Tier 3 — Optional paid upgrades, in order of value per dollar
1. **One-time $10 OpenRouter credit purchase** → lifts free-model cap from 50 to 1,000 req/day, permanently. Not a subscription. Highest leverage item on this list by a wide margin.
2. **A used GPU (RTX 3060 12 GB, ~$200 used)** → unlocks local SDXL/FLUX, LoRAs, AnimateDiff, Stable Audio Open, and removes every image quota permanently. Turns Tier 2 back into Tier 1.
3. Hugging Face PRO (~$9/mo) → larger inference credit pool.
4. ElevenLabs / hosted TTS → only if Kokoro's prosody becomes the quality ceiling.
5. Hosted video-gen APIs (Runway/Kling/etc.) → expensive per second; explicitly **not** recommended for an economically sane channel.

---

## Sources

- [YouTube Data API — Quota and Compliance Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)
- [YouTube Data API — Revision History (unverified-project upload restriction)](https://developers.google.com/youtube/v3/revision_history)
- [YouTube Data API — Getting Started (quota)](https://developers.google.com/youtube/v3/getting-started)
- [OpenRouter Rate Limits](https://openrouter.zendesk.com/hc/en-us/articles/39501163636379-OpenRouter-Rate-Limits-What-You-Need-to-Know)
- [OpenRouter free-tier limits summary](https://klymentiev.com/blog/openrouter-free-tier)
- [Hugging Face free inference credits discussion](https://www.edenai.co/post/top-free-image-generation-tools-apis-and-open-source-models)
- [Reddit Data API terms, rate limits and approval](https://prowlo.com/blog/reddit-data-api)
- [Pixabay API documentation (images and videos only)](https://pixabay.com/service/about/api/)
- [Freesound APIv2 Overview (rate limits)](https://freesound.org/docs/api/overview.html)
- [Kokoro-82M model card (Apache-2.0)](https://huggingface.co/hexgrad/Kokoro-82M)
- [Piper voice licensing discussion](https://github.com/rhasspy/piper/discussions/271)
- [Piper licensing / OHF-Voice fork summary](https://www.cekura.ai/discover/piper-tts)
- [Coqui XTTS CPML non-commercial](https://www.promptquorum.com/power-local-llm/local-tts-voice-cloning-piper-coqui-xtts)
- [Microsoft Learn — Edge TTS commercial use answer](https://learn.microsoft.com/en-us/answers/questions/2088770/are-opensource-edge-tts-free-for-commercial-use)
- [AudioCraft weights CC-BY-NC issue](https://github.com/facebookresearch/audiocraft/issues/198)
- [Open-source image model licences (FLUX schnell / Qwen-Image / SDXL)](https://localaimaster.com/blog/best-local-image-models-compared)
- [pytrends archived / Trends API alpha status](https://scrapebadger.com/blog/does-google-trends-have-an-api-what-to-use-in-2026)
- [pytrends repository (archived)](https://github.com/GeneralMills/pytrends)
- [Kaggle free GPU allowance](https://aimultiple.com/free-cloud-gpu)
- [Cloudflare Workers AI free Neurons](https://costbench.com/software/llm-api-providers/cloudflare-workers-ai/free-plan/)
- [Groq free tier limits](https://tokenmix.ai/blog/groq-free-tier-limits-2026)
- [Gemini API rate limits (official)](https://ai.google.dev/gemini-api/docs/rate-limits)
- [YouTube AI disclosure policy summary](https://minimatters.com/youtube-altered-or-synthetic-content-disclosure/)
- [Royalty-free music sources for monetised YouTube](https://www.foximusic.com/blog/7-best-free-royalty-free-music-sites/)
