# 07 — Accounts to Apply For, and Linux Setup

_Target machine: Kali Linux (Debian-based), x86_64, no GPU. Verified 2026-09-01._

---

## 1. What you actually need — the short list

**Only 3 signups are required to build and run the MVP.** Everything else is optional or manual.

| # | Service | Required? | Approval | Time | Fills |
|---|---|---|---|---|---|
| 1 | **OpenRouter** | **Required** | Instant | 2 min | `OPENROUTER_API_KEY` |
| 2 | **Hugging Face** | **Required** | Instant | 2 min | `HF_TOKEN` |
| 3 | **Google Cloud + YouTube APIs** | **Required to publish** | Instant for keys; **weeks for the audit** | 20 min + wait | `YT_CLIENT_ID`, `YT_CLIENT_SECRET` |
| 4 | Freesound | Recommended | Instant | 3 min | `FREESOUND_API_KEY` |
| 5 | Kaggle | Optional | Instant | 3 min | (manual notebooks) |
| 6 | Reddit | Optional | **Manual, 2–4 weeks, may be rejected** | 15 min + wait | `REDDIT_CLIENT_ID/SECRET` |
| — | Wikipedia Pageviews | **No signup** | — | — | — |
| — | YouTube RSS feeds | **No signup** | — | — | — |
| — | Pixabay music / YouTube Audio Library | **No API exists** — manual download | — | 1 afternoon | `assets/music/` |

**Apply for #3's audit on day one.** It is the only thing here with a multi-week lead time that
you cannot work around, and nothing publishes publicly until it clears.

---

## 2. Step by step

### 1. OpenRouter — all story/scene/metadata/QC reasoning

1. Sign up at **https://openrouter.ai** (Google/GitHub sign-in works; **no card required**).
2. Go to **https://openrouter.ai/keys** → *Create Key*. Name it `asa-prod`.
3. Paste into `config/.env` as `OPENROUTER_API_KEY`.
4. Browse **https://openrouter.ai/models?max_price=0** and copy two exact model IDs ending in
   `:free` — one capable model for `story`, one fast one for `cheap`. Put them in
   `config/config.yaml` under `providers.llm.openrouter_free.models`.

**Limits (verified 2026-09-01):** 20 requests/min on any `:free` model. **50 requests/day** while
lifetime credit purchases are under $10; **1,000/day** after a single one-time $10 purchase, and
it stays raised afterwards.

> A video costs ~8–9 calls, so 50/day ≈ 5 videos/day — well beyond what this laptop can render.
> **You do not need to spend the $10.** It is listed in Tier 3 only as the highest-leverage
> optional upgrade if you ever want headroom.

⚠ The `:free` roster changes. Pin exact IDs in config and re-check them if calls start failing.

### 2. Hugging Face — background plates and character turnaround sheets

1. Sign up at **https://huggingface.co/join** (no card).
2. **https://huggingface.co/settings/tokens** → *Create new token* → type **Read** (Fine-grained
   with "Make calls to Inference Providers" also works). Name it `asa`.
3. Paste into `config/.env` as `HF_TOKEN`.
4. Accept any model gate you plan to use — visit the model page and click through the licence.
   `black-forest-labs/FLUX.1-schnell` is Apache-2.0 and generally ungated.
5. **Check your allowance at https://huggingface.co/settings/billing** and set
   `providers.image.huggingface.monthly_credit_budget` in config to what you actually see.

⚠ **This is the biggest unverified number in the whole design.** HF does not publish the free
inference-credit allowance as a fixed figure and has changed it repeatedly. Measure it in week
one. The image cache is content-addressed and permanent specifically so credits are only ever
spent on genuinely new art (~3–6 images per video, falling as locations get reused).

### 3. Google Cloud — YouTube Data API v3 + YouTube Analytics API

**What `YT_CLIENT_ID` / `YT_CLIENT_SECRET` actually are:** they are not your Google password and
not a channel key. They are the ID and secret of an **OAuth client** — a registration that says
"this piece of software is allowed to ask a Google user for permission." Later, when you run
`--authorize`, *you* consent in a browser and Google hands back a refresh token, and that token
is what actually touches your channel.

For a **Desktop app** client the "secret" is not truly secret — Google documents that native apps
cannot keep it confidential. Keep it out of git anyway, but don't panic about it the way you
would about the refresh token in `data/secrets/yt_token.json`, which is the sensitive one.

⚠ Google reorganised this console during 2025–2026 into the **Google Auth Platform**
(Branding / Audience / Data Access / Clients). Older tutorials describing an "OAuth consent
screen" page will not match what you see. Steps below match the current UI.

**A. Create the project**
1. Go to **https://console.cloud.google.com** and sign in with the Google account that manages
   your YouTube channel.
2. Project dropdown (top bar) → **New Project** → name it `animal-story-automation` → **Create**.
3. Make sure that project is selected in the top bar before continuing.

**B. Enable the two APIs**
4. **APIs & Services → Library** → search **"YouTube Data API v3"** → **Enable**.
5. Library again → search **"YouTube Analytics API"** → **Enable**.

**C. Configure the Auth Platform (one-time wizard)**
6. Left menu → **Google Auth Platform**. A fresh project shows *"Google Auth Platform not
   configured yet"* → click **Get started**.
7. Fill the wizard:
   - **App name**: `Animal Story Automation`
   - **User support email**: your address
   - **Audience**: **External**
   - **Contact information**: your address
   - Agree → **Create**
8. Open the **Audience** tab → under **Test users** → **Add users** → add your own Gmail address.
   Without this, authorization fails with `access_denied`.
9. Open the **Data Access** tab → **Add or remove scopes** → add all three:
   - `https://www.googleapis.com/auth/youtube.upload`
   - `https://www.googleapis.com/auth/youtube.force-ssl`
   - `https://www.googleapis.com/auth/yt-analytics.readonly`

**D. Create the OAuth client — this is where the two values come from**
10. **Google Auth Platform → Clients** → **Create client**.
11. **Application type: Desktop app**. Name it `asa-desktop` → **Create**.
12. A dialog shows **Client ID** and **Client secret**. Copy both now (or **Download JSON** and
    read them out of it). You can reopen them any time from the Clients list.
13. Put them in `config/.env`:
    ```
    YT_CLIENT_ID=123456789012-abcdefghijklmnop.apps.googleusercontent.com
    YT_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxxxxx
    ```
    Then move or delete the downloaded JSON — it is not needed again.

**E. Authorize once, then submit the audit**
14. `python -m asa.publish.oauth --authorize` → browser consent → refresh token written to
    `data/secrets/yt_token.json` (mode 600).
    - You will see an **"unverified app"** warning. That is expected for a test-user app you
      built yourself: *Advanced → Go to Animal Story Automation (unsafe)*.
    - ⚠ Test-user refresh tokens have historically expired after 7 days. Moving the app to
      **In production** on the Audience tab avoids that. **Needs verification** of current
      behaviour before relying on unattended runs.
15. **Submit the audit form now** — linked from
    https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits

**Why the audit matters:** videos uploaded via `videos.insert` from an unaudited project created
after 2020-07-28 are **locked to `private`** and cannot be made public. Until it clears, the
pipeline uploads privately and you flip each video public in YouTube Studio — one click, which
fits the human-approval mode anyway.

**Unaudited quota:** 100 `search.list`/day, 100 `videos.insert`/day, 10,000 units/day for
everything else. The pipeline budgets ~2,454 units/day.

**Also needed on the channel itself (not the API):** verify your YouTube account by phone to
enable custom thumbnails and videos longer than 15 minutes.

**You can defer all of section 3.** Nothing before Phase 9 of the build order touches YouTube.
Build and render videos first; wire up publishing when you have something worth publishing.

### 4. Freesound — sound effects (recommended)

1. Sign up at **https://freesound.org**.
2. Apply for a key at **https://freesound.org/apiv2/apply** — instant, self-service.
3. Paste into `config/.env` as `FREESOUND_API_KEY`.

**Limits (verified):** 60 req/min, 2,000 req/day (30/min, 500/day for write operations).

⚠ Freesound sounds carry **per-sound** licences. The pipeline filters to
`Creative Commons 0` and `Attribution` only, and writes a `licenses` row for every download.
Never lift this filter — CC-BY-NC in a monetised video is a licence violation.

### 5. Kaggle — occasional free GPU for character sheets (optional)

1. Sign up at **https://www.kaggle.com** → Settings → **Phone verify** (required for GPU).
2. No API key needed for the intended use — you open a notebook manually, batch-generate ~10
   character turnaround sheets, and download the PNGs.

**Allowance:** ~30 GPU-hours/week, 12 h max per session, P100 or 2×T4, 20 GB persistent storage.
Kaggle is for interactive/notebook work, **not production workloads** — which is exactly how it's
used here: occasional, attended, bounded batches. It is deliberately **not** in the runtime
provider chain.

### 6. Reddit — theme mining (optional; apply, don't depend on it)

1. **https://www.reddit.com/prefs/apps** → *create another app* → type **script** →
   redirect URI `http://localhost:8080`.
2. ⚠ **Self-service registration closed in late 2025** under Reddit's Responsible Builder Policy.
   Every new OAuth client now goes through a manual approval ticket; reported queues are 2–4
   weeks with a real chance of silent rejection.
3. If approved: 100 QPM per client, averaged over a rolling 10-minute window. Custom User-Agent
   `<platform>:<appid>:<version> (by /u/<username>)` is mandatory — generic/browser UAs get blocked.

Ships **disabled** in `config.example.yaml`. Wikipedia pageviews + YouTube RSS already cover the
research signal; Reddit only adds theme vocabulary.

### 7. No signup needed

- **Wikipedia Pageviews API** — no key, courtesy rate limit. Strongest free trend proxy you have.
- **YouTube channel RSS** — `https://www.youtube.com/feeds/videos.xml?channel_id=<ID>`. No quota
  at all. This is the primary competitor watch, which is why the expensive `search.list` calls
  can be rationed to 4/day.
- **Pollinations.ai** — keyless overflow for backgrounds only. ⚠ Commercial rights are not
  clearly stated, so it is never used for character sheets and never as primary.

### 8. Music — manual, one afternoon, no API

There is no free music API worth depending on. **Pixabay's API serves images and videos only —
there is no music/audio endpoint** (verified 2026-09-01), and the good open music models
(MusicGen) ship **CC-BY-NC weights** that forbid monetised use.

So build the library by hand, once:
1. **YouTube Studio → Audio Library** — free, and explicitly Content-ID-safe. Download 20–30
   tracks across moods (curious, warm, tense, triumphant, sad, comedic).
2. **https://pixabay.com/music/** — Pixabay Content License, no attribution required.
3. **https://incompetech.com** — CC-BY, attribution mandatory.
4. Drop them in `assets/music/<mood>/` — moods are `curious warm tense triumphant sad comedic
   neutral`. Then register each file, which takes seconds:

```bash
asa assets add assets/music/warm/autumn_errand.mp3 \
    --kind music --source yt_audio_library --license YT-AUDIO-LIB

# CC-BY is refused unless you supply the credit line - omitting it voids the licence
asa assets add assets/music/sad/quiet_rain.mp3 --kind music --source incompetech \
    --license CC-BY --attribution '"Quiet Rain" by Kevin MacLeod - incompetech.com - CC BY 4.0' \
    --url https://incompetech.com/...

asa assets scan     # lists any file on disk with no licence row
asa assets audit    # exits non-zero if anything is non-commercial or missing attribution
```

`asa assets audit` is the same query QC runs before every upload, and it **fails closed**: an
unknown licence blocks publication rather than warning. `asa doctor` reports an empty music
library as a hard FAIL for exactly this reason.

---

## 3. Linux setup (Kali / Debian)

### System packages

```bash
sudo apt update && sudo apt install -y \
  ffmpeg sqlite3 espeak-ng libsndfile1 \
  fonts-dejavu-core fonts-liberation2 \
  build-essential pkg-config git curl
```

`ffmpeg 8.1.2` and `sqlite3 3.53.4` are already present on this machine — the rest are new.
`espeak-ng` is needed as a grapheme-to-phoneme fallback for Kokoro/misaki; `libsndfile1` for
audio I/O.

**First Kokoro run downloads two things automatically** (~350 MB total): the Kokoro-82M weights
from Hugging Face, and the spaCy `en_core_web_sm` model that misaki's G2P needs. Budget ~25 s of
one-off init; after that the pipeline is a process-level singleton.

### Python 3.12 (the system 3.14 will not work)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
uv python install 3.12
cd ~/Projects/youtube-automation
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

⚠ The ML/audio stack (`torch`, `onnxruntime`, `kokoro`, `sentence-transformers`) does not
reliably publish 3.14 wheels — you would be compiling from source on 4 cores. Re-check wheel
availability before ever moving the pin.

### CPU-only PyTorch (skip the ~2.5 GB CUDA download)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

There is no CUDA device here, so the default GPU wheels are pure waste. Also set:

```bash
export OMP_NUM_THREADS=6      # leave 2 threads for the desktop
```

### Config and database

```bash
cp config/config.example.yaml config/config.yaml
cp config/.env.example config/.env && chmod 600 config/.env
$EDITOR config/.env
mkdir -p data/secrets && chmod 700 data/secrets
sqlite3 data/asa.db < migrations/001_initial.sql
```

### Systemd user timers (no root, no cron)

```bash
systemctl --user enable --now asa-pipeline.timer asa-research.timer asa-analytics.timer
loginctl enable-linger "$USER"    # timers keep running when you're logged out
```

Units are in [06-MVP-PLAN.md §4](06-MVP-PLAN.md), with `Nice=10` and
`IOSchedulingClass=idle` so rendering yields to you on your own laptop.

### Kali-specific notes

- Kali is Debian-testing-tracked and rolls fast. **Pin the venv, not the system.** `uv` handles
  the interpreter so a Kali upgrade can't break the pipeline.
- Kali runs as a single user by default; `loginctl enable-linger` is still needed for unattended
  timers.
- If you use a hardened/`nftables` setup, the dashboard binds `127.0.0.1:8420` only — leave it
  that way. It has no authentication and must never be exposed.

---

## 4. Copy-paste `.env` checklist

```bash
OPENROUTER_API_KEY=      # https://openrouter.ai/keys                       instant
HF_TOKEN=                # https://huggingface.co/settings/tokens           instant
FREESOUND_API_KEY=       # https://freesound.org/apiv2/apply                instant
YT_CLIENT_ID=            # console.cloud.google.com -> OAuth Desktop app    instant
YT_CLIENT_SECRET=        #   + submit the audit form -> WEEKS
REDDIT_CLIENT_ID=        # optional; manual approval, 2-4 weeks, may fail
REDDIT_CLIENT_SECRET=
```

**Total cost of everything above: $0.00.** No card is required for any of it.

## Sources

- [OpenRouter rate limits](https://openrouter.zendesk.com/hc/en-us/articles/39501163636379-OpenRouter-Rate-Limits-What-You-Need-to-Know)
- [YouTube Data API — Quota and Compliance Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)
- [YouTube Data API — Revision History (unverified-project upload restriction)](https://developers.google.com/youtube/v3/revision_history)
- [Freesound APIv2 Overview](https://freesound.org/docs/api/overview.html)
- [Reddit Data API — terms, rate limits, approval queue](https://prowlo.com/blog/reddit-data-api)
- [Pixabay API — images and videos only](https://pixabay.com/service/about/api/)
- [Manage OAuth Clients — Google Cloud Console Help](https://support.google.com/cloud/answer/15549257)
- [OAuth 2.0 for iOS & Desktop Apps](https://developers.google.com/identity/protocols/oauth2/native-app)
- [Kaggle free GPU allowance](https://aimultiple.com/free-cloud-gpu)
- [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M)
