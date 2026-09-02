# Animal Story Automation

An end-to-end pipeline that writes, casts, draws, voices, animates, masters, captions,
thumbnails, quality-checks and (with your approval) publishes original animated animal
stories — on a 2017 laptop with no GPU, using only free-tier APIs.

It is built to behave like a small automated animation studio, not a content farm: a
recurring cast with fixed designs and fixed voices, a human approval gate before anything
is published, a licence ledger that fails closed, and no engagement manipulation of any
kind.

---

## The one idea everything rests on

**Characters are layered PNG puppets, drawn once and re-composited forever.**

That single decision is what makes the rest possible on this hardware. Character
consistency stops being a model property — something you beg a diffusion model for and
never quite get — and becomes a file-reuse property, which is exact by construction. It
also makes animation ~100× faster than diffusion on a CPU: compositing a frame is
resizing and alpha-blending a dozen small images.

Backgrounds are the only generated art, and they are cached by prompt hash, so a location
reused in episode 12 costs nothing and looks *identical* to episode 1.

---

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e .
cp config/config.example.yaml config/config.yaml
cp config/.env.example config/.env && chmod 600 config/.env   # add your keys
asa doctor                          # check the machine is ready
asa research                        # collect and score topics
asa job new --topic "a clever fox opens a village bakery"
asa run 1                           # advance the job through every stage
asa dashboard                       # review it at http://127.0.0.1:8420
asa approve 1 --who you
```

`asa doctor` is the fastest way to find out what is missing. It checks system tools,
Python version, config, `.env` permissions, key prefixes, LLM buffer depth, gitignore
coverage, the database, and the asset libraries.

---

## What it does, stage by stage

| Stage | What happens |
|---|---|
| `select_topic` | Highest-scoring unused topic from the research table |
| `story` | 3 structured LLM calls: outline → draft → scene breakdown |
| `art` | One background plate per location, cached and licence-recorded |
| `audio` | Kokoro-82M speaks every line locally; **the audio sets the scene durations** |
| `animate` | Puppets composited over parallaxed plates, 6 processes, streamed to ffmpeg |
| `assemble` | Stream-copy concat, one-pass audio graph, two-pass loudnorm to −14 LUFS |
| `subtitles` | SRT + VTT timed from the synthesised audio, not from the script |
| `thumbnail` | 6 variants from assets already owned, scored at feed size |
| `metadata` | Titles generated then scored in code, with a hard anti-clickbait gate |
| `qc` | 16 mechanical checks; any failure blocks publication |
| `approval` | Stops and waits for a human. Always, until you turn that off deliberately |
| `upload` | YouTube Data API, private by default |

Everything is resumable. Each stage's output is durable before the state advances, so a
crash at the metadata stage does not re-render eighteen minutes of video.

---

## Reliability: the model buffer

Free LLM endpoints are unreliable — in live testing, 5 of 7 OpenRouter `:free` models
returned 429 inside one minute. So model choice is adaptive rather than configured:

* Every free model is discovered at runtime and scored on **observed** success rate,
  latency, 429s, schema failures and truncations — persisted in SQLite across runs.
* A rate-limited model goes cold and the next candidate is tried immediately.
* Six providers chain behind that (OpenRouter → Groq → Gemini → Cloudflare → HF → local
  llama.cpp); a provider with no key is skipped silently.
* Malformed output gets one repair attempt carrying the validator's own error back.
* Output truncated at the token limit is **salvaged** rather than discarded.

`asa dashboard` → Models shows what each model is actually doing for you.

---

## Cast

29 species, each with its own silhouette, proportions, relative on-screen height and
casting profile — so an elephant reads as a mentor and a mouse reads as an underdog
because of what they *are*, not because of what they say. All share one rig, so the
compositor never knows what animal it is drawing.

`bear boar buffalo bull camel cat crow deer dog elephant fox giraffe goat hedgehog hippo
horse leopard lion monkey mouse otter owl panther rabbit raccoon rhino squirrel tiger wolf`

---

## Cost and limits

**$0/month.** Read `docs/01-TOOL-COMPARISON.md` for every quota, verified with dates, and
`docs/08-BUILD-STATUS.md` for what is measured versus what is still unverified.

Three things this code cannot do for you:

1. **The YouTube API audit.** Until Google approves your project, every API upload is
   locked to `private`. That is Google's gate, not a bug here.
2. **The music library.** You must accept the licences and download the tracks yourself.
   `asa assets scan` fails until you do, on purpose.
3. **Analytics.** It needs published videos before it can tell you anything.

---

## Documentation

| | |
|---|---|
| `docs/00-ARCHITECTURE.md` | Hardware reality, the puppet idea, why not Airflow/n8n/Celery |
| `docs/01-TOOL-COMPARISON.md` | Every option with its verified free quota and licence |
| `docs/02-PIPELINE.md` | All 15 stages, the scene JSON contract, measured performance |
| `docs/03-DATABASE.md` | Schema rationale |
| `docs/04-PROMPTS.md` | Prompt architecture and the per-video call budget |
| `docs/05-COMPLIANCE.md` | Made-for-Kids, synthetic disclosure, attribution, OAuth |
| `docs/06-MVP-PLAN.md` | The 12-phase build order and cost analysis |
| `docs/07-ACCOUNTS-AND-SETUP.md` | Every API to apply for, with links and approval times |
| `docs/08-BUILD-STATUS.md` | **What is actually done, measured, and not yet verified** |

## Tests

```bash
pytest tests/ -q     # 79 tests, no network, no keys, no GPU
```
# devCreator
