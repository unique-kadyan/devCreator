# 02 — Pipeline Stages

One video = one row in `jobs`. Each stage is **idempotent**, writes its outputs to
`data/work/<job_id>/`, records completion in `job_stages`, and can be re-run safely. The
runner always resumes from the first incomplete stage.

```
RESEARCHED → TOPIC_SELECTED → SCRIPT_GENERATED → CHARACTERS_READY → SCENES_PLANNED
→ ART_READY → AUDIO_READY → SCENES_RENDERED → VIDEO_RENDERED → SUBTITLED
→ THUMBNAILED → METADATA_READY → QC_PASSED → AWAITING_APPROVAL → APPROVED
→ UPLOADED → PUBLISHED → ANALYZED
                                          ↘ FAILED · QC_FLAGGED · REJECTED · QUOTA_BLOCKED
```

---

## 1. Research (`asa.research`) — runs on its own schedule, not per-job

Six independent collectors, each writing into `research_topics`. Any may fail without
blocking anything.

| Collector | Signal produced | Cost |
|---|---|---|
| `youtube.py` | Titles, view counts, publish dates, tags for animal-story queries | 100 units per `search.list`; **max 4 calls/day** budgeted (see below) |
| `rss.py` | New uploads from a watchlist of competitor channels | Free, unlimited |
| `wikipedia.py` | 60-day pageview curves for animal + theme articles | Free |
| `reddit.py` | Top posts/comments from animal-story subreddits → theme vocabulary | Free tier, **but requires manual app approval (2–4 wks)** — ships disabled |
| `seasonal.py` | Static calendar → upcoming hooks (back-to-school, winter, festivals) | Free, local |
| `subjects` | Curated Wikipedia articles on discoveries, inventions, ancient technology and texts → topics an episode can be *about*. Marks `signals["subject"]`, which is what unlocks `SUBJECT_BONUS` in the scorer | Free |

**YouTube quota budget (unaudited project = 100 `search.list`/day, 10,000 units/day shared):**

| Use | Calls/day | Units |
|---|---|---|
| Research search | 4 | 400 |
| `videos.list` enrichment (batched 50 IDs/call) | 4 | 4 |
| Upload (`videos.insert`) | 1 | 1,600 |
| `thumbnails.set` | 1 | 50 |
| `captions.insert` | 1 | 400 |
| Analytics (separate API, separate quota) | — | 0 |
| **Total** | | **~2,454 / 10,000** |

Comfortable headroom. The binding limit is 100 `videos.insert`/day, which is far beyond what
this hardware can render.

### Scoring (`research/scoring.py`)

Each candidate gets nine sub-scores in `[0,1]`, combined into `overall_score` with configurable
weights. **All weights live in `config.yaml`** because the analytics feedback loop (§14) tunes
them.

| Sub-score | How it's computed |
|---|---|
| `trend_score` | Wikipedia pageview slope over 60d, normalised; + Reddit velocity |
| `search_score` | Query-frequency proxy from YouTube autocomplete + result counts |
| `competition_score` | **Inverted**: many recent high-view videos → *lower* score |
| `emotional_score` | LLM rubric, 1 call, batched across 20 candidates |
| `entertainment_score` | LLM rubric, same call |
| `story_score` | LLM rubric: can this become a 5-beat arc with a moral? Same call |
| `thumbnail_score` | LLM rubric: is there a single readable visual conflict? Same call |
| `short_form_score` | Heuristic: hook compressible to <60 s? |
| `long_form_score` | Heuristic: enough beats for 6–10 min? |

Note the LLM rubric costs **one** OpenRouter call per 20 candidates, not one per candidate —
essential under a 50 req/day cap.

`dedupe.py` rejects a candidate whose MiniLM embedding is within `similarity_threshold` (default
0.86) of any topic already in `stories` or `research_topics` with `status IN (used, queued)`.

---

## 2. Topic selection

Highest `overall_score` with `status='new'`, subject to:
- cooldown: same primary animal not used in the last N videos (default 3)
- cooldown: same story archetype not used in the last M videos (default 4)
- `characters` availability: prefers topics castable from existing puppets (free) over topics
  needing a new character (costs HF credits + a manual rigging pass)

This is where "don't be a content spammer" is enforced mechanically.

---

## 3. Story generation (`asa.story`)

**Three LLM calls, not one.** Splitting improves structure adherence dramatically and keeps
each response inside free-model context limits.

| Call | Output | Approx tokens out |
|---|---|---|
| 1. **Outline** | title, hook, audience, genre, moral, setting, 5-beat arc, cast requirements, plus `subject` / `facts` / `period` when the episode is about something real | ~600 |
| 2. **Draft** | full prose narration + dialogue per beat | ~2,000 |
| 3. **Scene breakdown** | strict JSON scene list with the fields below | ~2,500 |

All three are validated against Pydantic schemas (`story/schema.py`); a schema failure triggers
one repair call with the validator error appended, then fails the stage.

**What the outline call is required to establish**, beyond the fields themselves: the story
opens *on* the trouble rather than on the setup that led to it; `logline` states the one
concrete question the opening plants and the ending answers; each beat costs the protagonist
more than the last; there is one turn whose evidence is planted in an earlier beat; and the
stakes are human-sized, because save-the-world stakes cannot be paid off in the target
runtime and read as empty when they are not. The draft call then constrains the opening
specifically - the first thing in `beginning` is a line of dialogue reacting to something
that has already gone wrong, not narration and not a character introducing themselves.
See `prompts/_blocks/retention_bible.md` and `docs/04-PROMPTS.md`.

**When the episode is about something real** the outline also carries `subject`, `facts`
and `period`. `subject` without `facts` is a schema error, and each fact must be a full
sentence rather than a label - the list is both the ceiling the later two calls are held to
and the audit trail a reviewer checks. `period` is consumed by the ART stage, which runs in
a different process, and replaces `channel.region_hint` for anything not set in the present
day. The rules live in `prompts/_blocks/subject_bible.md`; the columns in
`migrations/010_story_subject.sql`.

### Scene object (the contract everything downstream consumes)

```json
{
  "index": 1,
  "duration_hint_s": 5.0,
  "location_id": "forest_village",
  "characters": ["milo_fox", "bea_rabbit"],
  "staging": {"milo_fox": {"x": 0.35, "y": 0.72, "scale": 1.0, "facing": "right"},
              "bea_rabbit": {"x": 0.68, "y": 0.74, "scale": 0.92, "facing": "left"}},
  "action": "Milo discovers a mysterious box half-buried in leaves",
  "narration": "Milo had walked this path a hundred times. He had never seen the box.",
  "dialogue": [{"character": "milo_fox", "line": "That... wasn't there yesterday.",
                "emotion": "curious"}],
  "emotion": "curious",
  "camera": {"move": "push_in", "from": "wide", "to": "medium", "ease": "in_out"},
  "shot": "two_shot",
  "visual_prompt": "forest village clearing at golden hour, fallen leaves, storybook flat-vector style, ...",
  "sfx": ["leaves_rustle", "wood_creak"],
  "music_cue": "curious_light",
  "transition_in": "cut"
}
```

`duration_hint_s` is advisory — actual scene duration is computed from the rendered audio
(§6), never from the LLM's guess.

### Anti-repetition (`story/similarity.py`)

Three checks, all local and free:
1. **Embedding similarity** of the logline against every prior story (MiniLM, cosine).
2. **Beat-signature similarity** — the 5-beat arc reduced to a verb/role skeleton, compared by
   Jaccard on shingles. Catches "same story, different animals".
3. **Archetype counter** — a rolling histogram of `(archetype, primary_animal, moral)` triples;
   the story stage receives the *least-used* combinations as a negative-constraint block in its
   prompt.

Any story above threshold is regenerated once with an explicit "avoid these plots" block, then
rejected.

---

## 4. Casting & character creation (`asa.characters`)

`casting.py` maps each required role to an existing puppet where possible. When a genuinely new
character is needed, the job pauses at `CHARACTERS_READY` with `needs_human: true` and the
dashboard shows a **"New character required"** card. This is deliberate: rigging is the one
step where a few minutes of human attention buys permanent quality.

### Creating a puppet (one-time, per character)

1. `new_character.py` writes a character spec row (species, palette, clothing, accessories,
   personality, voice, art-style token).
2. Generate a **turnaround sheet** — one image, front/three-quarter/side, T-pose-ish, on flat
   background — via HF FLUX.1-schnell, using the shared **style bible** token block so every
   character in the channel matches.
3. `rembg` removes the background; `puppet_builder.py` proposes layer cuts.
4. Human confirms/adjusts cuts in the dashboard's rig editor (drag anchor points). ~5–10 min.
5. Output: layer PNGs + `rig.json` with joint anchors, mouth-shape set, blink set, and
   Z-order.

```json
// assets/characters/milo_fox/rig.json
{
  "character_id": "milo_fox",
  "canvas": [1024, 1024],
  "anchors": {"neck": [512, 360], "shoulder_l": [430, 430], "shoulder_r": [594, 430],
              "hip": [512, 640], "mouth": [512, 300], "eye_l": [472, 268], "eye_r": [552, 268]},
  "z_order": ["arm_far", "leg_far", "body", "leg_near", "head", "eyes", "mouth", "arm_near", "accessory_hood"],
  "visemes": {"rest": "mouth_rest.png", "A": "mouth_A.png", "E": "mouth_E.png",
              "I": "mouth_I.png", "O": "mouth_O.png", "U": "mouth_U.png", "M": "mouth_M.png"},
  "blink": ["eyes_open.png", "eyes_half.png", "eyes_closed.png"],
  "style_hash": "sha256:…"
}
```

`style_hash` is the hash of the style-bible prompt block used, so you can detect a character
generated under an older art style.

---

## 5. Art: backgrounds only (`asa.media.images`)

Characters are never generated at this stage — only **location plates**.

- Key = `sha256(location_id + visual_prompt + style_block + model_id)` → `data/cache/`.
  A repeat location is free forever.
- Each plate is generated in **three parallax layers** where the prompt supports it
  (`far`, `mid`, `near`) by generating one image and depth-slicing it with a simple luminance/
  blur heuristic, or by generating a single flat plate and using a synthetic parallax offset.
  Free, CPU-cheap, and enough for the effect.
- Target 2048×1152 so the camera can push in 1.5× without softening at 1080p.

**Expected credit use:** a 40-scene episode across 8 distinct locations = 8 images, of which
typically 3–5 are cache hits from prior episodes. Realistically **3–6 HF image calls per
video.**

---

## 6. Voice (`asa.media.tts`) — this stage sets the timeline

Ordering matters: **audio is generated before animation**, and the audio determines every
duration.

1. For each scene, synthesise narration and each dialogue line separately (never as one blob) —
   separate files give per-line timing, per-character voices, and easy re-generation.
2. Kokoro-82M, voice from `characters.voice_id`; narrator voice fixed per channel.
3. Per-character deterministic post-processing (`voices.py`): semitone shift, rate scale, mild
   formant tweak via ffmpeg `rubberband`/`asetrate`+`atempo`. This turns 54 preset voices into a
   much larger distinguishable cast.
4. Measure each clip's true duration; write to `audio` table.
5. `scenes.duration_s = sum(clip durations) + pre_pad + post_pad + beat_pauses`.

Emotion is expressed through **pacing**, plus the puppet's expression set — Kokoro has no
emotion token. `voices.py` holds an `EMOTION_RATE` table (sad 0.88 … excited 1.14) that
multiplies the character's base speed, and applies deterministic per-character pitch offsets
via ffmpeg `asetrate`+`atempo`.

**Measured on this machine (2026-09-01):** Kokoro-82M synthesises at **1.20× realtime** on CPU,
with a ~24 s one-off pipeline init (so the pipeline is a process-level singleton). Utterances
are cached by `sha256(voice|speed|pitch|text)`, so editing one line re-synthesises one clip
rather than the episode.

---

## 7. Sound & music (`asa.media.audio`)

- **SFX:** `sfx.py` resolves each scene's `sfx` tags against the local library first; on a miss
  it queries Freesound with `filter=license:("Creative Commons 0" OR "Attribution")`,
  downloads, normalises to −20 LUFS, caches into `assets/sfx/`, and **writes a `licenses` row**.
- **Music:** selected from `assets/music/` by mood tag + target duration; looped and crossfaded
  to length. No generation — the licensing risk of NC-weighted music models isn't worth it.
- **Mix (`mixer.py`):** voice bus at −16 LUFS target, music ducked −12 dB under speech via
  ffmpeg `sidechaincompress`, SFX at −22 LUFS, final master limited to **−14 LUFS integrated,
  −1 dBTP** (YouTube's normalisation target).

---

## 8. Animation (`asa.media.animation`) — the compositor

Per frame, at 1920×1080 / 24 fps, in NumPy:

```
background far layer  →  camera transform (pan/zoom, parallax factor 0.3)
   mid layer   →  camera transform (parallax factor 0.6)
   near layer  →  camera transform (parallax factor 1.0)
   for each character (in staging Z-order):
       body           → position + scale + subtle idle bob (sine, 0.5 Hz)
       head           → attached at neck anchor + head-turn/tilt easing
       eyes           → blink automaton (Poisson, ~1 blink / 4 s)
       mouth          → viseme from the phoneme track (§8.1)
       arms           → gesture curve for the active shot
   → foreground vignette / particles (leaves, dust, rain) — cheap sprite system
   → camera transform applied last as a single affine warp
```

### 8.1 Lip sync, free and local

Two tiers:
- **Tier A (default):** amplitude-driven. RMS envelope of the character's audio clip →
  mouth-openness → viseme index. Costs nothing, reads convincingly at 24 fps for stylised
  characters.
- **Tier B (better):** run `faster-whisper` with `word_timestamps=True` over the already-known
  text (forced alignment against known text is very accurate), map graphemes → a 7-viseme set.
  ~2–4× realtime on this CPU, so a 7-minute video costs ~2–3 min. Enable when the extra polish
  is worth it.

### 8.2 Shot grammar (`shots.py`)

The LLM chooses from a **closed vocabulary** rather than describing motion freely, so every
value is renderable:

`camera.move ∈ {static, push_in, pull_out, pan_left, pan_right, tilt_up, tilt_down, handheld_drift}`
`shot ∈ {wide, full, medium, close_up, extreme_close_up, two_shot, over_shoulder, insert}`
`gesture ∈ {idle, talk, point, wave, shrug, jump, run_cycle, walk_cycle, sit, react_shock, react_sad, laugh}`
`transition ∈ {cut, dissolve, fade_black, whip_pan, iris}`

Closed vocabularies are why this pipeline is reliable: there is no free-text field that can
produce an unrenderable instruction.

### 8.3 Performance budget — measured, not estimated

Benchmarked on this machine (i7-8550U, 4c/8t, no GPU) rendering 30 s of 1080p24:

| Configuration | ms/frame | Realtime factor |
|---|---|---|
| Single-threaded, LANCZOS backgrounds | 333 | 0.15× |
| **6 workers, BICUBIC backgrounds** | **~100** | **0.39×** |
| …static camera (background cache hits) | 65 | 0.64× |
| …pull_out / drift (cache misses every frame) | 141 | 0.30× |

**The dominant cost is resizing the three parallax planes every frame**, not the puppet. Two
things follow, both implemented:

1. **BICUBIC, not LANCZOS, for backgrounds.** ~2–3× faster and invisible on a moving camera
   over flat-vector art. LANCZOS is retained for the character layers, which are small.
2. **Parallelise across frames, not within them.** Frames are independent; each worker builds
   its own renderer once via a pool initializer, because pickling tens of MB of background
   planes per frame would cost more than the compositing. Workers are pinned to
   `OMP_NUM_THREADS=1` so processes, not threads, provide the parallelism.

**A 7-minute episode renders in ~18 minutes.** Still ~100× faster than diffusion-based
animation on this hardware, but the earlier ~10-minute estimate in this document was wrong —
it under-counted background resampling.

Further speedups available if needed: cache background frames when the camera is static
(already done), drop the world plate from 2688×1512 to 2304×1296, or render at 1600×900 and
upscale on encode.

### 8.4 The cinematic path: shots and a 2.5D performance

Everything above describes the **puppet** renderer. `production.render_mode: cinematic`
takes a different route: the image model generates photoreal frames with the characters
already in them, because a procedural rig cannot reach the look this channel wants -
anthropomorphic animals in real clothes, in Indian classrooms, streets and corridors.

A generated still cannot be re-posed, which raises the question of what to do with it for
the twenty seconds a scene lasts. The first implementation held it under a slow zoom, and
that is what made finished episodes read as a photograph with a voice-over: nobody on
screen was ever the person talking. `production.performance` replaces it with a shot layer
and a warp layer.

**The shot layer (`assemble/shotlist.py`).** A scene is cut into shots, one per run of lines
by the same speaker. Cuts land in the silence between two lines, never on a line's first
frame. Narration gets the wide, full and insert framings of the place and the action;
speaking gets a tight framing of whoever is talking. Shots share generated images through
`image_key`, so a six-line exchange between two characters cuts six times on two
generations, capped by `max_images_per_scene`.

`plan_shots` is a pure function of the scene's dialogue, because it runs twice: once in the
art stage, which has no audio yet and needs to know which pictures to buy, and once in the
animate stage, which has measured durations. `time_shots` is the only half that needs the
audio. A shot's image is addressed by a hash of its `image_key`, so a missing file means
exactly one thing - the art stage has not run - and says so.

**The warp layer (`media/animation/performance.py`).** The same per-character RMS envelope
that drives Tier A lip-sync above drives a 2.5D performance on the still: a muzzle-sized band
is displaced about the mouth, the eye band squashes for a blink, the head drifts, the camera
breathes.

Read that as jaw motion, because that is what it is. **The local renderer does not do
lip-sync and cannot be made to.** Displacing a region of a photograph translates a muzzle; it
cannot part lips or show teeth, and it does not know where the mouth is (below). What it is
for is that a held still should not look frozen.

The whole warp is one primitive: a band redrawn from a source slice offset at its top and
bottom edges is a linear vertical displacement field. A jaw is two of them - one carrying
the chin down, one below it absorbing the motion so the chest does not slide - and a blink
is one with a negative displacement at its lower edge. Both bands read from a copy taken
before either wrote, or the lower one warps the upper one's output a second time and leaves
a seam across the face.

**Where the face is** (`media/animation/face.py`) comes from the prompt, not a detector. A
speaking shot is generated from a composition string that names the head's edges, and the
priors were measured off that output.

This is the layer's weak point, and it has been measured rather than assumed. The framing
word is only a location while the image model obeys it, and FLUX.1-schnell does not reliably
obey it: in one finished episode a shot requested as a close-up of an owl came back as a wide
two-shot of a rooftop, so the "mouth" was forty seconds of skyline. Every prior therefore
carries a low `confidence` that scales the warp down, and the band is bounded on all four
sides inside the head box. The earlier geometry ran from the eyes to below the frame across
the full picture width; its frame-to-frame difference map was an edge map of the whole image,
with the glasses, hoodie, laptop and background bookshelf all moving with the jaw. That is
what "no lip movement, only frame flicking" was describing.

Two rescues were tried and rejected, both empirically:

* **A detector.** YuNet on twelve generated stills missed five, and on one preferred a human
  in the background to the animal filling the frame. A real detector can still be dropped in
  through the `.face.json` sidecar, at full confidence.
* **Mouth-state pairs.** Generate the same shot twice on a fixed seed, once with the mouth
  open and once closed, and cross-fade the mouth region. With FLUX.1-schnell, changing that
  one clause at a fixed seed changed 63% of the pixels: the pair are two different pictures,
  not two expressions of one.

A third rescue is not available at all: **hands and legs.** There is no pose, limb or gesture
concept anywhere on the local path - only a vertical displacement of a photograph - so no
setting makes a character move an arm. It is not a tuning problem.

What the two findings above have in common is that the local renderer is guessing where the
face is. So it now declines to warp a box it cannot vouch for
(`production.performance.min_face_confidence`, default 0.5). A prior rates itself 0.35 and a
sidecar 1.0, so guesses render camera-only. Measured on a finished still: asked for a
close-up of a fox, FLUX returned a seated two-thirds figure with its muzzle in the upper
right; the prior put the muzzle band on the character's lap and the desk behind it, and the
output tracked the voice at 0.995 correlation - on furniture, 10.2% of the frame, x
0.37-0.64 and y 0.61-0.91, nowhere near the muzzle. Confidence scaling made that quieter,
not absent. The floor gates blink and head drift too, since all three read the same
rectangle.

**Real lip-sync - and any gesture at all - is a hosted model** (`media/video/replicate.py`),
which is the seam the shot layer was built for. The default speaking model is
`bytedance/omni-human`: image plus audio to a whole-figure performance, so mouth, head,
shoulders, arms and hands come out of one prediction. A talking-head model such as SadTalker
solves only half of it - with `still_mode` on, where its artefacts live, the body is as
frozen as it is locally, and a character explaining something for forty seconds never moves
an arm. A provider answers per SHOT, not per episode - `accepts()` - because hosted video is
billed by the second: the talking close-ups go to the hosted model, and the establishing
wides go to the local renderer for free. On the measured episode that was 9 shots hosted out
of 27.

**Two axes of fallback, and both are needed.** The provider chain (`providers.video.chain`)
ends at the local renderer; inside the Replicate provider, each mode has a chain of MODELS
(`lipsync_fallbacks`, `i2v_fallbacks`) walked for one shot before the provider gives up.
Without the second, a renamed slug or a prediction that choked on one image costs the shot
its lip-sync and its gesture - everything the hosted call was for - when another hosted model
was right there. The lipsync list is ordered by how much of the performance survives: whole
figure (OmniHuman), whole figure directed by a prompt (Wan-S2V), a talking head with a frozen
body (SadTalker), then the local camera move.

Each entry carries its own `inputs`, `extra` and `durations`, held together in a `ModelSpec`,
because every one of those is a property of the model: OmniHuman takes `image`/`audio` and
rejects anything else, SadTalker calls the same two files `source_image`/`driven_audio` and
needs `preprocess: full`, Kling's `duration` is an enum of two values and Wan has no duration
field at all. A fallback list of bare slugs would inherit the primary's mapping and fail
every prediction it was added to rescue.

`QuotaExhausted` and `AuthError` do not walk the list - every slug on the account fails those
identically, so walking it would spend the shot's wall clock to be told the same thing three
times. Everything else advances, and if the whole list fails the error names every model
tried, because "the hosted provider failed" without saying which three models and how is not
a diagnosis. `asa video preflight` checks every model in every chain for the same reason.

**The provider chain is several free allowances, not one paid account.** Each hosted entry
is a separate service with its own signup credit, and `QuotaExhausted` advancing the chain is
what turns "the allowance ran out" from an episode-wide downgrade into a per-shot hop to the
next account. `AuthError` advances it as well, which it did not used to: aborting the episode
over one revoked key made sense when there was a single hosted provider and the alternative
was a silent downgrade, and stopped making sense the moment there were three.

Everything that is not a service's HTTP dialect lives in `hosted.HostedShotProvider` - the
spend policy, the model chain, the credit latch, the frame-count conform - and Replicate is
one subclass of it. The others are `Service` profiles: fal answers with a `status_url` to
poll and a separate `response_url` to collect the clip from, WaveSpeed wraps everything in
`data` and builds its poll URL from an id, and WaveSpeed will not take a data URI so its
profile names an upload endpoint instead. That is a difference in JSON paths, not behaviour,
so a new service is a table entry rather than a module.

Six services sit in that table now, and the shape of the difference between them is worth
stating once: `fal`, `wavespeed`, `dashscope`, `novita` and `siliconflow` are all serving
Wan, because Wan is open weights and hosts therefore compete on it. So the provider chain
buys separate free allowances and a newer model version, not five different looks. The
version is the part that matters - Novita's `wan2.7-i2v` against everyone else's 2.2 - and
the reason it is not first in the chain is that its audio-driven path is unproven and fails
in the one way nothing downstream can catch: a fluent performance of the wrong words, on a
clip that is never retimed. Prove it on one shot, then promote it.

**The first-party APIs stretched the profile in one direction the resellers did not.** fal
and WaveSpeed name the model in the URL and take our mapped fields as the whole body.
DashScope and Google post to one fixed route, name the model in the body, and want those
fields inside an envelope - so `Service` also carries `submit_path`, `model_field` and
`input_path`, and `wire_body` reshapes the flat mapping on the way out. Two conventions do
the rest of that work without a profile field per case: a **dotted** target name in a mapping
(`duration: parameters.durationSeconds`) is placed at the root of the request rather than in
the input envelope, which is how DashScope's `parameters` and Google's are reached from
config; and `asset_object` turns every data URI in the body into the `{bytesBase64Encoded,
mimeType}` object Google insists on. All of it is inert on a profile that does not ask for
it, which is what keeps the reseller bodies byte-identical to what they were before.

Polling stretched it once more. Most services serve the job AT a URL - handed over, or
built from an id - and SiliconFlow instead POSTs the id to one fixed `/video/status` route,
so `poll_method` and `poll_body_field` exist and the submitted id is carried as far as the
poll rather than being baked into a URL on the way. Novita puts the id in a query string,
which the same `poll_template` covers with no new field at all.

Google is the awkward one, and worth reading before enabling: the model is in the path but
with a `:predictLongRunning` suffix, the key is not a Bearer token, the input is a batch of
one, completion is a boolean rather than a status string, and the finished clip lives behind
the same key as the API - an anonymous GET is answered 401, which arrives as a corrupt mp4
rather than as an error. Six profile fields, one per hazard. The seventh is not a field at
all: **Veo has no audio-driven mode**, so a Google entry configures `i2v_model` and leaves
`lipsync_model` empty, and a mode with no model configured now contributes no `ModelSpec` and
is declined in `accepts` - before that, the empty string went out as a slug and every
speaking shot spent its wall clock reaching a 404.

Free allowances are sized for evaluation. Together they are a handful of clips against a
target of four videos a day, and the honest reading of this chain is that it removes the
cliff at the end of one allowance, not the per-second cost of hosted video.

**An account that cannot pay is asked once.** Learning it costs a full `CREATE_RETRY` cycle -
five attempts, about three and a half minutes of measured backoff - because an uncredited
account is throttled to a trickle and answers 429 to the very retries trying to reach the
honest 402. That is a fair price once and not twelve times: the chain is built once per
animate stage and the provider instance is reused, so before the latch a twelve-speaking-shot
episode spent forty minutes being told the same thing and fell back to the local renderer
every time anyway. Anything a hosted model returns is forced through
`media/video/conform.py` to exactly `frames` frames at `fps` and `size`, because clips are
stream-copy concatenated against an audio timeline and a clip half a frame long walks the
whole episode out of sync while remaining a perfectly valid file. Lip-synced clips are
trimmed or clone-padded, never retimed; image-to-video clips are retimed, so a model that
only emits five-second clips does not get to dictate the edit.

Model slugs are config, not code (`providers.video.replicate.*_model`, with `*_inputs`
mapping our fields onto the model's parameter names), because hosted catalogues churn and a
wrong slug baked into code would fail every shot at 3am. That extends to the prompt: the
audio-driven models split on it - OmniHuman's schema is exactly image and audio and it
rejects anything else, while Wan-S2V *requires* a prompt and rejects the prediction without
one - so the lipsync payload sends `motion_prompt` only when the mapping declares a prompt
field, and swapping between the two is three lines of config. `asa video preflight` checks the
configured models against the live catalogue - auth, slug resolution, input mappings, extras
and required inputs - without creating a prediction, so it is free to run and worth running
after any long gap.

Two classes of defect live here and neither is visible from the code, which is why the
preflight exists: an input that is an **enum** rather than the number it looks like (Kling's
`duration` accepts 5 or 10 and rejects everything else, so a 4.5-second shot asking for 4
fails the whole prediction - we ask for the shortest offered length that covers the shot and
let the conformer retime), and a **default that quietly discards our work** (SadTalker's
`preprocess` defaults to `crop`, which throws away the composed frame and returns a bare head
at its own scale; cut against the local shots either side it reads as a different film).

Extras are kept apart per mode - Replicate validates against the chosen model's schema and
rejects unknown fields, so one shared bag posts SadTalker's `preprocess` to Kling.

**A shot is driven by its own slice of the voice.** A long speech is covered by several
setups, and they all carry the same line index; handing each the whole wav and trimming the
returned video afterwards would sync every part to the line's opening words - a mouth moving
convincingly and saying the wrong thing, which is worse than one that does not move.
`ShotJob.voice_offset_s` / `voice_duration_s` carry the window and the provider cuts it.

**Long speeches.** A speech longer than `PLAN_SHOT_S` is planned as up to `MAX_PARTS`
*different* setups, not one picture under several camera moves. The split has to happen in
`plan_shots`, which has no audio, so it is estimated from word count at a measured 2.3 w/s;
`time_shots` then divides the speech's real span between the parts. This is what the previous
`_split_long` got wrong - its sub-shots shared the parent's image - and it left 52% of a
finished episode resting on a still. On that episode the change took the longest unbroken
hold on one picture from 39.8 s to 10.3 s, and the shot count from 19 to 27, for two extra
image generations. The framing rotations must contain no repeats, or two adjacent parts
resolve to the same `image_key` and merge back into one long hold.

**Cost.** About 45 ms/frame at 1080p on 6 workers for the local renderer - roughly realtime.
Images, not compute, are the constraint: a scene costs one generation per distinct camera
setup, capped by `max_images_per_scene`.

---

## 9. Assembly (`asa.assemble`)

Scenes are rendered to individual mp4s (so a failed scene re-renders alone), then concatenated
with the ffmpeg concat demuxer. Audio is mixed as one continuous timeline and muxed last, so
there are no per-scene audio seams.

Export: `-c:v libx264 -profile:v high -pix_fmt yuv420p -crf 20 -preset veryfast -g 48`,
`-c:a aac -b:a 192k -ar 48000`, `+faststart`. 1920×1080 @ 24 fps.

Intro ≤ 3 s (skip it entirely on Shorts), outro 8–12 s with end-screen-safe margins.

---

## 10. Subtitles (`asa.media.subtitles`)

Timing comes free from §6 (you know exactly which clip plays when). Word-level timing for
animated captions comes from the same forced alignment as §8.1.

- `.srt` and `.vtt` written to `data/out/<job>/`
- Optional burned-in animated captions via an ASS file with `\k` karaoke tags rendered by
  ffmpeg's `subtitles` filter
- Style: 56 px bold sans, white with 4 px black outline + 40% drop shadow, max 2 lines,
  ~32 chars/line, positioned at 78% height (above the mobile progress bar, below faces)

Upload the `.srt` via `captions.insert` — do **not** rely on YouTube ASR for AI voices.

---

## 11. Thumbnails (`asa.assemble.thumbnail`)

Composited locally from assets you already have — no extra image API calls:

1. Character puppet at a scripted expression, scaled large (face ≥ 45% of frame height)
2. Background plate from the story's key scene, blurred + saturation-boosted
3. Rim light / outline stroke on the character to separate from background
4. 2–5 words, ~150 px, heavy display font, high-contrast fill with outline
5. Optional prop conveying the conflict

Generate **6 variants** (expression × text × layout), score each with a local heuristic:

| Criterion | Method |
|---|---|
| Face size ratio | Pixel measurement |
| Subject/background contrast | Mean ΔE between subject mask and surrounding ring |
| Text legibility at 320×180 | Downscale, measure edge energy in text region |
| Colour vibrancy | Mean saturation + palette spread |
| Composition | Rule-of-thirds distance of face centroid |
| Clutter | Edge density outside subject |
| **Truthfulness** | LLM check: does this depict something that happens in the story? Fails the whole set if not. |

Top scorer is uploaded; all 6 stored for A/B once the analytics loop has data.

---

## 12. Metadata (`asa.publish.metadata`)

One LLM call produces 8 title candidates + description + tags. Scored locally:

`curiosity`, `clarity`, `emotional_appeal`, `search_relevance` (keyword overlap with the
research topic), `click_potential`, **`accuracy`** — and accuracy is a *gate*, not a weight: any
title the LLM judges unsupported by the actual script is discarded outright.

Also enforced: length ≤ 70 chars, no ALL-CAPS words > 2, no "you won't believe", no fake
numbers, description carries the licence attributions block.

Hashtags are merged, not passed through: `channel.hashtags.lead` (the three YouTube renders
above the title) + the model's story-specific hashtags + `channel.hashtags.evergreen`, then
normalised to CamelCase, deduped case-insensitively and capped at **15** — one more than
that and YouTube ignores every hashtag on the video. Tags fill any shortfall, so a model
that returns no hashtags still ships something discoverable.

---

## 13. QC & approval (`asa.qc`, `asa.dashboard`)

**Technical gate** (hard fail): duration > 0, A/V sync drift < 100 ms, no silent stretch > 4 s,
loudness within ±1 LU of −14 LUFS, no black frames > 1 s, resolution/fps/codec correct,
subtitle count matches dialogue count, every asset in the timeline has a `licenses` row.

**Content gate** (flag → human): the safety checklist in
[05-COMPLIANCE.md](05-COMPLIANCE.md) §3.

**Dashboard** at `127.0.0.1:8420` shows: video preview, full script, all 6 thumbnails, title
candidates with scores, description, cast, duration, every safety flag with severity, and the
similarity report against prior episodes. Buttons: **Approve & schedule**, **Approve as
private**, **Send back to stage X**, **Reject**.

`auto_publish: false` is the default and stays false until you have ~20 clean manual approvals.

---

## 14. Analytics & learning (`asa.analytics`)

Daily pull from the **YouTube Analytics API** (separate quota from the Data API) at 1, 3, 7, 14,
28 days post-publish: views, estimatedMinutesWatched, averageViewDuration, averageViewPercentage,
impressions, CTR, likes, comments, shares, subscribersGained, trafficSourceType.

`analyze.py` attributes performance to features already recorded per video: primary animal,
archetype, moral, hook type, duration bucket, title style, thumbnail variant, narrator voice,
music mood, emotional theme.

With few videos, ranking by raw average is noise. Use **Bayesian shrinkage** toward the channel
mean (`score = (n·x̄ + k·μ) / (n + k)`, k ≈ 5) so a single lucky video doesn't hijack strategy.
Output is written to `strategy` as adjusted weights for §1's scorer and a "prefer/avoid" list
injected into the story prompt.

**Explicitly not done:** no engagement manipulation, no automated comments, likes, views, or
sub exchanges. The loop only changes *what you make*, never *who sees it*.

---

## 15. Error handling

| Concern | Mechanism |
|---|---|
| Retry | Exponential backoff + jitter, `max_attempts` per stage class (network 5, render 2, LLM 3) |
| Timeout | Per-stage wall clock in config; render stages get 3× the estimate |
| Rate limits | `core/quota.py` token bucket per provider, persisted in SQLite so it survives restarts; on 429 the provider is marked cold until its reset time |
| Quota exhaustion | `QuotaExhausted` → try next provider in chain → if all exhausted, job → `QUOTA_BLOCKED` and is retried automatically after the earliest reset |
| Failed job | `jobs.state='FAILED'`, `errors` row with traceback + stage + provider + payload hash |
| Logging | `structlog` JSON to `logs/asa.jsonl`, one `job_id` field on every line |
| Notification | `core/notify.py` — pluggable; default writes to the dashboard's alert list, optional ntfy.sh/webhook |
| Resume | The runner reads `job_stages` and starts at the first incomplete stage. Because every stage is idempotent and writes into `data/work/<job_id>/`, a crash mid-render costs one scene, not one video |
| Poison jobs | `jobs.attempts` — after 3 full-pipeline failures the job is parked in `FAILED` and requires manual reset |
