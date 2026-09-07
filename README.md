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
asa db migrate                      # apply any pending schema migrations
asa auto                            # topic -> story -> video -> QC, ready to review
asa approve 1 --who you && asa run 1
```

`asa auto` is the whole pipeline behind one verb. It collects topics if there is nothing
usable in the table, queues a job, advances it through every stage, and stops at the
approval gate with the file to watch printed. `asa auto --topic "a clever fox opens a
village bakery"` commissions a specific episode - a typed seed is stored as a *binding
brief*, so the script has to say what you asked for rather than being loosely inspired by
it. `--upload` approves as `--who` and uploads in one go; without it, nothing is published.

Before it renders anything it prints the style and checks whether any hosted video provider
can actually lip-sync, and refuses to start if none can. That failure is otherwise silent:
`providers.video` ends in `local`, so a chain with no funded account does not error, it
renders every speaking shot as a camera move over a still and reports success.
`--ignore-no-lipsync` renders anyway.

The four-command form still works and is what you want when you are debugging one stage:
`asa research`, `asa job new --topic ...`, `asa run <id>`, `asa approve <id> --who you`.

`asa doctor` is the fastest way to find out what is missing. It checks system tools,
Python version, config, `.env` permissions, key prefixes, LLM buffer depth, gitignore
coverage, the database, and the asset libraries.

---

## What it does, stage by stage

| Stage | What happens |
|---|---|
| `select_topic` | Highest-scoring unused topic from the research table |
| `story` | 3 structured LLM calls: outline → draft → scene breakdown |
| `art` | One background plate per location (puppet), or one image per camera setup (cinematic), cached and licence-recorded |
| `audio` | Kokoro-82M speaks every line locally; **the audio sets the scene durations** |
| `animate` | Puppets composited over parallaxed plates, or generated stills cut into shots and performed; 6 processes, streamed to ffmpeg |
| `assemble` | Stream-copy concat, one-pass audio graph, two-pass loudnorm to −14 LUFS |
| `subtitles` | SRT + VTT timed from the synthesised audio, not from the script |
| `thumbnail` | 6 variants from assets already owned, scored at feed size |
| `metadata` | Titles generated then scored in code, with a hard anti-clickbait gate |
| `qc` | 16 mechanical checks; any failure blocks publication |
| `approval` | Stops and waits for a human. Always, until you turn that off deliberately |
| `upload` | YouTube Data API, private by default |

### The look: `channel.art_style`

Two presets, and each switches three things *together* in `media/images/scene_image.py`:
what the subject is, how it is lit, and the negative prompt.

* **`cartoon`** - a modern 3D animated feature. Stylised animal faces with large readable
  eyes and a defined muzzle, human bodies in real clothes, bright high-key light,
  saturated colour. The muzzle clause is not decoration: a hosted lip-sync model animates
  the mouth in the still it is handed, and the image model's default for "cartoon fox"
  draws a dot.
* **`photoreal`** - the low-key night film the channel was first tuned against
  (`CINEMATIC_LOOK`, measured at mean luma 21/255 against the reference video).

The third of those - the negative - is the one that is easy to miss, and it is why this is
a preset rather than a style string you edit. The photoreal negative ends with `cartoon,
illustration, drawing, anime, 3d render, cgi`, so a cartoon look asked for while that
negative is in force is a prompt arguing with itself, and the model settles the argument
differently on every generation. That is art-style drift arriving from the opposite side to
the one `_PUPPET_STYLE_WORDS` guards, and it is invisible until someone watches the render.
`tests/test_art_style.py` asserts no preset forbids the medium it asks for.

`channel.look_hint` still overrides whichever preset's grade is in force, so the two dials
stay independent: the preset chooses the medium, `look_hint` regrades it. The preset also
reaches the hosted video model as a one-clause anchor on the end of the motion prompt -
an i2v model given a stylised frame and a prompt that never says what it is drifts toward
the photoreal mean over five seconds, and the drift shows up as a cut.

`render_mode: puppet` ignores all of this. The puppets *are* the style.

### The two render modes

`production.render_mode` picks how a scene becomes moving picture.

**`puppet`** is the original path: layered PNG characters composited onto an empty
background plate, with real viseme lip-sync, blinks and staging. Flat cartoon look, exact
character consistency, no image credits per scene.

**`cinematic`** generates photoreal frames containing the characters, which is the only way
to reach the anthropomorphic-animal-in-real-clothes look. A generated still cannot be
re-posed, so the question is what to do with it for the twenty seconds a scene lasts.

The first answer was: hold it, and pan slowly across it. That is what made finished
episodes read as a photograph with a voice-over - nobody on screen was ever the person
talking. `production.performance` replaces it with two things:

- **The scene is cut.** One shot per run of lines by the same speaker: a close-up of
  whoever is talking, a reverse when someone answers, a wide or an insert under narration.
  Cuts land in the silence between lines, so the picture changes when the voice does. A
  speech too long for one setup becomes up to four *different* setups rather than one still
  panned across. Several shots share one generated image, so a six-line exchange cuts six
  times on two generations (`max_images_per_scene` is the ceiling).
- **Each shot is performed.** The per-character amplitude envelope drives a 2.5D warp on the
  still: a muzzle-sized jaw band, blinks, head drift, camera breath.

**The local renderer does not do lip-sync**, and that is a property of the approach rather
than a bug waiting to be tuned out. It places the mouth from the framing word in the prompt,
and the image model does not reliably obey that word - in one finished episode a shot asked
for as a close-up of an owl came back as a wide two-shot of a rooftop. Two rescues were tried
and measured: a face detector (YuNet missed five of twelve generated stills, and on one
preferred a human in the background to the animal filling the frame), and generating
mouth-open/mouth-closed pairs on a fixed seed to blend between (changing that one clause
changed 63% of the pixels - two different pictures, not two expressions of one). So the local
warp is deliberately small, confined inside the head box, scaled by how far that box can be
trusted - and below `production.performance.min_face_confidence` (default 0.5) skipped
entirely, because scaling a wrong answer down does not make it a right one. Measured on a
finished still: asked for a close-up of a fox, the image model returned a seated two-thirds
figure with its muzzle in the upper right, so the prior put the muzzle band on the
character's lap and the desk behind it and the result tracked the voice at 0.995 correlation
- on furniture. A prior rates itself 0.35 and a detector's `.face.json` sidecar 1.0, so the
guesses now render as camera-only and only a measured box animates. What the local path
contributes is a still that does not look frozen. It is not a mouth.

**And it never will be a hand.** There is no pose, limb or gesture concept anywhere on the
local path - only a vertical displacement of a photograph - so arms and legs do not move on
it at any setting.

**For lip-sync or gesture, put a hosted model in `providers.video.chain`.** The shot layer is
provider-agnostic (`media/video/base.ShotVideoProvider`) and `media/video/replicate.py`
implements it. The default speaking model is `bytedance/omni-human`, which drives the whole
figure - mouth, head, shoulders, arms, hands - from the same audio, so gesture and lip-sync
are one prediction rather than two problems. (A talking-head model like SadTalker syncs the
mouth and leaves the body as frozen as the local path does.)

**Models fall back to other models, not straight to local.** Dropping to the local renderer
because a hosted model errored gives up lip-sync *and* gesture - the two things the hosted
call was made for - over something as ordinary as a slug renamed overnight or a prediction
that choked on one image. So each mode has a chain of its own, ordered by how much of the
performance survives: whole figure, then whole figure directed by a prompt
(`wan-video/wan-2.2-s2v`), then a talking head (`cjwbw/sadtalker`), and only then the local
camera move. Every entry carries its *own* input mapping, extras and clip-length enum,
because those are properties of the model - a fallback inheriting the primary's mapping would
fail every prediction it was added to rescue. A 402 or a rejected token does not walk the
list: every slug on the account fails those identically, so it drops to local at once.

**Several accounts, spent one after another.** Free signup allowances at these services are
small and separate, so running out is the expected case rather than the exceptional one. The
recommended chain is `[dashscope, wavespeed, fal, replicate, local]`: a provider whose
allowance is spent raises `QuotaExhausted`, which advances the chain, so the shot reaches the
next account with credit instead of losing its lip-sync to the free renderer. A key that is
missing or revoked at one service advances it too - with several providers that is exactly
what the others are for.

Most of them are hosting the *same* Wan, which is the point rather than an accident: it is
open weights, so hosts compete on it and the ordering decides what a shot costs rather than
what it looks like. WaveSpeed serves Wan-2.2-S2V at roughly $0.06/s at 720p against ~$0.14/s
for OmniHuman anywhere; `dashscope` is Alibaba Model Studio, the same model with no reseller
in front of it - another signup allowance for a model already known to render real gesture,
and the lowest per-second price after it. fal hosts OmniHuman and Wan-S2V on a separate
allowance again, and `siliconflow` hosts Wan i2v on one more.

**`novita` is the exception, and the only entry here that changes what the video looks
like.** Everything else in the chain runs Wan 2.2; Novita serves `wan2.7-i2v` - two model
generations, which is a larger quality difference than anything else in this config. Its i2v
route documents an `audio_url` input, so it may cover speaking shots from one route rather
than from a separate speech-to-video model. That last part is **unproven here and fails
silently if it is wrong**: a model that ignores the audio and generates its own dialogue
returns a fluent performance of the wrong words, and lip-synced clips are never retimed, so
nothing downstream catches it. It sits second in the chain behind the proven Wan-2.2 path
until one speaking shot has been watched; promoting it is a one-line change after that.

**Google Veo is in the table and out of the chain, on purpose.** `google` is the Gemini API
(*not* Google Flow, which is the subscription web app on the same model and has no API), and
it differs in kind rather than in price: Veo has no audio-driven mode - it generates its own
dialogue instead of performing our TTS line - so it cannot answer a speaking shot at all. It
is an image-to-video provider for the establishing wides and inserts that currently go to the
local renderer for free, it is several times Wan's per-second rate, and Veo is excluded from
the Gemini free tier, so unlike every other entry there is no allowance in front of the bill.
Add it before `local` only having decided that moving B-roll is worth paying for.

Be honest about the size of this: those allowances together are a handful of clips, not a
day of episodes, and some services restrict what free-tier output may be used for. What the
ordering removes is the cliff, not the cost. Adding a service is a `Service` profile in
`media/video/hosted.py` - the submit/poll/collect shape differs only in where each puts its
JSON fields - and everything else it needs is inherited from `HostedShotProvider`. Run
`asa video preflight` before trusting any of it: it walks the chain in order and probes each
slug without creating anything billable.

Providers answer per *shot*, not per episode, because hosted video is billed by the second:
`speaking_only` sends the talking close-ups to the hosted model and leaves the establishing
wides to the local renderer for free. An account with no credit is asked once and then skipped for the rest of the run -
finding that out costs a full retry cycle, because an uncredited account is throttled to a
trickle and answers 429 to the very retries trying to reach the honest 402, and paying that
on every shot cost a twelve-shot episode forty minutes to learn the same thing twelve times. Whatever comes back is forced
to exactly the frame count the edit expects, since clips are stream-copy concatenated against
an audio timeline. `local` is always appended last, so a hosted model out of credit degrades
to a render rather than to a failure. Set `REPLICATE_API_TOKEN`, then run:

```
asa video preflight
```

It reads the live catalogue (no prediction, so it costs nothing) and fails loudly on a model
slug that has been renamed, an input mapping that no longer matches the model's schema, or a
required input we never send - for *every* model in every chain, fallbacks included, since a
fallback exists to be used on the day the primary breaks and that is the worst possible day
to discover its mapping was wrong all along. Run it after any long gap - hosted catalogues churn, and
finding out mid-render is expensive: the episode has already spent its image budget getting
there. The first run of it caught two real defects, both invisible from the code: `duration`
on Kling is an enum of `[5, 10]` that rejects anything else, and SadTalker's `preprocess`
defaults to cropping the composed shot down to a bare floating head.

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

## Episodes about real things

The cast is animals and the story is a story. What an episode may additionally be *about*
is a real discovery, an invention, a piece of mathematics or astronomy, an ancient
technology, or a classical text - `story.subjects` lists the domains, and
`prompts/_blocks/subject_bible.md` is the contract.

Being true is a constraint, not a decoration, so it is enforced as data rather than as
tone:

* **The outline gains `subject`, `facts` and `period`.** All three default to empty, which
  is what fiction means. A `subject` with an empty `facts` list is rejected by the schema -
  an episode that names a real thing and lists nothing it claims about it has put the whole
  burden of accuracy on prose nobody validated. A "fact" of fewer than four words is
  rejected too: `"Eratosthenes"` cannot be checked against a source, a sentence can.
* **`facts` is a ceiling, restated in every later call.** The draft and scene prompts each
  reprint it with "do not add a fact that is not on this list - not a date, not a number,
  not a name". The outline JSON is passed whole into both calls anyway, but a requirement
  that is merely *present* in a field is not a requirement; the same reasoning as
  `brief_block`.
* **Real people are credited, never cast.** The safety block's no-real-person rule is
  narrowed rather than dropped: a name in narration, and nothing else - no character, no
  dialogue, no design, no voice, no likeness.
* **Ancient texts are covered as historical and literary objects.** The Vedas, the Sangam
  poems, Euclid, cuneiform tablets: what they are, when and where composed, in what
  language and metre, how they were transmitted, and the mathematics and linguistics that
  came with them. No claim that any belief is true or false, no deity as a character or an
  image, no ritual as instruction, and no "the ancients already knew modern physics".
* **The pseudo-history shapes are named and excluded** - suppressed civilisations, ancient
  astronauts, lost superior technology, the lone genius. They are the most viral framings
  in this genre, they are all false, and naming them is what beats them, exactly as the
  image negatives have to name "mascot".
* **`period` fixes the picture.** `channel.region_hint` pins every generated frame to
  contemporary India, which is right for this channel and was putting Indian street
  furniture in Alexandria. A non-empty period *replaces* that hint - the two cannot both be
  true - and carries an explicit anachronism negative, because power lines and printed
  labels are the default failure of a historical image prompt.
* **The description tells the truth about itself.** The synthetic-disclosure line used to
  end "All characters and events are fictional", which becomes a false statement the moment
  an episode is about something real. On a factual episode it now says the animals are
  invented and the subject is not, and the `facts` list is published above it - the audit
  trail, in public, where a viewer can check it.

**Nothing here verifies that a claim is true.** No check in this repository can, and one
that pretended to would be worse than none. What the machinery buys is that the claims are
enumerated before they are dramatised, bounded so the script cannot add more, carried to
whoever approves the episode, and published beside the video.

`asa auto --subject` commissions a factual episode on purpose. It exists because the
measurement said it had to: even with the scoring bonus a subject candidate lands around
0.30-0.35 while a strong wildlife story from RSS reaches 0.40, so on a mixed pool subjects
compete honestly and honestly lose most days. Rigging the scorer until they win would
decide, in arithmetic nobody reads, that this is a science channel now.

The `subjects` research collector seeds this from about fifty curated Wikipedia articles
(`collectors.SUBJECT_SEEDS`) across measurement, engineering, optics, the ancient world and
its texts. Curated rather than searched: a search for "ancient science" returns
pseudo-archaeology within a page, and the pipeline would spend every episode arguing with
its own research.

### Why factual topics needed a scoring fix

They scored 0.207-0.254 against a 0.35 selection floor and could never be chosen. The
deficit is structural rather than a judgement: `emotional` and `story` are measured by
matching a *fiction* vocabulary (lost, promise, betray, must, until) against the topic
sentence, and a subject candidate's sentence is an encyclopedia extract - it reads near zero
however good the episode would be. `SUBJECT_BONUS` lifts exactly those two components,
scaled by how much curiosity the sentence carries, and is gated on a `subject` signal only
the subject collector sets - so the animal-fact collector, which produces fiction seeded by
a true detail and scores fine on the fiction axes, is deliberately not touched.

While measuring that, something else turned up: on a live table of 306 topics the
highest-scoring *selectable* one reached **0.347**, so the 0.35 floor was rejecting the
entire pool and every episode on this install had come from a hand-typed `--topic`. The
floor is now `research.min_score` and is **left at 0.35** rather than quietly lowered,
because what sits immediately below it is not merely weaker material - at 0.347 the same
table held a real news item naming a real politician. `asa auto` now prints the best
rejected topic and its score, so the choice is visible instead of silent.

## What "goes viral" means here, and what it does not

Nothing in this repository can make a video go viral, and no setting claims to. What the
pipeline does is remove the reasons a good story loses its audience anyway, and every one
of those is mechanical:

* **The first five seconds.** `prompts/_blocks/retention_bible.md` requires the story to
  open on trouble already in progress, stated concretely, with no narrator explaining the
  setting first. The draft call enforces it again on the opening line specifically.
* **A question the viewer wants answered**, planted in the opening, answered at the end,
  with another open in between - and one honestly-planted turn before the climax.
* **Cuts that land when the voice changes.** `production.performance` cuts a scene on
  speaker changes rather than holding one still under a voice-over, which is what made
  earlier episodes read as a photograph someone was talking over.
* **A mouth that moves.** `providers.video` sends every speaking shot to a model trained on
  faces. `asa auto` refuses to start when nothing in that chain can, because the local
  fallback degrades silently rather than failing.
* **Titles and thumbnails that are true.** The anti-clickbait gate in
  `publish/metadata.score_title` and the thumbnail truthfulness check are *not* obstacles
  to this - a title that oversells is how a channel trains YouTube that its clicks do not
  turn into watch time.

There is no engagement manipulation here and none is going to be added. The upload volume
is bounded by the YouTube API quota at roughly four uploads a day, monetisation requires
Google's review on Google's timetable, and the API forces every upload to `private` until
your project passes their audit. Read `docs/05-COMPLIANCE.md` before planning around any of
that.

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
pytest tests/ -q     # 508 tests, no network, no keys, no GPU
```
# devCreator
