# 08 — Build Status

What exists, what is verified, and what is honestly not done. Written 2026-09-01 against
the code in this repository, not against the plan.

---

## 1. Phase completion

| Phase | Scope | State | Evidence |
|---|---|---|---|
| 1 | Compositor, camera, parallax, silent render | **done** | `data/out/phase1_slice.mp4`, 24 tests |
| 2 | Kokoro TTS, lip sync, mix, master, mux | **done** | `data/out/phase2_slice.mp4` at −14.0 LUFS / −1.0 dBTP |
| 3 | Story generation, adaptive model routing | **done** | `asa run` story stage; `model_health` table |
| 4 | Character factory, 29 species, suitability casting | **done** | `assets/characters/*`, `data/out/species_sheet.png` |
| 5 | Background images, cache, licence ledger | **done** | FLUX 5.4 s cold → 0.01 s cached; APACHE-2.0 recorded |
| 6 | Runner, state machine, leases, resume | **done** | `asa run/jobs/job retry`; `job_stages` rows |
| 7 | Subtitles, thumbnails, metadata | **done** | SRT/VTT, 4 layouts, title honesty gate |
| 8 | QC gate, approval dashboard | **done** | `asa dashboard`; 16 mechanical checks |
| 9 | YouTube OAuth + upload | **code complete, blocked on your OAuth** | see §3 |
| 10 | Research collectors + scoring | **done** | `asa research`; 8-component score |
| 11 | Analytics fetch + feedback loop | **code complete, unexercised** | see §3 |
| 12 | systemd units, backup, tests | **done** | `deploy/systemd/`, `scripts/backup.sh`, 79 tests |

---

## 2. Measured on this machine (i7-8550U, no GPU)

| Operation | Measurement | Notes |
|---|---|---|
| Frame composite, 1 worker, LANCZOS backgrounds | 333 ms | the original estimate of 40 ms was **8× wrong** |
| Frame composite, 6 workers, BICUBIC backgrounds | 51–161 ms | 0.26–0.81× realtime; long scenes amortise worker start-up |
| **7-minute episode, animation only** | **~12–18 min** | not the ~10 min first claimed |
| 9 background plates (FLUX, uncached) | 89 s | 4.1–5.3 s each |
| 13 lines of dialogue through Kokoro | 41 s | for 79 s of finished audio |
| Kokoro TTS | 1.20–1.46× realtime | plus ~24 s one-off init |
| FLUX.1-schnell 1024×576 via HF | 5.4–5.7 s | cache hit: 0.01 s |
| Story generation (3 LLM calls) | 30 s – 8 min | entirely dependent on which free model answers |

---

## 3. What is NOT verified, and why

These are stated plainly because a green tick you cannot reproduce is worse than a blank.

**YouTube upload has never run.** The path was exercised up to the credential check, which
failed correctly and legibly:

```
FAIL upload -> FAILED   no YouTube token at config/youtube_token.json.
               Run `asa youtube auth` once, on a machine with a browser, to grant access.
```

It was classified as `auth`, marked `needs_human`, and **not retried** — which is the
designed behaviour. But no bytes have been sent to YouTube from this machine, so treat the
upload itself as untested until you run `asa youtube auth` and one upload completes. Note
also that **an unaudited API project has every upload forced to `private` by Google** —
that is not something this code can work around.

**Analytics has never run.** It needs published videos to read, so it cannot be exercised
before Phase 9 is. `compute()` correctly returns an empty strategy below five videos.

**The music library is empty.** `asa assets scan` fails, deliberately. Episodes render
without a score. Populating it requires a human to accept licence terms; see docs/07 §8.

**Hugging Face's exact free allowance is unpublished.** `whoami-v2` confirms a monthly
credit window with `canPay: false`, so exhaustion errors rather than bills. The number in
config is a placeholder — rely on the error path, not the counter. ⚠ Needs verification.

**Free-model availability is genuinely unreliable.** In live runs, 5 of 7 `:free`
OpenRouter models returned 429 inside one minute. The router is built for this, but it
means wall-clock time per story is unpredictable, not that something is broken.

---

## 4. Bugs found and fixed during the build

Recorded because each one was a real defect that tests now cover.

| Bug | Consequence | Fix |
|---|---|---|
| `alimiter` defaults to `level=true` | master came out at −13.1 LUFS / 0.0 dBFS — it was *boosting* to the ceiling | two-pass loudnorm + `level=disabled` |
| Single-pass `loudnorm` is only ±1 LU accurate | loudness target missed | measure pass, then apply with measured values |
| `extract_json` required a *closing* fence | a reply truncated at `max_tokens` kept its ` ```json ` prefix and failed at column 1 | strip either fence independently |
| No salvage for truncated JSON | eight good scenes discarded because a ninth was cut off | `salvage_truncated_json` rewinds to the innermost open array |
| Scenes prompt never stated field names | model emitted `duration`, omitted `action`, made `sfx` a string | explicit field list + tolerant coercion in the schema |
| 403 treated as an account auth failure | one bad model aborted the entire provider chain | 403 demotes the model; only 401 (or an explicit key message) is auth |
| Cast budget checked outside the repair loop | "2 new characters, limit is 1" killed the job outright | moved into `post_check`, so it is repairable |
| `_keywords` required 4+ characters | silently dropped "fox", "cat", "dog", "owl" | 3+ characters with an expanded stopword list |
| Rhino horn drawn before the skull | invisible in every frame | moved after the muzzle |
| `assets scan` passed on an empty library | "all clear" on nothing at all | exits 1 with an explanation |
| Backgrounds resized with LANCZOS every frame | 333 ms/frame | BICUBIC + 6-worker pool → ~100 ms |
| Sky and ground shared one hue | "forest at dusk" produced a green sky | separate TIME_SKY and PLACE_GROUND tables |
| Scenes saved before their locations existed | `FOREIGN KEY constraint failed` killed a good 10-scene story | location rows created with the story, where they belong |
| `videos` upserted on `ON CONFLICT(id)` — an autoincrement PK | re-running assemble inserted a duplicate row; later stages updated the wrong one | `UNIQUE(job_id)` + a real upsert (migration 003) |
| QC compared the container duration against a value probed from that same container | the check could never fail | compares against the scene timeline, which comes from the audio |
| Render workers each recomputed the parallax split | 3 LANCZOS resizes of a 2.7k plate × 6 workers, per scene | workers load the planes the art stage already baked |
| `animate` returned a `seconds` key that collided with the runner's own log binding | `TypeError` aborted the job *after* three minutes of successful rendering | reserved keys are stripped from stage detail before logging |
| Thumbnail text anchored to fixed coordinates | title printed across the character's face | placement chosen from the character's actual alpha bbox |
| API keys echoed into terminal output | both keys exposed | `python-dotenv`, never shell-source; `.gitignore` widened; keys rotated |

---

## 5. First end-to-end episode (2026-09-01)

One full run, `--set production.target_minutes=2`, seed topic "a lost lantern must be
returned before the river floods the market bridge at night":

| Stage | Result |
|---|---|
| story | "Before the Bridge" — 10 scenes, 13 lines, **reused** Milo and Wren rather than inventing new characters |
| art | 9 background plates, 89 s |
| audio | 79.0 s of speech, 41 s |
| animate | 10 scenes, 0.24–0.81× realtime |
| assemble | 79.003 s, 18.8 MB, **−14.2 LUFS / −0.9 dBTP**, A/V start both 0.000 |
| subtitles | 16 captions, none overlapping |
| thumbnail | 6 variants, best scored 0.933 |
| metadata | "Fox and Owl Race the Rising River to Save the Lantern Bridge" — passed the honesty gate |
| qc | **0 failures, 0 warnings** |
| approval | stopped and waited for a human, as designed |

Known art limitation, visible in that episode: FLUX put human figures on a bridge and
garbled text on a shopfront despite both being in the negative prompt. Strengthening the
*positive* prompt ("completely deserted", "all signs are blank") removed the figures
entirely and mostly fixed the signage — but in-image text is still not reliable, and
nothing in this pipeline detects it automatically. Review thumbnails and establishing
shots before approving.

---

## 6. Test coverage

77 tests, no network, no API keys, no GPU. `pytest tests/ -q`.

The ones that matter most are the ones that encode a bug that actually happened:
`test_extract_json`, `test_scene_coerces_common_model_mistakes`,
`test_no_layer_escapes_the_canvas`, `test_horned_species_horns_are_visible`,
`test_thumbnail_text_never_covers_the_character`, `test_captions_never_overlap`,
`test_lease_prevents_double_claim`, and `test_chain_breaks_exactly_once_at_the_human_gate`.
