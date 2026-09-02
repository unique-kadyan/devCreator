# 05 — YouTube Policy, Safety, Copyright and OAuth

_Verified 2026-09-01. Policy text changes; re-read the primary sources before your first
public upload. Nothing here is legal advice._

---

## 1. The API audit gate — read this before writing any upload code

**Videos uploaded through `videos.insert` from an API project created after 2020-07-28 that has
not passed compliance audit are locked to `private`.** The uploader gets an email saying the
video is locked and that the fix is an audited or official client. Projects created before that
date are unaffected.

Practical consequences for this project:

| | |
|---|---|
| Can you automate upload on day one? | **Yes** — as private. |
| Can you automate *publishing* on day one? | **No.** |
| What lifts it? | Submitting the **YouTube API Services – Audit and Quota Extension Form** and passing human review. |
| How long? | Unpredictable. Weeks is normal. Plan for it. |
| Interim workflow? | Pipeline uploads as private → you flip to public in YouTube Studio. One manual click per video, which fits the human-approval mode you wanted anyway. |

Default quota for an unaudited project: **100 `search.list` calls/day, 100 `videos.insert`
calls/day, and 10,000 units/day for everything else combined.** The pipeline's budget
(~2,454 units/day, [02-PIPELINE.md §1](02-PIPELINE.md)) fits comfortably.

> **Build order implication:** do the audit application early, in parallel with development.
> It is the longest-lead item in the whole project and it is not something you can code around.

## 2. "Made for Kids" — the strategic decision you must make consciously

Talking-animal cartoons with morals are close to the centre of YouTube's Made-for-Kids
definition. If the channel is MFK (or individual videos are):

- **No personalised ads** — this is the large majority of typical ad revenue
- **No comments**, no end screens, no cards, no notifications, no playlist saves
- Data collection restricted under COPPA

You have three honest options. Pick one deliberately and record it in `config.yaml`:

| Option | What it means | Revenue posture |
|---|---|---|
| **A. Accept MFK** | Simple animal fables aimed at young children. Declare `made_for_kids: true`. | Low RPM, but a real, safe, compliant channel |
| **B. Write for teens/adults** | Animal *characters*, adult-shaped stories: workplace comedy, ambition, grief, social satire. Genuinely not child-directed in theme, vocabulary, or pacing. Declare `made_for_kids: false` **truthfully**. | Normal RPM |
| **C. Mixed, declared per video** | Requires honest per-video judgement | Workable but easy to get wrong |

**Do not** declare `false` on child-directed content to chase RPM. Misdeclaration is
enforceable and the schema deliberately makes `made_for_kids` a required, explicit field with no
default — you must state it.

My recommendation: **Option B.** It is the only one where the "animal characters with human-like
personalities, emotions and situations" premise in your brief is a *creative* advantage rather
than a compliance liability, and it separates you from the enormous, saturated field of
auto-generated kids' animal content.

## 3. Synthetic content disclosure

YouTube requires the **"Altered or synthetic content"** disclosure for *realistic* synthetic
media — content a viewer could mistake for a real person, place, or event. Clearly stylised
animation, AI voiceover on animated content, AI-written scripts and AI thumbnails are generally
**exempt**.

This pipeline produces obviously-animated cartoon animals, so disclosure is likely not required.
**Disclose anyway.** The `youtube_uploads.synthetic_disclosed` column defaults to `0` but
`config.yaml` ships with `disclose_synthetic: true`. The cost of over-disclosing is nil; the cost
of under-disclosing is removal or YPP suspension.

## 4. Inauthentic / mass-produced content

The monetisation bar explicitly targets templated, repetitive, low-effort output. This is the
policy your brief's §23 is really about, and the system is designed against it:

| Mechanism | Where |
|---|---|
| Archetype + animal cooldowns block back-to-back sameness | [02 §2](02-PIPELINE.md) |
| Beat-signature dedupe catches "same plot, new animal" | [02 §3](02-PIPELINE.md) |
| Embedding dedupe on loglines | [02 §3](02-PIPELINE.md) |
| Hard cap on videos/week in config, not "as many as possible" | `config.yaml` |
| Human approval gate by default | [02 §13](02-PIPELINE.md) |
| Reusable named cast building genuine continuity | [00 §1](00-ARCHITECTURE.md) |

A recurring cast across episodes is, incidentally, the strongest available signal that a channel
is a *show* rather than a content farm — both to viewers and to review.

## 5. The automated safety layer (`asa.qc`)

Runs before `AWAITING_APPROVAL`. **Flags for human review; never auto-blocks silently, never
auto-publishes over a flag.**

| Check | Method | Severity |
|---|---|---|
| Story similarity to own library | MiniLM cosine + beat-signature Jaccard | block if > 0.90, flag > 0.82 |
| Franchise/trademark leakage | Denylist of studio, franchise and character names + LLM review for unnamed evocation | block |
| Character-design similarity | Manual review at rig time; `style_hash` drift detection | flag |
| Violence / injury / weapons | LLM rubric against §4's `safety_rules` | block |
| Hate / demeaning content | LLM rubric + slur denylist | block |
| Sexual / romantic-adult content | LLM rubric | block |
| Dangerous imitable acts | LLM rubric | block |
| Misleading title/thumbnail | `accuracy` gate ([02 §11-12](02-PIPELINE.md)) | block |
| MFK classification | Heuristic on vocabulary, pacing, themes → suggests a value, human confirms | flag |
| Asset licensing | SQL query in [03](03-DATABASE.md) — any non-commercial or unlicensed asset | block |
| Music Content-ID risk | Source must be YT Audio Library / Pixabay / CC0 / CC-BY | block |
| Technical QC | [02 §13](02-PIPELINE.md) | block |

## 6. Attribution

Every CC-BY asset used generates a line in the description, built from the `licenses` and
`assets` tables:

```
Sound: "Leaves Rustling" by <author> — freesound.org — CC BY 4.0
Music: "Autumn Errand" by <author> — CC BY 4.0
```

`qc/policy.py` fails the job if a CC-BY asset is in the timeline and its attribution string is
absent from the rendered description. Attribution is not a nice-to-have; omitting it voids the
licence.

## 7. OAuth and secrets

**Never in source. Never in the DB. Never in a log line.**

```
config/.env            chmod 600, gitignored
  OPENROUTER_API_KEY=...
  HF_TOKEN=...
  FREESOUND_API_KEY=...
  YT_CLIENT_ID=...
  YT_CLIENT_SECRET=...
data/secrets/yt_token.json    chmod 600, gitignored — refresh token store
```

`.gitignore` covers `config/config.yaml`, `config/.env`, `data/`, `logs/`, `assets/music/*`,
`assets/sfx/*`. `scripts/doctor.py` refuses to start the pipeline if any secret file is
group/world-readable or if a key appears in a tracked file.

### One-time YouTube OAuth setup (manual, ~15 minutes)

1. Google Cloud Console → new project → enable **YouTube Data API v3** and **YouTube Analytics API**.
2. OAuth consent screen → External → add yourself as a **test user**.
   (Test-user refresh tokens historically expire in 7 days; publishing the app to
   "In production" avoids that. ⚠ **Needs verification** of current behaviour.)
3. Credentials → OAuth client ID → **Desktop app** → download JSON.
4. Scopes needed:
   `youtube.upload`, `youtube.force-ssl` (thumbnails + captions), `yt-analytics.readonly`.
5. Run `python -m asa.publish.oauth --authorize` once. It opens a browser, you consent, and the
   refresh token is written to `data/secrets/yt_token.json` with mode 600.
6. Submit the **Audit and Quota Extension Form** (§1). Do this now, not later.

The runtime only ever reads the refresh token; it never prompts interactively. If refresh fails,
the job goes to `FAILED` with `kind='auth'` and the dashboard raises an alert — it does not
retry blindly against Google.

## Sources

- [YouTube Data API — Quota and Compliance Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)
- [YouTube Data API — Revision History (2020-07-28 unverified-project restriction)](https://developers.google.com/youtube/v3/revision_history)
- [Complying with YouTube's Developer Policies](https://developers.google.com/youtube/terms/developer-policies-guide)
- [YouTube altered/synthetic content disclosure — summary](https://minimatters.com/youtube-altered-or-synthetic-content-disclosure/)
- [Royalty-free music sources and Content ID safety](https://www.foximusic.com/blog/7-best-free-royalty-free-music-sites/)
