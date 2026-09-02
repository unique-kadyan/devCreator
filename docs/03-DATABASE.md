# 03 — Database

**Engine: SQLite in WAL mode.** Not Postgres. Reasons: single machine, single writer, no
network, zero ops, and the whole state file is one thing to back up. `STRICT` tables give you
real type enforcement, which was the main historical argument for Postgres here.

Move to Postgres only if you later split rendering onto a second machine. The schema is written
to port cleanly (no SQLite-only constructs beyond `STRICT` and `datetime('now')`).

- Schema: [`migrations/001_initial.sql`](../migrations/001_initial.sql) — **applied and verified**
- Apply: `sqlite3 data/asa.db < migrations/001_initial.sql`
- Seed demo rows: `sqlite3 data/asa.db < scripts/seed_example.sql`

## Tables

| Table | Purpose |
|---|---|
| `research_topics` | Candidate topics with all nine sub-scores + embedding for dedupe |
| `watch_channels` | Competitor channels polled via free RSS (no API quota) |
| `characters` | The permanent cast. `puppet_dir` + `rig_json` are the consistency guarantee |
| `character_relationships` | Cross-character graph, feeds the story prompt |
| `locations` | Reusable settings; background plates cache against these |
| `stories` | One row per script, with the 5 beats as first-class columns + `beat_signature` for anti-repetition |
| `story_cast` | Story ↔ character with role |
| `scenes` | The animation contract. `duration_s` is authoritative (from audio), `duration_hint_s` is the LLM's guess |
| `dialogue` | Per-line, per-character, so TTS and captions map 1:1 |
| `licenses` | Seeded lookup of licence codes and whether commercial use is allowed |
| `assets` | Every file with provenance. **No asset may enter a timeline without a `license_id`** |
| `audio` | Every rendered clip with true duration, LUFS, and `text_sha256` so unchanged lines aren't re-synthesised |
| `jobs` | One row per video; `state` is the resume point; `lease_owner`/`lease_expires` prevent double-running |
| `job_stages` | Per-stage status/attempts/duration — this is what makes resume exact rather than approximate |
| `errors` | Structured failures with stage, provider, traceback |
| `provider_usage` | Persisted token buckets so quota accounting survives restarts |
| `prompts` | Versioned, hashed prompt templates + the exact `model_id` used |
| `videos` | Rendered output + QC report |
| `thumbnails` | 6 variants per job with per-criterion scores |
| `metadata_candidates` | Title candidates with scores and the `accurate` gate |
| `youtube_uploads` | Upload record incl. `made_for_kids` and `synthetic_disclosed` |
| `analytics` | Time-series snapshots at 1/3/7/14/28 days |
| `strategy` | Learned per-feature scores with Bayesian shrinkage and a prefer/neutral/avoid verdict |

## Design notes

**Why `duration_s` and `duration_hint_s` are separate columns.** The LLM's duration estimate is
useful for planning scene count and pacing before audio exists, but it is always wrong by
20–40%. Keeping both lets you (a) plan, and (b) measure prompt quality by comparing them over
time.

**Why `text_sha256` on `audio`.** Editing one line of dialogue should re-synthesise one clip,
not forty. This column turns re-runs from minutes into milliseconds.

**Why `beat_signature` is a stored column.** Embedding similarity catches paraphrase; the beat
skeleton (`inherit|lack|deceive|confess|repair`) catches "same plot, different animal", which is
the actual failure mode of LLM story generation and the thing that gets channels flagged for
repetitious content.

**Why `provider_usage` is in the DB and not memory.** A crash at 3 a.m. must not reset your
OpenRouter daily counter and burn the next day's allowance.

**Why `licenses` is seeded, not free text.** `qc/policy.py` runs exactly this query before
every publish, and it must be able to fail closed:

```sql
SELECT a.path, l.license_code
FROM assets a LEFT JOIN licenses l ON l.id = a.license_id
WHERE l.id IS NULL OR l.usage_allowed <> 'commercial';
-- any row returned  →  block publish
```

## Example records

See [`scripts/seed_example.sql`](../scripts/seed_example.sql) — a complete worked example
(topic → story → character → scene → dialogue → job → stages → asset → upload row) that has been
inserted and query-verified against the real schema.
