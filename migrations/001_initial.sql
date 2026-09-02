-- animal-story-automation :: initial schema
-- SQLite 3.35+ (uses STRICT tables, generated columns, RETURNING-friendly design)
-- Apply with: sqlite3 data/asa.db < migrations/001_initial.sql

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ---------------------------------------------------------------- research

CREATE TABLE IF NOT EXISTS research_topics (
    id                  INTEGER PRIMARY KEY,
    topic               TEXT    NOT NULL,
    keywords            TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    primary_animal      TEXT,
    archetype           TEXT,                            -- underdog, trickster, redemption, mystery, friendship, survival, comedy
    source              TEXT    NOT NULL,                -- youtube_search | rss | wikipedia | reddit | seasonal | manual
    source_ref          TEXT,                            -- url / channel id / article title
    trend_score         REAL    NOT NULL DEFAULT 0,
    search_score        REAL    NOT NULL DEFAULT 0,
    competition_score   REAL    NOT NULL DEFAULT 0,      -- higher = LESS competition
    emotional_score     REAL    NOT NULL DEFAULT 0,
    entertainment_score REAL    NOT NULL DEFAULT 0,
    story_score         REAL    NOT NULL DEFAULT 0,
    thumbnail_score     REAL    NOT NULL DEFAULT 0,
    short_form_score    REAL    NOT NULL DEFAULT 0,
    long_form_score     REAL    NOT NULL DEFAULT 0,
    overall_score       REAL    NOT NULL DEFAULT 0,
    embedding           BLOB,                            -- float32 MiniLM vector for dedupe
    date_found          TEXT    NOT NULL DEFAULT (datetime('now')),
    status              TEXT    NOT NULL DEFAULT 'new',  -- new | queued | used | rejected | duplicate
    reject_reason       TEXT,
    notes               TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS ix_topics_status_score ON research_topics(status, overall_score DESC);
CREATE INDEX IF NOT EXISTS ix_topics_animal       ON research_topics(primary_animal);

-- competitor channels watched via free RSS (no quota)
CREATE TABLE IF NOT EXISTS watch_channels (
    id           INTEGER PRIMARY KEY,
    channel_id   TEXT NOT NULL UNIQUE,
    title        TEXT,
    added_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_polled  TEXT,
    active       INTEGER NOT NULL DEFAULT 1
) STRICT;

-- ---------------------------------------------------------------- characters

CREATE TABLE IF NOT EXISTS characters (
    id                  TEXT PRIMARY KEY,                -- slug, e.g. 'milo_fox'
    name                TEXT NOT NULL,
    species             TEXT NOT NULL,
    age_band            TEXT,                            -- child | teen | young_adult | adult | elder
    presentation        TEXT,                            -- free text; used for voice + pronoun selection
    pronouns            TEXT NOT NULL DEFAULT 'they/them',
    appearance          TEXT NOT NULL,
    palette             TEXT NOT NULL DEFAULT '{}',      -- JSON {fur:'#E07A35', chest:'#FFF6E8', eyes:'#4C9A54'}
    clothing            TEXT,
    accessories         TEXT NOT NULL DEFAULT '[]',      -- JSON array
    personality         TEXT NOT NULL,
    backstory           TEXT,
    voice_id            TEXT NOT NULL,                   -- Kokoro preset id
    voice_pitch_semi    REAL NOT NULL DEFAULT 0,
    voice_rate          REAL NOT NULL DEFAULT 1.0,
    voice_notes         TEXT,
    style_hash          TEXT,                            -- hash of the style-bible block used to draw it
    puppet_dir          TEXT,                            -- assets/characters/<id>/
    rig_json            TEXT,                            -- assets/characters/<id>/rig.json
    reference_image     TEXT,                            -- turnaround sheet path
    status              TEXT NOT NULL DEFAULT 'draft',   -- draft | needs_rig | ready | retired
    appearances         INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;
CREATE INDEX IF NOT EXISTS ix_char_status ON characters(status);

CREATE TABLE IF NOT EXISTS character_relationships (
    id            INTEGER PRIMARY KEY,
    character_id  TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    other_id      TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    relation      TEXT NOT NULL,                          -- sibling | friend | rival | mentor | parent | neighbour
    notes         TEXT,
    UNIQUE(character_id, other_id, relation)
) STRICT;

-- reusable locations (background plates are cached against these)
CREATE TABLE IF NOT EXISTS locations (
    id            TEXT PRIMARY KEY,                       -- 'forest_village'
    name          TEXT NOT NULL,
    description   TEXT NOT NULL,
    visual_prompt TEXT NOT NULL,
    plate_dir     TEXT,                                   -- assets/backgrounds/<id>/
    layers        TEXT NOT NULL DEFAULT '[]',             -- JSON ['far.png','mid.png','near.png']
    uses          INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

-- ---------------------------------------------------------------- stories

CREATE TABLE IF NOT EXISTS stories (
    id               INTEGER PRIMARY KEY,
    topic_id         INTEGER REFERENCES research_topics(id),
    title            TEXT NOT NULL,
    hook             TEXT NOT NULL,
    logline          TEXT NOT NULL,
    target_audience  TEXT NOT NULL,
    genre            TEXT NOT NULL,
    archetype        TEXT NOT NULL,
    moral            TEXT NOT NULL,
    setting          TEXT NOT NULL,
    beat_beginning   TEXT NOT NULL,
    beat_conflict    TEXT NOT NULL,
    beat_rising      TEXT NOT NULL,
    beat_climax      TEXT NOT NULL,
    beat_resolution  TEXT NOT NULL,
    ending           TEXT NOT NULL,
    beat_signature   TEXT NOT NULL DEFAULT '',            -- verb/role skeleton for dedupe
    embedding        BLOB,
    est_duration_s   REAL,
    word_count       INTEGER,
    model_id         TEXT,                                -- exact LLM id used
    prompt_id        INTEGER REFERENCES prompts(id),
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;
CREATE INDEX IF NOT EXISTS ix_stories_archetype ON stories(archetype);

CREATE TABLE IF NOT EXISTS story_cast (
    story_id     INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    character_id TEXT    NOT NULL REFERENCES characters(id),
    role         TEXT    NOT NULL,                        -- protagonist | antagonist | ally | mentor | comic_relief
    PRIMARY KEY (story_id, character_id)
) STRICT;

CREATE TABLE IF NOT EXISTS scenes (
    id               INTEGER PRIMARY KEY,
    story_id         INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    idx              INTEGER NOT NULL,
    location_id      TEXT REFERENCES locations(id),
    duration_hint_s  REAL,
    duration_s       REAL,                                -- authoritative, computed from audio
    action           TEXT NOT NULL,
    narration        TEXT,
    emotion          TEXT,
    shot             TEXT NOT NULL DEFAULT 'medium',
    camera_move      TEXT NOT NULL DEFAULT 'static',
    camera_from      TEXT,
    camera_to        TEXT,
    transition_in    TEXT NOT NULL DEFAULT 'cut',
    staging          TEXT NOT NULL DEFAULT '{}',          -- JSON per-character x/y/scale/facing
    visual_prompt    TEXT,
    sfx_tags         TEXT NOT NULL DEFAULT '[]',          -- JSON array
    music_cue        TEXT,
    render_path      TEXT,
    render_sha256    TEXT,
    status           TEXT NOT NULL DEFAULT 'planned',     -- planned | art_ready | audio_ready | rendered | failed
    UNIQUE(story_id, idx)
) STRICT;

CREATE TABLE IF NOT EXISTS dialogue (
    id           INTEGER PRIMARY KEY,
    scene_id     INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    character_id TEXT REFERENCES characters(id),          -- NULL = narrator
    line         TEXT NOT NULL,
    emotion      TEXT,
    UNIQUE(scene_id, idx)
) STRICT;

-- ---------------------------------------------------------------- assets & licensing

CREATE TABLE IF NOT EXISTS licenses (
    id                   INTEGER PRIMARY KEY,
    license_code         TEXT NOT NULL,                   -- CC0 | CC-BY | YT-AUDIO-LIB | PIXABAY | APACHE-2.0 | PROPRIETARY-OK
    attribution_required INTEGER NOT NULL DEFAULT 0,
    attribution_text     TEXT,
    usage_allowed        TEXT NOT NULL DEFAULT 'commercial', -- commercial | noncommercial | unknown
    license_url          TEXT,
    notes                TEXT,
    UNIQUE(license_code)
) STRICT;

CREATE TABLE IF NOT EXISTS assets (
    id             INTEGER PRIMARY KEY,
    kind           TEXT NOT NULL,                         -- background | character_layer | sfx | music | font | plate | thumbnail | intro | outro
    path           TEXT NOT NULL UNIQUE,
    sha256         TEXT NOT NULL,
    source         TEXT NOT NULL,                         -- huggingface | pollinations | freesound | pixabay | yt_audio_library | local | procedural
    source_ref     TEXT,                                  -- url or model id
    license_id     INTEGER REFERENCES licenses(id),
    attribution    TEXT,
    download_date  TEXT NOT NULL DEFAULT (datetime('now')),
    usage_allowed  INTEGER NOT NULL DEFAULT 1,
    meta           TEXT NOT NULL DEFAULT '{}',            -- JSON: dimensions, duration, model, seed, prompt hash
    bytes          INTEGER
) STRICT;
CREATE INDEX IF NOT EXISTS ix_assets_kind ON assets(kind);
CREATE INDEX IF NOT EXISTS ix_assets_sha  ON assets(sha256);

CREATE TABLE IF NOT EXISTS audio (
    id            INTEGER PRIMARY KEY,
    scene_id      INTEGER REFERENCES scenes(id) ON DELETE CASCADE,
    dialogue_id   INTEGER REFERENCES dialogue(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,                          -- narration | dialogue | sfx | music | mixdown
    character_id  TEXT REFERENCES characters(id),
    path          TEXT NOT NULL,
    duration_s    REAL NOT NULL,
    lufs          REAL,
    provider      TEXT,                                   -- kokoro | piper | library | freesound
    voice_id      TEXT,
    text_sha256   TEXT,                                   -- lets us skip re-synthesis on unchanged text
    word_timings  TEXT,                                   -- JSON [[word,start,end],...] when aligned
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;
CREATE INDEX IF NOT EXISTS ix_audio_scene ON audio(scene_id);

-- ---------------------------------------------------------------- jobs

CREATE TABLE IF NOT EXISTS jobs (
    id             INTEGER PRIMARY KEY,
    story_id       INTEGER REFERENCES stories(id),
    topic_id       INTEGER REFERENCES research_topics(id),
    state          TEXT NOT NULL DEFAULT 'RESEARCHED',
    -- RESEARCHED TOPIC_SELECTED SCRIPT_GENERATED CHARACTERS_READY SCENES_PLANNED ART_READY
    -- AUDIO_READY SCENES_RENDERED VIDEO_RENDERED SUBTITLED THUMBNAILED METADATA_READY
    -- QC_PASSED QC_FLAGGED AWAITING_APPROVAL APPROVED REJECTED UPLOADED PUBLISHED
    -- ANALYZED FAILED QUOTA_BLOCKED
    format         TEXT NOT NULL DEFAULT 'long',          -- long | short
    needs_human    INTEGER NOT NULL DEFAULT 0,
    attempts       INTEGER NOT NULL DEFAULT 0,
    lease_owner    TEXT,
    lease_expires  TEXT,
    retry_after    TEXT,
    work_dir       TEXT,
    out_dir        TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at    TEXT
) STRICT;
CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(state, retry_after);

CREATE TABLE IF NOT EXISTS job_stages (
    id           INTEGER PRIMARY KEY,
    job_id       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',         -- pending | running | done | failed | skipped
    attempts     INTEGER NOT NULL DEFAULT 0,
    started_at   TEXT,
    finished_at  TEXT,
    duration_s   REAL,
    output_ref   TEXT,
    UNIQUE(job_id, stage)
) STRICT;

CREATE TABLE IF NOT EXISTS errors (
    id         INTEGER PRIMARY KEY,
    job_id     INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    stage      TEXT,
    provider   TEXT,
    kind       TEXT NOT NULL,                             -- quota | network | validation | render | policy | unknown
    message    TEXT NOT NULL,
    traceback  TEXT,
    payload    TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;
CREATE INDEX IF NOT EXISTS ix_errors_job ON errors(job_id, created_at);

-- provider quota accounting, survives restarts
CREATE TABLE IF NOT EXISTS provider_usage (
    id            INTEGER PRIMARY KEY,
    provider      TEXT NOT NULL,
    window_key    TEXT NOT NULL,                          -- e.g. '2026-09-01' or '2026-09-01T14:32'
    units_used    REAL NOT NULL DEFAULT 0,
    requests      INTEGER NOT NULL DEFAULT 0,
    cold_until    TEXT,
    UNIQUE(provider, window_key)
) STRICT;

-- ---------------------------------------------------------------- prompts

CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,                            -- story.outline | scene.breakdown | metadata.titles
    version     INTEGER NOT NULL DEFAULT 1,
    template    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    model_id    TEXT,
    params      TEXT NOT NULL DEFAULT '{}',               -- JSON temperature/top_p/max_tokens
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, version)
) STRICT;

-- ---------------------------------------------------------------- video & publishing

CREATE TABLE IF NOT EXISTS videos (
    id             INTEGER PRIMARY KEY,
    job_id         INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    story_id       INTEGER NOT NULL REFERENCES stories(id),
    path           TEXT NOT NULL,
    sha256         TEXT,
    duration_s     REAL NOT NULL,
    width          INTEGER NOT NULL DEFAULT 1920,
    height         INTEGER NOT NULL DEFAULT 1080,
    fps            REAL    NOT NULL DEFAULT 24,
    bytes          INTEGER,
    lufs           REAL,
    srt_path       TEXT,
    vtt_path       TEXT,
    thumbnail_path TEXT,
    qc_report      TEXT NOT NULL DEFAULT '{}',            -- JSON
    render_seconds REAL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE TABLE IF NOT EXISTS thumbnails (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    variant     INTEGER NOT NULL,
    path        TEXT NOT NULL,
    text_used   TEXT,
    expression  TEXT,
    layout      TEXT,
    score       REAL,
    scores      TEXT NOT NULL DEFAULT '{}',               -- JSON per-criterion
    chosen      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(job_id, variant)
) STRICT;

CREATE TABLE IF NOT EXISTS metadata_candidates (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'title',
    value       TEXT NOT NULL,
    score       REAL,
    scores      TEXT NOT NULL DEFAULT '{}',
    accurate    INTEGER NOT NULL DEFAULT 1,
    chosen      INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE TABLE IF NOT EXISTS youtube_uploads (
    id                 INTEGER PRIMARY KEY,
    job_id             INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    video_id           TEXT UNIQUE,
    title              TEXT NOT NULL,
    description        TEXT NOT NULL,
    tags               TEXT NOT NULL DEFAULT '[]',
    category_id        INTEGER,                           -- 1 = Film & Animation
    playlist_id        TEXT,
    privacy_status     TEXT NOT NULL DEFAULT 'private',   -- private | unlisted | public
    publish_at         TEXT,
    made_for_kids      INTEGER,                           -- must be set explicitly
    synthetic_disclosed INTEGER NOT NULL DEFAULT 0,
    thumbnail_set      INTEGER NOT NULL DEFAULT 0,
    captions_set       INTEGER NOT NULL DEFAULT 0,
    upload_status      TEXT NOT NULL DEFAULT 'pending',   -- pending | uploading | uploaded | failed
    api_response       TEXT,
    error              TEXT,
    approved_by        TEXT,
    approved_at        TEXT,
    uploaded_at        TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS analytics (
    id                      INTEGER PRIMARY KEY,
    video_id                TEXT NOT NULL,
    snapshot_date           TEXT NOT NULL,
    days_since_publish      INTEGER NOT NULL,
    views                   INTEGER NOT NULL DEFAULT 0,
    minutes_watched         REAL    NOT NULL DEFAULT 0,
    avg_view_duration_s     REAL,
    avg_view_percentage     REAL,
    impressions             INTEGER,
    ctr                     REAL,
    likes                   INTEGER NOT NULL DEFAULT 0,
    dislikes                INTEGER,
    comments                INTEGER NOT NULL DEFAULT 0,
    shares                  INTEGER NOT NULL DEFAULT 0,
    subscribers_gained      INTEGER NOT NULL DEFAULT 0,
    subscribers_lost        INTEGER NOT NULL DEFAULT 0,
    traffic_sources         TEXT NOT NULL DEFAULT '{}',   -- JSON {BROWSE:0.4,SUGGESTED:0.3,...}
    retention_curve         TEXT,                         -- JSON [[pct_through, pct_watching],...]
    UNIQUE(video_id, snapshot_date)
) STRICT;

-- learned weights / preferences produced by analytics/analyze.py
CREATE TABLE IF NOT EXISTS strategy (
    id          INTEGER PRIMARY KEY,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    feature     TEXT NOT NULL,                            -- animal | archetype | hook_type | duration_bucket | title_style | thumb_style | voice | emotion
    value       TEXT NOT NULL,
    n           INTEGER NOT NULL,
    raw_mean    REAL NOT NULL,
    shrunk      REAL NOT NULL,                            -- Bayesian-shrunk score
    verdict     TEXT NOT NULL DEFAULT 'neutral',          -- prefer | neutral | avoid
    UNIQUE(computed_at, feature, value)
) STRICT;

-- ---------------------------------------------------------------- seed licence rows

INSERT OR IGNORE INTO licenses (license_code, attribution_required, attribution_text, usage_allowed, license_url) VALUES
 ('CC0',           0, NULL, 'commercial', 'https://creativecommons.org/publicdomain/zero/1.0/'),
 ('CC-BY',         1, NULL, 'commercial', 'https://creativecommons.org/licenses/by/4.0/'),
 ('CC-BY-NC',      1, NULL, 'noncommercial', 'https://creativecommons.org/licenses/by-nc/4.0/'),
 ('YT-AUDIO-LIB',  0, NULL, 'commercial', 'https://www.youtube.com/audiolibrary'),
 ('PIXABAY',       0, NULL, 'commercial', 'https://pixabay.com/service/license-summary/'),
 ('APACHE-2.0',    0, NULL, 'commercial', 'https://www.apache.org/licenses/LICENSE-2.0'),
 ('OPENRAIL-PP-M', 0, NULL, 'commercial', 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md'),
 ('UNKNOWN',       1, NULL, 'unknown', NULL);
