-- Adaptive model routing: observed health per free model, so selection learns.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_health (
    id                   INTEGER PRIMARY KEY,
    provider             TEXT NOT NULL,
    model_id             TEXT NOT NULL,
    role                 TEXT NOT NULL DEFAULT 'any',   -- any | story | cheap | structured
    calls                INTEGER NOT NULL DEFAULT 0,
    successes            INTEGER NOT NULL DEFAULT 0,
    rate_limits          INTEGER NOT NULL DEFAULT 0,
    errors               INTEGER NOT NULL DEFAULT 0,
    schema_failures      INTEGER NOT NULL DEFAULT 0,    -- returned unparseable/invalid JSON
    empty_returns        INTEGER NOT NULL DEFAULT 0,
    total_latency_s      REAL    NOT NULL DEFAULT 0,
    total_out_tokens     INTEGER NOT NULL DEFAULT 0,
    context_length       INTEGER,
    cold_until           TEXT,                          -- skip until this time
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_ok              TEXT,
    last_error           TEXT,
    first_seen           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(provider, model_id, role)
) STRICT;
CREATE INDEX IF NOT EXISTS ix_model_health_pick ON model_health(provider, role, cold_until);

-- cached discovery of the live free-model roster
CREATE TABLE IF NOT EXISTS model_catalog (
    id              INTEGER PRIMARY KEY,
    provider        TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    context_length  INTEGER,
    is_free         INTEGER NOT NULL DEFAULT 1,
    meta            TEXT NOT NULL DEFAULT '{}',
    refreshed_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(provider, model_id)
) STRICT;
