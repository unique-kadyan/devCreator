-- Multi-part episodes: one story becomes several released videos.
--
-- Parts are a dimension of the *output*, not of the story. The story, its scenes, its cast
-- and its audio stay single rows keyed by story_id - splitting happens at assemble time
-- from measured scene durations, because that is the first point where a real duration
-- exists. Only the artefacts downstream of that need a part number.
--
-- Migrations 003 and 004 made videos and youtube_uploads one row per job. That was right
-- when a job produced one video; it is now one row per (job, part), and the unique indexes
-- have to widen or the second part silently upserts over the first.

ALTER TABLE videos ADD COLUMN part INTEGER NOT NULL DEFAULT 1;
ALTER TABLE videos ADD COLUMN part_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE youtube_uploads ADD COLUMN part INTEGER NOT NULL DEFAULT 1;

DROP INDEX IF EXISTS idx_videos_job;
CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_job_part ON videos(job_id, part);

DROP INDEX IF EXISTS idx_youtube_uploads_job;
CREATE UNIQUE INDEX IF NOT EXISTS idx_youtube_uploads_job_part
    ON youtube_uploads(job_id, part);

-- Which scenes ended up in which part. Written by assemble, read by anything that needs to
-- explain a cut after the fact (QC, the dashboard, a re-render of one part only).
CREATE TABLE IF NOT EXISTS video_parts (
    id          INTEGER PRIMARY KEY,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    part        INTEGER NOT NULL,
    scene_from  INTEGER NOT NULL,
    scene_to    INTEGER NOT NULL,
    duration_s  REAL NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_video_parts_job_part ON video_parts(job_id, part);
