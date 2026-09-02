-- One video row per job.
--
-- The assemble stage upserted with ON CONFLICT(id), but `id` is an autoincrement primary
-- key, so the conflict target could never fire: re-running assemble (after a crash, or
-- after `asa job retry`) inserted a SECOND row for the same job. The subtitle and
-- thumbnail stages then updated both, and QC's `SELECT ... WHERE job_id = ?` picked
-- whichever came back first. Making job_id unique turns that into a real upsert.

-- Deduplicate any rows an earlier run already created, keeping the newest.
DELETE FROM videos
WHERE id NOT IN (SELECT MAX(id) FROM videos GROUP BY job_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_job ON videos(job_id);

-- Supporting indexes for the queries the runner and dashboard make on every tick.
CREATE INDEX IF NOT EXISTS idx_jobs_state       ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_retry       ON jobs(retry_after);
CREATE INDEX IF NOT EXISTS idx_scenes_story     ON scenes(story_id, idx);
CREATE INDEX IF NOT EXISTS idx_dialogue_scene   ON dialogue(scene_id, idx);
CREATE INDEX IF NOT EXISTS idx_audio_scene      ON audio(scene_id);
CREATE INDEX IF NOT EXISTS idx_topics_status    ON research_topics(status, overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_errors_job       ON errors(job_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_video  ON analytics(video_id, snapshot_date DESC);
