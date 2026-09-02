-- One upload row per job.
--
-- Same class of bug migration 003 fixed for `videos`, in a form that hides better.
-- record_upload() upserts with ON CONFLICT(video_id). video_id IS declared UNIQUE, so the
-- conflict target is legal and no error is ever raised - but a FAILED upload records
-- video_id = NULL, and SQLite treats NULLs as distinct from each other under a unique
-- constraint. Two failures for one job therefore never conflict, and each one inserts a
-- fresh row instead of updating the previous attempt.
--
-- The damage is not the wasted rows, it is the reads. Anything doing
-- `SELECT ... FROM youtube_uploads WHERE job_id = ?` and taking the first result can pick
-- up a stale failure row and conclude the upload failed while the video is live on the
-- channel - which is exactly what happened while verifying job 1, whose two rows are the
-- ones this migration collapses.

-- Keep the successful row for any job that has one, otherwise the newest attempt.
DELETE FROM youtube_uploads
WHERE id NOT IN (
    SELECT MAX(id) FROM youtube_uploads
    WHERE video_id IS NOT NULL
    GROUP BY job_id
    UNION
    SELECT MAX(id) FROM youtube_uploads
    WHERE job_id NOT IN (SELECT job_id FROM youtube_uploads WHERE video_id IS NOT NULL)
    GROUP BY job_id
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_youtube_uploads_job ON youtube_uploads(job_id);
