-- One synthesised clip per dialogue line, enforced by the schema.
--
-- `_record_audio` INSERTed unconditionally, and nothing stopped a second row for the same
-- line. That matters because the audio stage is re-runnable by design: `ctx.load_audio`
-- rebuilds the whole timeline from this table (stages run in separate processes, so
-- nothing may be passed in memory), and after one retry every line appeared twice. Scene
-- durations doubled, the shot list stretched to fill them, subtitles drifted, and the
-- assembled video ran 197s against a 119s soundtrack. QC caught the drift only at the very
-- end, after two full re-renders had been paid for.
--
-- Deduplicate first, keeping the newest row per line - it names the file the most recent
-- synthesis actually wrote - then make the duplicate impossible.
DELETE FROM audio
 WHERE dialogue_id IS NOT NULL
   AND id NOT IN (SELECT MAX(id) FROM audio
                   WHERE dialogue_id IS NOT NULL
                   GROUP BY scene_id, dialogue_id);

-- Partial: sfx, music and mixdown rows carry no dialogue_id and may repeat freely.
CREATE UNIQUE INDEX IF NOT EXISTS ux_audio_line
    ON audio(scene_id, dialogue_id) WHERE dialogue_id IS NOT NULL;
