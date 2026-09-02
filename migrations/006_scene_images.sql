-- Per-scene generated frames for the cinematic render mode.
--
-- The puppet path stores one plate per LOCATION under assets/backgrounds/<location_id>/,
-- because the plate is an empty stage reused by every scene set there. The cinematic path
-- generates a finished frame per SCENE - the characters are in the picture, so two moments
-- in the same classroom are two different images - and there is nowhere on `scenes` to
-- record which file belongs to which scene.

ALTER TABLE scenes ADD COLUMN plate_path TEXT;
