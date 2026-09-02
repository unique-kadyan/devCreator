"""Persist a GeneratedStory into the relational schema.

The DB, not the in-memory object, is the pipeline's source of truth: a job that dies during
rendering must be resumable from SQLite alone. Everything the later stages need - scene
order, staging, camera, dialogue, cast - lands here in one transaction.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.db import jdump, read, tx
from ..core.logging import get_logger

log = get_logger("persist")


def save_story(db: Path, gen, topic_id: int | None, cast_ids: list[str],
               roles: dict[str, str], est_duration_s: float | None = None) -> int:
    o = gen.outline
    with tx(db) as con:
        cur = con.execute("""
            INSERT INTO stories (topic_id, title, hook, logline, target_audience, genre,
                archetype, moral, setting, beat_beginning, beat_conflict, beat_rising,
                beat_climax, beat_resolution, ending, beat_signature, est_duration_s,
                word_count, model_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (topic_id, o.title, o.hook, o.logline, o.target_audience, o.genre,
              o.archetype, o.moral, o.setting, o.beats.beginning, o.beats.conflict,
              o.beats.rising, o.beats.climax, o.beats.resolution, o.ending,
              o.beat_signature, est_duration_s, gen.word_count,
              gen.model_ids.get("scenes") or gen.model_ids.get("outline")))
        story_id = cur.lastrowid
        con.executemany(
            "INSERT OR IGNORE INTO story_cast (story_id, character_id, role) VALUES (?,?,?)",
            [(story_id, cid, roles.get(cid, "ally")) for cid in cast_ids])

        # Locations must exist before scenes can reference them. The location is part of
        # the story PLAN - the art stage only adds the plate later - so creating the row
        # here is not a workaround for the foreign key, it is where the row belongs.
        seen: set[str] = set()
        for sc in gen.scenes.scenes:
            if sc.location_id in seen:
                continue
            seen.add(sc.location_id)
            con.execute("""
                INSERT INTO locations (id, name, description, visual_prompt)
                VALUES (?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    visual_prompt = CASE WHEN locations.plate_dir IS NULL
                                         THEN excluded.visual_prompt
                                         ELSE locations.visual_prompt END
            """, (sc.location_id, sc.location_id.replace("_", " ").title(),
                  sc.visual_prompt[:400], sc.visual_prompt))

        for sc in gen.scenes.scenes:
            cur = con.execute("""
                INSERT INTO scenes (story_id, idx, location_id, duration_hint_s, action,
                    narration, emotion, shot, camera_move, camera_from, camera_to,
                    transition_in, staging, visual_prompt, sfx_tags, music_cue, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'planned')
            """, (story_id, sc.index, sc.location_id, sc.duration_hint_s, sc.action,
                  sc.narration, sc.emotion, sc.shot, sc.camera.move, sc.camera.from_shot,
                  sc.camera.to_shot, sc.transition_in,
                  jdump({k: v.model_dump() for k, v in sc.staging.items()}),
                  sc.visual_prompt, jdump(sc.sfx), sc.music_cue))
            scene_id = cur.lastrowid
            rows = []
            idx = 0
            if sc.narration.strip():
                rows.append((scene_id, idx, None, sc.narration.strip(), sc.emotion))
                idx += 1
            for d in sc.dialogue:
                rows.append((scene_id, idx, d.character_id, d.line, d.emotion))
                idx += 1
            if rows:
                con.executemany(
                    "INSERT INTO dialogue (scene_id, idx, character_id, line, emotion) "
                    "VALUES (?,?,?,?,?)", rows)
    log.info("story_saved", story_id=story_id, scenes=len(gen.scenes.scenes),
             cast=len(cast_ids))
    return story_id


def load_scenes(db: Path, story_id: int) -> list[dict]:
    """Everything the art / audio / animate stages read, in one query each."""
    with read(db) as con:
        scenes = [dict(r) for r in con.execute(
            "SELECT * FROM scenes WHERE story_id = ? ORDER BY idx", (story_id,))]
        for s in scenes:
            s["staging"] = json.loads(s["staging"] or "{}")
            s["sfx_tags"] = json.loads(s["sfx_tags"] or "[]")
            s["dialogue"] = [dict(r) for r in con.execute(
                "SELECT * FROM dialogue WHERE scene_id = ? ORDER BY idx", (s["id"],))]
    return scenes


def load_story(db: Path, story_id: int) -> dict:
    with read(db) as con:
        row = con.execute("SELECT * FROM stories WHERE id = ?", (story_id,)).fetchone()
        if row is None:
            raise KeyError(f"story {story_id} not found")
        story = dict(row)
        story["cast"] = [dict(r) for r in con.execute(
            "SELECT sc.character_id, sc.role, c.name, c.species, c.voice_id, "
            "       c.voice_pitch_semi, c.voice_rate, c.puppet_dir "
            "FROM story_cast sc JOIN characters c ON c.id = sc.character_id "
            "WHERE sc.story_id = ?", (story_id,))]
    return story


def recent_beat_signatures(db: Path, limit: int = 25) -> list[str]:
    with read(db) as con:
        return [r[0] for r in con.execute(
            "SELECT beat_signature FROM stories WHERE beat_signature <> '' "
            "ORDER BY id DESC LIMIT ?", (limit,))]


def set_scene_duration(db: Path, scene_id: int, duration_s: float, status: str) -> None:
    with tx(db) as con:
        con.execute("UPDATE scenes SET duration_s = ?, status = ? WHERE id = ?",
                    (duration_s, status, scene_id))


def set_scene_render(db: Path, scene_id: int, path: str, sha: str) -> None:
    with tx(db) as con:
        con.execute("UPDATE scenes SET render_path = ?, render_sha256 = ?, "
                    "status = 'rendered' WHERE id = ?", (path, sha, scene_id))
