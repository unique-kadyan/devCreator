"""The pipeline as an explicit list of stages, each a pure function of (ctx, job).

Why a hand-written state machine rather than Airflow / Celery / n8n: this pipeline has one
worker, one machine, no network of services, and a hard requirement to resume mid-episode
after a crash. SQLite plus a state column gives that in ~200 lines with no daemon to keep
alive. See docs/00 for the full comparison.

Each stage declares the state it consumes and the state it produces. The runner does the
bookkeeping; stages only do work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .db import jdump, read, tx
from .errors import PolicyViolation, ValidationError
from .logging import get_logger

log = get_logger("stages")

StageFn = Callable[["object", dict], dict]


@dataclass(frozen=True)
class Stage:
    name: str
    from_state: str
    to_state: str
    fn: StageFn
    timeout_key: str = "story"
    retryable: bool = True


REGISTRY: list[Stage] = []


def stage(name: str, from_state: str, to_state: str, timeout_key: str = "story",
          retryable: bool = True):
    def deco(fn: StageFn) -> StageFn:
        REGISTRY.append(Stage(name, from_state, to_state, fn, timeout_key, retryable))
        return fn
    return deco


def by_state(state: str) -> Stage | None:
    return next((s for s in REGISTRY if s.from_state == state), None)


def order() -> list[str]:
    return [s.name for s in REGISTRY]


# ===========================================================================
# Stage implementations
# ===========================================================================

@stage("select_topic", "RESEARCHED", "TOPIC_SELECTED", "story")
def select_topic(ctx, job: dict) -> dict:
    from ..research.scoring import mark_used, select_next
    if job.get("topic_id"):
        return {"topic_id": job["topic_id"]}
    topic = select_next(ctx.db)
    if topic is None:
        raise ValidationError(
            "no unused research topic scores above the threshold. Run `asa research run` "
            "to collect more, or queue one manually with `asa job new --topic ...`.")
    mark_used(ctx.db, topic["id"])
    with tx(ctx.db) as con:
        con.execute("UPDATE jobs SET topic_id = ? WHERE id = ?", (topic["id"], job["id"]))
    return {"topic_id": topic["id"], "topic": topic["topic"]}


@stage("story", "TOPIC_SELECTED", "SCRIPT_GENERATED", "story")
def generate_story(ctx, job: dict) -> dict:
    from ..analytics.feedback import prompt_hints
    from ..characters.factory import slug
    from ..scenes.persist import recent_beat_signatures, save_story
    from ..story.generator import StoryGenerator

    with read(ctx.db) as con:
        row = con.execute("SELECT topic, keywords FROM research_topics WHERE id = ?",
                          (job["topic_id"],)).fetchone()
    topic = row["topic"] if row else "an original animal story"
    keywords = json.loads(row["keywords"]) if row else []

    prefer, avoid = prompt_hints(ctx.db)
    available = ctx.characters.existing()
    gen = StoryGenerator(
        ctx.llm, target_minutes=ctx.target_minutes,
        max_new_characters=int(ctx.cfg.get("story.max_new_characters_per_story", 2)),
        archetypes=ctx.cfg.get("story.archetypes"))
    story = gen.generate(
        topic=topic, keywords=keywords, available_characters=available,
        recent_signatures=recent_beat_signatures(ctx.db),
        existing_locations=_known_locations(ctx.db),
        sfx_library=ctx.sfx.tags() if hasattr(ctx.sfx, "tags") else [],
        strategy_prefer=prefer, strategy_avoid=avoid)

    # Characters must exist before the story row can reference them.
    cast_members, roles = [], {}
    for m in story.outline.cast:
        if m.character_id:
            cast_members.append({"id": m.character_id})
            roles[m.character_id] = m.role
        elif m.new_character_spec:
            cid = slug(m.new_character_spec.name, m.new_character_spec.species)
            cast_members.append({"id": cid, "spec": m.new_character_spec})
            roles[cid] = m.role
    built = ctx.characters.ensure_cast(cast_members)

    story_id = save_story(ctx.db, story, job.get("topic_id"), list(built), roles,
                          est_duration_s=sum(s.duration_hint_s for s in story.scenes.scenes))
    with tx(ctx.db) as con:
        con.execute("UPDATE jobs SET story_id = ? WHERE id = ?", (story_id, job["id"]))
    ctx.characters.bump_appearances(list(built))
    return {"story_id": story_id, "title": story.outline.title,
            "scenes": len(story.scenes.scenes), "cast": list(built),
            "repairs": story.repairs, "models": story.model_ids}


@stage("art", "SCRIPT_GENERATED", "ART_READY", "art")
def generate_art(ctx, job: dict) -> dict:
    from ..scenes.persist import load_scenes
    scenes = load_scenes(ctx.db, job["story_id"])
    made, reused = 0, 0
    for location_id in dict.fromkeys(s["location_id"] for s in scenes):
        prompt = next(s["visual_prompt"] for s in scenes if s["location_id"] == location_id)
        out_dir = ctx.assets / "backgrounds" / location_id
        plate = ctx.images.background(location_id, prompt, out_dir)
        ctx.images.plates(plate.path, ctx.world, out_dir)
        reused += int(plate.cached)
        made += int(not plate.cached)
    with tx(ctx.db) as con:
        con.execute("UPDATE scenes SET status = 'art_ready' WHERE story_id = ?",
                    (job["story_id"],))
    return {"locations": made + reused, "generated": made, "cached": reused}


@stage("audio", "ART_READY", "AUDIO_READY", "voice")
def generate_audio(ctx, job: dict) -> dict:
    from ..media.audio.build import synthesise_story
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    audio = synthesise_story(
        ctx.db, ctx.tts, scenes, story["cast"], paths.audio, ctx.tts_cache(),
        fps=ctx.fps,
        narrator_voice=ctx.cfg.get("providers.tts.kokoro_local.narrator_voice", "bm_fable"))
    ctx.cache_audio(job["id"], audio)
    return {"scenes": len(audio.scenes), "total_s": round(audio.total_s, 2)}


@stage("animate", "AUDIO_READY", "SCENES_RENDERED", "animate")
def animate(ctx, job: dict) -> dict:
    from ..assemble.video import render_scenes
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    audio = ctx.load_audio(job["id"], scenes, story["cast"])
    cast_dirs = {c["character_id"]: ctx.assets / "characters" / c["character_id"]
                 for c in story["cast"]}
    plates = {s["location_id"]: ctx.assets / "backgrounds" / s["location_id"] / "plate.png"
              for s in scenes}
    rendered = render_scenes(
        ctx.db, scenes, {a.scene_id: a for a in audio.scenes}, cast_dirs, plates,
        paths.scenes, ctx.world, ctx.resolution, ctx.fps,
        workers=ctx.cfg.get("production.render_workers"),
        crf=int(ctx.cfg.get("production.crf", 20)),
        species={c["character_id"]: c["species"] for c in story["cast"]})
    return {"scenes": len(rendered),
            "seconds": round(sum(r.stats.get("seconds", 0) for r in rendered), 1)}


@stage("assemble", "SCENES_RENDERED", "VIDEO_RENDERED", "assemble")
def assemble(ctx, job: dict) -> dict:
    from ..assemble.mixdown import assemble_episode
    return assemble_episode(ctx, job)


@stage("subtitles", "VIDEO_RENDERED", "SUBTITLED", "assemble")
def subtitles(ctx, job: dict) -> dict:
    from ..media.subtitles.build import captions_from_audio, write_srt, write_vtt
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    audio = ctx.load_audio(job["id"], scenes, story["cast"])
    offsets, clock = {}, 0.0
    for sa in audio.scenes:
        offsets[sa.scene_id] = clock
        clock += sa.duration_s
    caps = captions_from_audio(
        audio, offsets,
        max_chars_per_line=int(ctx.cfg.get("subtitles.max_chars_per_line", 32)))
    srt = write_srt(caps, paths.out / "captions.srt")
    vtt = write_vtt(caps, paths.out / "captions.vtt")
    with tx(ctx.db) as con:
        con.execute("UPDATE videos SET srt_path = ?, vtt_path = ? WHERE job_id = ?",
                    (str(srt), str(vtt), job["id"]))
    return {"captions": len(caps), "srt": str(srt)}


@stage("thumbnail", "SUBTITLED", "THUMBNAILED", "assemble")
def thumbnail(ctx, job: dict) -> dict:
    from ..publish.thumbnail import generate_set
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    hero = next((c for c in story["cast"] if c["role"] == "protagonist"),
                story["cast"][0] if story["cast"] else None)
    if hero is None:
        raise ValidationError("story has no cast; cannot build a thumbnail")
    # Pick the most visually distinctive scene: the climax, then any interior/exterior with
    # the protagonist staged large.
    plate_scene = max(scenes, key=lambda s: (
        float((s["staging"].get(hero["character_id"]) or {}).get("scale", 0)), s["idx"]))
    plate = ctx.assets / "backgrounds" / plate_scene["location_id"] / "plate.png"
    texts = _thumbnail_texts(ctx.db, job["id"], story)
    variants = generate_set(ctx.db, job["id"], plate,
                            ctx.assets / "characters" / hero["character_id"],
                            texts, paths.thumbs,
                            variants=int(ctx.cfg.get("thumbnail.variants", 6)))
    with tx(ctx.db) as con:
        con.execute("UPDATE videos SET thumbnail_path = ? WHERE job_id = ?",
                    (str(variants[0].path), job["id"]))
    return {"variants": len(variants), "best": variants[0].score,
            "chosen": str(variants[0].path)}


@stage("metadata", "THUMBNAILED", "METADATA_READY", "story")
def metadata(ctx, job: dict) -> dict:
    from ..core.ledger import attribution_block
    from ..publish.metadata import generate
    from ..scenes.persist import load_story
    story = load_story(ctx.db, job["story_id"])
    with read(ctx.db) as con:
        row = con.execute("SELECT duration_s FROM videos WHERE job_id = ?",
                          (job["id"],)).fetchone()
    minutes = (row["duration_s"] if row else ctx.target_minutes * 60) / 60
    attribution = attribution_block(ctx.db, _job_asset_paths(ctx, job))
    meta = generate(ctx.llm, ctx.db, job["id"], story, story["cast"], minutes,
                    ctx.cfg.get("channel.name", "channel"), attribution,
                    bool(ctx.cfg.get("channel.disclose_synthetic", True)),
                    lead_hashtags=ctx.cfg.get("channel.hashtags.lead", []) or [],
                    evergreen_hashtags=ctx.cfg.get("channel.hashtags.evergreen", []) or [],
                    cta=ctx.cfg.get("channel.cta", "") or "")
    ctx.cache_metadata(job["id"], meta)
    return {"title": meta.title, "tags": len(meta.tags),
            "hashtags": len(meta.hashtags),
            "description_chars": len(meta.description)}


@stage("qc", "METADATA_READY", "QC_PASSED", "assemble", retryable=False)
def quality_control(ctx, job: dict) -> dict:
    from ..qc import checks
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    meta = ctx.load_metadata(job["id"])
    with read(ctx.db) as con:
        v = con.execute("SELECT * FROM videos WHERE job_id = ?", (job["id"],)).fetchone()
    if v is None:
        raise ValidationError("no video row for this job")

    # Compare the container against the SCENE TIMELINE, not against the video row - the
    # video row's duration came from probing that same file, so checking one against the
    # other could never fail. The scenes' durations came from the synthesised audio, which
    # is the independent source of truth this check needs.
    planned = sum(sc["duration_s"] or 0.0 for sc in scenes)
    if planned <= 0:
        planned = v["duration_s"]
    report = checks.run(
        ctx.db, Path(v["path"]),
        expected_duration_s=planned,
        srt=Path(v["srt_path"]) if v["srt_path"] else None,
        thumbnail=Path(v["thumbnail_path"]) if v["thumbnail_path"] else None,
        title=meta.title, description=meta.description, tags=meta.tags,
        story=story, scenes=scenes,
        target_lufs=float(ctx.cfg.get("audio.target_lufs", -14.0)),
        true_peak_db=float(ctx.cfg.get("audio.true_peak_db", -1.0)),
        made_for_kids=ctx.cfg.get("channel.made_for_kids"),
        disclose_synthetic=bool(ctx.cfg.get("channel.disclose_synthetic", True)),
        asset_paths=_job_asset_paths(ctx, job))

    # QC is the stage that actually measured the finished file, so it owns the recorded
    # loudness. Leaving it to `assemble` means a job resumed after that stage keeps a
    # stale (or zero) value in a column the dashboard displays as fact.
    measured = next((f.detail.get("lufs") for f in report.findings
                     if f.check == "audio.loudness" and "lufs" in f.detail), None)
    with tx(ctx.db) as con:
        con.execute("UPDATE videos SET qc_report = ?, "
                    "lufs = COALESCE(?, lufs) WHERE job_id = ?",
                    (jdump(report.to_dict()), measured, job["id"]))
    if not report.passed:
        raise PolicyViolation("QC failed: " + "; ".join(
            f.message for f in report.failures)[:600])
    return {"passed": True, "warnings": len(report.warnings),
            "warning_list": [f.check for f in report.warnings]}


@stage("approval", "QC_PASSED", "AWAITING_APPROVAL", "assemble", retryable=False)
def approval(ctx, job: dict) -> dict:
    """Auto-publish is off until a human has approved enough episodes by hand.

    The threshold is a policy decision, not a technical one: a pipeline that publishes
    before anyone has watched its output is exactly the content farm this project is not
    supposed to be.
    """
    auto = bool(ctx.cfg.get("production.auto_publish", False))
    with read(ctx.db) as con:
        approved = con.execute(
            "SELECT COUNT(*) FROM youtube_uploads WHERE approved_by IS NOT NULL"
        ).fetchone()[0]
    threshold = int(ctx.cfg.get("production.auto_publish_after_approvals", 20))
    if auto and approved >= threshold:
        with tx(ctx.db) as con:
            con.execute("UPDATE jobs SET state = 'APPROVED', needs_human = 0 WHERE id = ?",
                        (job["id"],))
        return {"auto_approved": True, "prior_approvals": approved}
    with tx(ctx.db) as con:
        con.execute("UPDATE jobs SET needs_human = 1 WHERE id = ?", (job["id"],))
    ctx.notifier.needs_human(job["id"], "QC passed; waiting for your approval to upload.")
    return {"auto_approved": False, "prior_approvals": approved,
            "threshold": threshold, "waiting": True}


@stage("upload", "APPROVED", "UPLOADED", "upload", retryable=True)
def upload(ctx, job: dict) -> dict:
    from ..publish.youtube import record_upload
    from ..scenes.persist import load_story
    meta = ctx.load_metadata(job["id"])
    with read(ctx.db) as con:
        v = dict(con.execute("SELECT * FROM videos WHERE job_id = ?",
                             (job["id"],)).fetchone())
    made_for_kids = ctx.cfg.get("channel.made_for_kids")
    if made_for_kids is None:
        raise PolicyViolation(
            "channel.made_for_kids is unset. YouTube requires an explicit audience "
            "declaration for every upload; read docs/05-COMPLIANCE.md §2 and set it.")
    privacy = ctx.cfg.get("production.privacy_on_upload", "private")
    try:
        result = ctx.youtube.upload(
            Path(v["path"]), title=meta.title, description=meta.description,
            tags=meta.tags, category_id=int(ctx.cfg.get("channel.category_id", 1)),
            privacy=privacy, made_for_kids=bool(made_for_kids),
            language=ctx.cfg.get("channel.language", "en"),
            thumbnail=Path(v["thumbnail_path"]) if v["thumbnail_path"] else None,
            captions=Path(v["srt_path"]) if v["srt_path"] else None)
    except Exception as e:                                     # noqa: BLE001
        record_upload(ctx.db, job["id"], meta, None, bool(made_for_kids), privacy,
                      error=str(e)[:800])
        raise
    record_upload(ctx.db, job["id"], meta, result, bool(made_for_kids), privacy)
    ctx.notifier.send(f"Uploaded: {meta.title}", result.watch_url, level="info",
                      link=result.watch_url)
    return {"video_id": result.video_id, "privacy": result.privacy_status,
            "url": result.watch_url, "units": result.units_spent}


# ---------------------------------------------------------------- helpers

def _known_locations(db: Path) -> list[str]:
    with read(db) as con:
        return [r[0] for r in con.execute(
            "SELECT id FROM locations ORDER BY uses DESC LIMIT 40")]


def _job_asset_paths(ctx, job: dict) -> list[str]:
    """Every asset this episode actually uses, for the licence audit and the credits."""
    from ..scenes.persist import load_scenes, load_story
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    paths: list[str] = []
    for c in story["cast"]:
        d = ctx.assets / "characters" / c["character_id"]
        paths += [str(p) for p in sorted(d.glob("*.png"))]
    for loc in dict.fromkeys(s["location_id"] for s in scenes):
        paths.append(str(ctx.assets / "backgrounds" / loc / "plate.png"))
    with read(ctx.db) as con:
        paths += [r[0] for r in con.execute(
            "SELECT path FROM audio WHERE kind IN ('music','sfx') AND scene_id IN "
            "(SELECT id FROM scenes WHERE story_id = ?)", (job["story_id"],))]
    return paths


def _thumbnail_texts(db: Path, job_id: int, story: dict) -> list[str]:
    """Short, true phrases drawn from the story itself.

    A thumbnail that promises something the video does not contain is misleading metadata
    under YouTube's policies, so the text can only come from what is already in the story.
    """
    out = [story["title"]]
    hook = (story.get("hook") or "").strip().rstrip(".")
    if hook:
        words = hook.split()
        out.append(" ".join(words[:5]))
    moral = (story.get("moral") or "").strip().rstrip(".")
    if moral:
        out.append(" ".join(moral.split()[:4]))
    return [t for t in dict.fromkeys(out) if len(t) >= 6][:3]
