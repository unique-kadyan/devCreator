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
    from ..research.scoring import DEFAULT_MIN_SCORE, mark_used, near_miss, select_next
    if job.get("topic_id"):
        return {"topic_id": job["topic_id"]}
    floor = float(ctx.cfg.get("research.min_score", DEFAULT_MIN_SCORE))
    topic = select_next(ctx.db, min_score=floor)
    if topic is None:
        # Say which of the two situations this is. "Nothing scores above the threshold"
        # reads as an empty table and sends the operator to collect more, which does not
        # help at all when the table is full and the floor is rejecting every row in it.
        best = near_miss(ctx.db, min_score=floor)
        detail = (f" The best rejected topic scores {best['overall_score']:.3f}: "
                  f"{best['topic'][:90]!r}. Lower `research.min_score` below that to take "
                  f"it - read what else sits under the floor first."
                  if best else " The table holds no unused topics at all.")
        raise ValidationError(
            f"no unused research topic scores at or above {floor:.2f}.{detail} "
            f"Run `asa research` to collect more, or queue one manually with "
            f"`asa job new --topic ...`.")
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
        row = con.execute("SELECT topic, keywords, source FROM research_topics WHERE id = ?",
                          (job["topic_id"],)).fetchone()
    topic = row["topic"] if row else "an original animal story"
    keywords = json.loads(row["keywords"]) if row else []
    # A collected topic is a seed to develop. A topic a person typed is a commission - an
    # ad, a requested episode - and the script has to say what it says, so it is passed as
    # a binding brief as well (story/prompts.brief_block).
    brief = topic if (row and row["source"] == "manual") else ""

    prefer, avoid = prompt_hints(ctx.db)
    available = ctx.characters.existing()
    gen = StoryGenerator(
        ctx.llm, target_minutes=ctx.target_minutes,
        max_new_characters=int(ctx.cfg.get("story.max_new_characters_per_story", 2)),
        archetypes=ctx.cfg.get("story.archetypes"),
        language=str(ctx.cfg.get("channel.language", "en")),
        subjects=ctx.cfg.get("story.subjects"))
    story = gen.generate(
        topic=topic, keywords=keywords, available_characters=available,
        recent_signatures=recent_beat_signatures(ctx.db),
        existing_locations=_known_locations(ctx.db),
        sfx_library=ctx.sfx.tags() if hasattr(ctx.sfx, "tags") else [],
        strategy_prefer=prefer, strategy_avoid=avoid, brief=brief)

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


def _cinematic(ctx) -> bool:
    return str(ctx.cfg.get("production.render_mode", "puppet")).lower() == "cinematic"


def _performance(ctx) -> bool:
    """Whether cinematic scenes are cut into shots and performed, or held as one still.

    Defaults ON. The one-still-per-scene behaviour it replaces is the reason a finished
    episode read as a photograph with a voice-over, so it is kept only as an escape hatch
    for debugging the image stage without paying for a shot list.
    """
    return _cinematic(ctx) and bool(
        ctx.cfg.get("production.performance.enabled", True))


@stage("art", "SCRIPT_GENERATED", "ART_READY", "art")
def generate_art(ctx, job: dict) -> dict:
    from ..scenes.persist import load_scenes
    scenes = load_scenes(ctx.db, job["story_id"])
    if _cinematic(ctx):
        return _generate_scene_images(ctx, job, scenes)
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


def _generate_scene_images(ctx, job: dict, scenes: list[dict]) -> dict:
    """The pictures a cinematic episode is cut from.

    With `production.performance.enabled` this buys one image per camera SETUP rather than
    one per scene: a close-up of whoever is speaking, a reverse for whoever answers, a wide
    for the narration. That is what lets the animate stage cut on every line instead of
    holding one photograph for thirty seconds while voices play over it.

    Several shots share a setup (`shotlist.image_key`), and the image cache is keyed by
    prompt, so a six-line exchange between two characters cuts six times on two
    generations. `max_images_per_scene` is the ceiling when a scene is unusually talkative.
    """
    from ..assemble.shot_render import generate_shot_images, shot_image_name
    from ..assemble.shotlist import image_keys, plan_shots
    from ..media.images.scene_image import (negative_for, period_hint, scene_prompt,
                                            styled_for)
    from ..scenes.persist import load_story
    story = load_story(ctx.db, job["story_id"])
    cast_by_id = {c["character_id"]: c for c in story["cast"]}
    out_dir = ctx.paths_for(job["id"]).work / "scene_images"
    # Where and WHEN. A story that declared a period replaces the channel's contemporary
    # region hint with it rather than adding to it: "set in India, Indian street furniture"
    # and "Alexandria, 3rd century BC" cannot both be true, and an image model handed both
    # averages them into a place that never existed.
    region = period_hint(story.get("period")) or ctx.cfg.get("channel.region_hint") or None
    if story.get("period"):
        log.info("period_overrides_region_hint", story=job["story_id"],
                 period=str(story["period"])[:80])
    # None (key absent) keeps the module default; an explicit empty string in config is a
    # deliberate "no grade", so `or None` would be wrong here.
    look = ctx.cfg.get("channel.look_hint")
    # The medium: `cartoon` or `photoreal`. It decides the subject clause AND the negative,
    # which is why it is read once here and handed to every prompt this stage builds.
    art_style = ctx.cfg.get("channel.art_style")
    performance = _performance(ctx)
    max_images = int(ctx.cfg.get("production.performance.max_images_per_scene", 6))

    made = reused = shots = 0
    for sc in scenes:
        if performance:
            result = generate_shot_images(ctx, sc, cast_by_id, out_dir,
                                          max_images=max_images, region_hint=region,
                                          look_hint=look, art_style=art_style)
            made += result["generated"]
            reused += result["cached"]
            shots += result["shots"]
            # `plate_path` keeps pointing at the scene's opening setup: it is what the rest
            # of the pipeline means by "the picture of this scene".
            first = image_keys(plan_shots(sc, max_images=max_images))[0]
            plate_path = out_dir / shot_image_name(sc["idx"], first)
        else:
            kwargs = {"region_hint": region} if region else {}
            prompt = scene_prompt(sc, cast_by_id,
                                  style=styled_for(ctx.images.size, look, art_style),
                                  **kwargs)
            plate = ctx.images.scene(sc["idx"], prompt, out_dir, negative_for(art_style))
            reused += int(plate.cached)
            made += int(not plate.cached)
            plate_path = plate.path
        with tx(ctx.db) as con:
            con.execute("UPDATE scenes SET status = 'art_ready', plate_path = ? "
                        "WHERE id = ?", (str(plate_path), sc["id"]))
    log.info("scene_images_ready", scenes=len(scenes), shots=shots, generated=made,
             cached=reused, mode="performance" if performance else "one_per_scene")
    return {"mode": "cinematic", "scenes": len(scenes), "shots": shots,
            "generated": made, "cached": reused}


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


def _animate_performance(ctx, job: dict, scenes: list[dict], audio) -> dict:
    """Cut each scene into shots and perform them: the picture changes when the speaker does.

    Two things are happening here, and both were missing before. The scene is CUT - a
    close-up of whoever is talking, a reverse when someone answers, a wide under narration -
    so the frame belongs to the voice on the soundtrack. And each shot is PERFORMED - jaw,
    blink, head and camera driven by that speaker's own amplitude envelope - so the
    character on screen is visibly the one speaking.

    The scene clip remains the unit above this line: shots are concatenated into
    `scenes.render_path` exactly as before, so the part planner, the audio bus and the
    resume logic are untouched.
    """
    from ..assemble.shot_render import build_shot_jobs, render_scene_shots
    from ..assemble.video import _sha
    from ..media.video.factory import build_video_chain
    from ..scenes.persist import load_story
    # Loaded for the motion prompt, not for the pictures - those were bought at art time.
    # A hosted model needs telling who is in the shot in the same words the image prompt
    # used, or the fox it animates is not the fox that was drawn.
    cast_by_id = {c["character_id"]: c for c in load_story(ctx.db, job["story_id"])["cast"]}
    paths = ctx.paths_for(job["id"])
    by_scene = {a.scene_id: a for a in audio.scenes}
    out_dir = paths.scenes
    shot_dir = out_dir / "shots"
    image_dir = paths.work / "scene_images"
    crf = int(ctx.cfg.get("production.crf", 20))
    chain = build_video_chain(ctx.cfg, workers=ctx.cfg.get("production.render_workers"),
                              crf=crf)
    perf = "production.performance"
    max_images = int(ctx.cfg.get(f"{perf}.max_images_per_scene", 6))
    min_shot_s = float(ctx.cfg.get(f"{perf}.min_shot_s", 0.75))
    max_shot_s = float(ctx.cfg.get(f"{perf}.max_shot_s", 6.5))

    rendered, shots, total = 0, 0, 0.0
    for sc in scenes:
        sa = by_scene[sc["id"]]
        dest = out_dir / f"scene_{sc['idx']:03d}.mp4"
        total += sa.duration_s
        if dest.exists() and sc.get("render_sha256") and _sha(dest) == sc["render_sha256"]:
            log.info("scene_render_reused", scene=sc["idx"])
            continue
        jobs = build_shot_jobs(sc, sa.timing, image_dir, fps=ctx.fps,
                               size=ctx.resolution, max_images=max_images,
                               min_shot_s=min_shot_s, max_shot_s=max_shot_s,
                               cast_by_id=cast_by_id,
                               art_style=ctx.cfg.get("channel.art_style"))
        stats = render_scene_shots(chain, jobs, sc["idx"], shot_dir, dest)
        shots += stats["shots"]
        with tx(ctx.db) as con:
            con.execute("UPDATE scenes SET render_path = ?, render_sha256 = ?, "
                        "status = 'rendered' WHERE id = ?",
                        (str(dest), _sha(dest), sc["id"]))
        log.info("scene_rendered", scene=sc["idx"], **stats)
        rendered += 1
    return {"mode": "performance", "scenes": len(scenes), "rendered": rendered,
            "shots": shots, "seconds": round(total, 1)}


def _animate_cinematic(ctx, job: dict, scenes: list[dict], audio) -> dict:
    """Camera moves over the generated frames. No rig, no lip-sync.

    The scripted camera_move is honoured so the story's shot language survives, and a
    "static" scene still gets a 2% drift - a truly frozen frame under narration reads as a
    slideshow, or as a video that has stalled.
    """
    from ..assemble.cinematic import render_still
    from ..assemble.video import _sha
    paths = ctx.paths_for(job["id"])
    by_scene = {a.scene_id: a for a in audio.scenes}
    out_dir = paths.scenes
    out_dir.mkdir(parents=True, exist_ok=True)
    crf = int(ctx.cfg.get("production.crf", 20))

    rendered, total = 0, 0.0
    for sc in scenes:
        image = Path(sc["plate_path"] or "")
        if not image.exists():
            raise ValidationError(
                f"scene {sc['idx']} has no generated image; re-run the art stage")
        dest = out_dir / f"scene_{sc['idx']:03d}.mp4"
        duration = by_scene[sc["id"]].duration_s
        # Resume, same contract as the puppet renderer: an unchanged scene is not re-cut.
        if dest.exists() and sc.get("render_sha256") and _sha(dest) == sc["render_sha256"]:
            log.info("scene_render_reused", scene=sc["idx"])
            total += duration
            continue
        stats = render_still(image, dest, duration_s=duration, fps=ctx.fps,
                             size=ctx.resolution,
                             camera_move=sc.get("camera_move") or "static", crf=crf)
        with tx(ctx.db) as con:
            con.execute("UPDATE scenes SET render_path = ?, render_sha256 = ?, "
                        "status = 'rendered' WHERE id = ?",
                        (str(dest), _sha(dest), sc["id"]))
        log.info("scene_rendered", scene=sc["idx"], **stats)
        rendered += 1
        total += duration
    return {"mode": "cinematic", "scenes": len(scenes), "rendered": rendered,
            "seconds": round(total, 1)}


@stage("animate", "AUDIO_READY", "SCENES_RENDERED", "animate")
def animate(ctx, job: dict) -> dict:
    from ..assemble.video import render_scenes
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    audio = ctx.load_audio(job["id"], scenes, story["cast"])
    if _performance(ctx):
        return _animate_performance(ctx, job, scenes, audio)
    if _cinematic(ctx):
        return _animate_cinematic(ctx, job, scenes, audio)
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


def video_parts(ctx, job: dict) -> list[dict]:
    """Every rendered part of this job, in release order. One row for a single-part job."""
    with read(ctx.db) as con:
        rows = con.execute("SELECT * FROM videos WHERE job_id = ? ORDER BY part",
                           (job["id"],)).fetchall()
    return [dict(r) for r in rows]


def _part_scene_ids(ctx, job: dict, part: int) -> tuple[int, int] | None:
    with read(ctx.db) as con:
        row = con.execute(
            "SELECT scene_from, scene_to FROM video_parts WHERE job_id = ? AND part = ?",
            (job["id"], part)).fetchone()
    return (row["scene_from"], row["scene_to"]) if row else None


@stage("subtitles", "VIDEO_RENDERED", "SUBTITLED", "assemble")
def subtitles(ctx, job: dict) -> dict:
    from ..media.audio.build import StoryAudio
    from ..media.subtitles.build import captions_from_audio, write_srt, write_vtt
    from ..scenes.persist import load_scenes, load_story
    paths = ctx.paths_for(job["id"])
    story = load_story(ctx.db, job["story_id"])
    scenes = load_scenes(ctx.db, job["story_id"])
    audio = ctx.load_audio(job["id"], scenes, story["cast"])
    by_id = {sa.scene_id: sa for sa in audio.scenes}
    idx_of = {s["id"]: s["idx"] for s in scenes}
    max_chars = int(ctx.cfg.get("subtitles.max_chars_per_line", 32))

    out = []
    for v in video_parts(ctx, job):
        span = _part_scene_ids(ctx, job, v["part"])
        mine = [sa for sa in audio.scenes
                if span is None or span[0] <= idx_of.get(sa.scene_id, -1) <= span[1]]
        # Caption times restart at zero for every part. They are absolute offsets into a
        # finished video, and each part IS a finished video - carrying the whole-story
        # clock through would put part 3's captions three minutes past its own end.
        offsets, clock = {}, 0.0
        for sa in mine:
            offsets[sa.scene_id] = clock
            clock += sa.duration_s
        part_audio = StoryAudio()
        part_audio.scenes = mine
        caps = captions_from_audio(part_audio, offsets, max_chars_per_line=max_chars)
        suffix = "" if len(video_parts(ctx, job)) == 1 else f"_part{v['part']}"
        srt = write_srt(caps, paths.out / f"captions{suffix}.srt")
        vtt = write_vtt(caps, paths.out / f"captions{suffix}.vtt")
        with tx(ctx.db) as con:
            con.execute("UPDATE videos SET srt_path = ?, vtt_path = ? "
                        "WHERE job_id = ? AND part = ?",
                        (str(srt), str(vtt), job["id"], v["part"]))
        out.append({"part": v["part"], "captions": len(caps), "srt": str(srt)})
    return {"parts": len(out), "captions": sum(o["captions"] for o in out),
            "files": [o["srt"] for o in out]}


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
    texts = _thumbnail_texts(ctx.db, job["id"], story)
    parts = video_parts(ctx, job)
    out = []
    for v in parts:
        span = _part_scene_ids(ctx, job, v["part"])
        # Draw each part's plate from its OWN scenes. Reusing one frame across a series
        # gives every part an identical thumbnail, which reads as a duplicate upload in a
        # subscriber's feed - the exact opposite of what parts are for.
        pool = [s for s in scenes
                if span is None or span[0] <= s["idx"] <= span[1]] or scenes
        plate_scene = max(pool, key=lambda s: (
            float((s["staging"].get(hero["character_id"]) or {}).get("scale", 0)),
            s["idx"]))
        # Where the picture comes from depends on the render mode, and getting this wrong
        # is why the thumbnail stage failed on every cinematic episode: it went looking for
        # assets/backgrounds/<location>/plate.png, which only the puppet path ever writes.
        # A cinematic episode's pictures are its generated scene frames - and they already
        # contain the cast, so no puppet is composited over them.
        if _cinematic(ctx):
            plate = _cinematic_thumbnail_plate(ctx, job, plate_scene, hero)
            puppet_dir = None
        else:
            plate = ctx.assets / "backgrounds" / plate_scene["location_id"] / "plate.png"
            puppet_dir = ctx.assets / "characters" / hero["character_id"]
        suffix = "" if len(parts) == 1 else f"_part{v['part']}"
        variants = generate_set(ctx.db, job["id"], plate, puppet_dir,
                                texts, paths.thumbs / suffix.lstrip("_") if suffix
                                else paths.thumbs,
                                variants=int(ctx.cfg.get("thumbnail.variants", 6)))
        with tx(ctx.db) as con:
            con.execute("UPDATE videos SET thumbnail_path = ? WHERE job_id = ? AND part = ?",
                        (str(variants[0].path), job["id"], v["part"]))
        out.append({"part": v["part"], "variants": len(variants),
                    "best": variants[0].score, "chosen": str(variants[0].path)})
    return {"parts": len(out), "thumbnails": out}


def _cinematic_thumbnail_plate(ctx, job: dict, scene: dict, hero: dict) -> Path:
    """The best generated frame of the protagonist for a thumbnail.

    A close-up of the hero beats the scene's opening setup, which is often a wide
    establishing shot under narration - a thumbnail of an empty street sells nothing. The
    shot images already exist on disk, so this costs a directory lookup, and it falls back
    to the scene's own frame when the scene has no close-up of them.
    """
    from ..assemble.shot_render import shot_image_name
    from ..assemble.shotlist import plan_shots
    image_dir = ctx.paths_for(job["id"]).work / "scene_images"
    for shot in plan_shots(scene):
        if shot.speaker == hero["character_id"] and shot.animates_face:
            candidate = image_dir / shot_image_name(scene["idx"], shot.image_key)
            if candidate.exists():
                return candidate
    return Path(scene["plate_path"] or "")


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
                    cta=ctx.cfg.get("channel.cta", "") or "",
                    language=str(ctx.cfg.get("channel.language", "en")))
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
    parts = video_parts(ctx, job)
    if not parts:
        raise ValidationError("no video row for this job")

    idx_of = {s["id"]: s["idx"] for s in scenes}
    warnings, failures = [], []
    for v in parts:
        span = _part_scene_ids(ctx, job, v["part"])
        mine = [sc for sc in scenes
                if span is None or span[0] <= sc["idx"] <= span[1]] or scenes
        # Compare the container against the SCENE TIMELINE, not against the video row - the
        # video row's duration came from probing that same file, so checking one against
        # the other could never fail. The scenes' durations came from the synthesised
        # audio, which is the independent source of truth this check needs. For a part,
        # only ITS scenes count; measuring a 3-minute part against the whole story's
        # timeline would fail every part of every series.
        planned = sum(sc["duration_s"] or 0.0 for sc in mine)
        if planned <= 0:
            planned = v["duration_s"]
        report = checks.run(
            ctx.db, Path(v["path"]),
            expected_duration_s=planned,
            srt=Path(v["srt_path"]) if v["srt_path"] else None,
            thumbnail=Path(v["thumbnail_path"]) if v["thumbnail_path"] else None,
            title=meta.title, description=meta.description, tags=meta.tags,
            story=story, scenes=mine,
            target_lufs=float(ctx.cfg.get("audio.target_lufs", -14.0)),
            true_peak_db=float(ctx.cfg.get("audio.true_peak_db", -1.0)),
            made_for_kids=ctx.cfg.get("channel.made_for_kids"),
            disclose_synthetic=bool(ctx.cfg.get("channel.disclose_synthetic", True)),
            asset_paths=_job_asset_paths(ctx, job),
            min_minutes=float(ctx.cfg.get("production.part_min_minutes", 1.0) or 1.0))

        # QC is the stage that actually measured the finished file, so it owns the recorded
        # loudness. Leaving it to `assemble` means a job resumed after that stage keeps a
        # stale (or zero) value in a column the dashboard displays as fact.
        measured = next((f.detail.get("lufs") for f in report.findings
                         if f.check == "audio.loudness" and "lufs" in f.detail), None)
        with tx(ctx.db) as con:
            con.execute("UPDATE videos SET qc_report = ?, lufs = COALESCE(?, lufs) "
                        "WHERE job_id = ? AND part = ?",
                        (jdump(report.to_dict()), measured, job["id"], v["part"]))
        warnings += [f"part {v['part']}: {f.check}" for f in report.warnings]
        failures += [f"part {v['part']}: {f.message}" for f in report.failures]

    if failures:
        # Every part is checked before raising. Stopping at the first failure would hide
        # the rest, and a series where two parts are broken needs both reported at once.
        raise PolicyViolation("QC failed: " + "; ".join(failures)[:600])
    return {"parts": len(parts), "passed": True, "warnings": len(warnings),
            "warning_list": warnings}


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
    from ..assemble.parts import Part, part_title
    from ..publish.metadata import part_description
    meta = ctx.load_metadata(job["id"])
    parts = video_parts(ctx, job)
    if not parts:
        raise ValidationError("no video row for this job")
    made_for_kids = ctx.cfg.get("channel.made_for_kids")
    if made_for_kids is None:
        raise PolicyViolation(
            "channel.made_for_kids is unset. YouTube requires an explicit audience "
            "declaration for every upload; read docs/05-COMPLIANCE.md §2 and set it.")
    privacy = ctx.cfg.get("production.privacy_on_upload", "private")
    total = len(parts)

    uploaded = []
    for v in parts:
        with read(ctx.db) as con:
            done = con.execute(
                "SELECT video_id FROM youtube_uploads WHERE job_id = ? AND part = ? "
                "AND video_id IS NOT NULL", (job["id"], v["part"])).fetchone()
        if done:
            # Upload is retryable and videos.insert costs 1,600 units. Re-uploading a part
            # that already succeeded because a LATER part failed would burn the day's quota
            # and leave a duplicate on the channel.
            log.info("part_already_uploaded", job=job["id"], part=v["part"],
                     video_id=done["video_id"])
            uploaded.append({"part": v["part"], "video_id": done["video_id"],
                             "skipped": True})
            continue

        pobj = Part(v["part"], [], float(v["duration_s"] or 0.0))
        title = part_title(meta.title, pobj, total)
        description = part_description(meta.description, v["part"], total,
                                       [u.get("video_id") for u in uploaded])
        try:
            result = ctx.youtube.upload(
                Path(v["path"]), title=title, description=description,
                tags=meta.tags, category_id=int(ctx.cfg.get("channel.category_id", 1)),
                privacy=privacy, made_for_kids=bool(made_for_kids),
                language=ctx.cfg.get("channel.language", "en"),
                thumbnail=Path(v["thumbnail_path"]) if v["thumbnail_path"] else None,
                captions=Path(v["srt_path"]) if v["srt_path"] else None)
        except Exception as e:                                     # noqa: BLE001
            record_upload(ctx.db, job["id"], meta, None, bool(made_for_kids), privacy,
                          error=str(e)[:800], part=v["part"], title=title,
                          description=description)
            raise
        record_upload(ctx.db, job["id"], meta, result, bool(made_for_kids), privacy,
                      part=v["part"], title=title, description=description)
        uploaded.append({"part": v["part"], "video_id": result.video_id,
                         "url": result.watch_url, "units": result.units_spent})
        ctx.notifier.send(f"Uploaded: {title}", result.watch_url, level="info",
                          link=result.watch_url)

    first = next((u for u in uploaded if u.get("url")), uploaded[0] if uploaded else {})
    return {"parts": total, "uploaded": uploaded,
            "video_id": first.get("video_id"), "url": first.get("url"),
            "units": sum(u.get("units", 0) for u in uploaded)}


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
