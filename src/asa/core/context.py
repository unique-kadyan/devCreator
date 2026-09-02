"""Everything a stage needs, built once per run and shared.

Provider chains are expensive to construct (model discovery, token checks, Kokoro's ~24s
first-call init) so they are lazy properties rather than constructor arguments: a job that
only needs to re-render never touches the LLM chain, and a job that fails at story never
loads a TTS model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .config import Config, load_config
from .db import read
from .logging import get_logger
from .notify import Notifier
from .quota import QuotaTracker

log = get_logger("context")


@dataclass
class JobPaths:
    work: Path
    out: Path

    @property
    def audio(self) -> Path: return self.work / "audio"

    @property
    def scenes(self) -> Path: return self.work / "scenes"

    @property
    def thumbs(self) -> Path: return self.work / "thumbnails"

    def ensure(self) -> "JobPaths":
        for p in (self.work, self.out, self.audio, self.scenes, self.thumbs):
            p.mkdir(parents=True, exist_ok=True)
        return self


class Context:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.db = self.cfg.path("paths.db", "data/asa.db")
        self.cache = self.cfg.path("paths.cache", "data/cache")
        self.work_root = self.cfg.path("paths.work", "data/work")
        self.out_root = self.cfg.path("paths.out", "data/out")
        self.assets = self.cfg.root / "assets"
        self.fps = int(self.cfg.get("production.fps", 24))
        self.resolution = tuple(self.cfg.get("production.resolution", [1920, 1080]))
        self.world = (self.resolution[0] * 7 // 5, self.resolution[1] * 7 // 5)
        self.target_minutes = float(self.cfg.get("production.target_minutes", 7))
        self._audio_memo: dict[int, object] = {}
        self._meta_memo: dict[int, object] = {}

    # ------------------------------------------------------------- lazy chains

    @cached_property
    def quota(self) -> QuotaTracker:
        return QuotaTracker(self.db)

    @cached_property
    def notifier(self) -> Notifier:
        return Notifier(self.cfg.get("runtime.notify.driver", "dashboard"),
                        self.cfg.get("runtime.notify.url"), self.db)

    @cached_property
    def llm(self):
        from ..llm.factory import build_chain
        return build_chain(self.cfg, self.db)

    @cached_property
    def images(self):
        from ..media.images.factory import build_image_chain
        return build_image_chain(self.cfg, self.db)

    @cached_property
    def tts(self):
        from ..media.tts.kokoro_tts import KokoroTTS
        return KokoroTTS()

    @cached_property
    def characters(self):
        from ..characters.factory import CharacterFactory
        return CharacterFactory(self.db, self.assets)

    @cached_property
    def sfx(self):
        from ..media.audio.sfx import SFXLibrary
        return SFXLibrary(self.db, self.assets / "sfx",
                          self.cfg.secret("FREESOUND_API_KEY", required=False))

    @cached_property
    def music(self):
        from ..media.audio.music import MusicLibrary
        return MusicLibrary(self.assets / "music", self.db)

    @cached_property
    def youtube(self):
        from ..publish.youtube import YouTubeClient
        return YouTubeClient(self.cfg.secret("YT_CLIENT_ID", required=False),
                             self.cfg.secret("YT_CLIENT_SECRET", required=False),
                             self.cfg.root / "config" / "youtube_token.json",
                             quota=self.quota)

    # ------------------------------------------------------------------ paths

    def paths_for(self, job_id: int) -> JobPaths:
        return JobPaths(self.work_root / f"job_{job_id:05d}",
                        self.out_root / f"job_{job_id:05d}").ensure()

    # ------------------------------------------------- cross-stage artefacts
    # Stages run in separate invocations (and after a crash, separate processes), so
    # nothing may be passed between them in memory alone. Audio is rebuilt from the DB
    # plus the wav files on disk; metadata is written to a JSON sidecar. The in-process
    # dicts below are only a memo to avoid redoing that work inside a single run.

    def cache_audio(self, job_id: int, audio) -> None:
        self._audio_memo[job_id] = audio

    def load_audio(self, job_id: int, scenes: list[dict], cast: list[dict]):
        memo = self._audio_memo.get(job_id)
        if memo is not None:
            return memo
        from ..media.audio.build import SceneAudio, StoryAudio
        from ..media.audio.envelope import rms_envelope
        from ..media.audio.timeline import build_scene_timing
        from ..media.tts.base import Utterance
        paths = self.paths_for(job_id)
        story_audio = StoryAudio()
        with read(self.db) as con:
            for sc in scenes:
                rows = con.execute(
                    "SELECT a.*, d.line, d.idx AS didx FROM audio a "
                    "JOIN dialogue d ON d.id = a.dialogue_id "
                    "WHERE a.scene_id = ? AND a.kind IN ('narration','dialogue') "
                    "ORDER BY d.idx", (sc["id"],)).fetchall()
                pairs, utts = [], []
                for r in rows:
                    wav = Path(r["path"])
                    if not wav.exists():
                        raise FileNotFoundError(
                            f"scene {sc['idx']} line {r['didx']} was synthesised to {wav} "
                            f"but that file is gone; re-run the audio stage")
                    u = Utterance(text=r["line"], path=wav, duration_s=r["duration_s"],
                                  sample_rate=24000, character_id=r["character_id"],
                                  voice_id=r["voice_id"] or "", provider=r["provider"] or "",
                                  text_sha256=r["text_sha256"] or "")
                    u.envelope = rms_envelope(wav, fps=self.fps)
                    u.envelope_fps = self.fps
                    utts.append(u)
                    pairs.append((u, r["character_id"]))
                timing = build_scene_timing(sc["idx"], pairs)
                story_audio.scenes.append(
                    SceneAudio(scene_id=sc["id"], index=sc["idx"], timing=timing,
                               utterances=utts))
        self._audio_memo[job_id] = story_audio
        return story_audio

    def cache_metadata(self, job_id: int, meta) -> None:
        self._meta_memo[job_id] = meta
        p = self.paths_for(job_id).work / "metadata.json"
        p.write_text(json.dumps({"title": meta.title, "description": meta.description,
                                 "tags": meta.tags, "model_id": meta.model_id,
                                 "hashtags": meta.hashtags}, indent=2))

    def load_metadata(self, job_id: int):
        memo = self._meta_memo.get(job_id)
        if memo is not None:
            return memo
        from ..publish.metadata import Metadata
        p = self.paths_for(job_id).work / "metadata.json"
        if not p.exists():
            raise FileNotFoundError(
                f"no metadata for job {job_id} at {p}; re-run the metadata stage")
        d = json.loads(p.read_text())
        meta = Metadata(title=d["title"], description=d["description"], tags=d["tags"],
                        candidates=[], model_id=d.get("model_id", ""),
                        hashtags=d.get("hashtags", []))
        self._meta_memo[job_id] = meta
        return meta

    def tts_cache(self) -> Path:
        p = self.cache / "tts"
        p.mkdir(parents=True, exist_ok=True)
        return p
