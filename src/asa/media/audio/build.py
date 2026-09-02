"""Synthesise every line in a story and derive the scene timings from the result.

Audio comes before animation, always. The LLM's duration_hint_s is a planning number; the
authoritative length of a scene is how long its speech actually takes. Getting this
backwards is the classic way to end up with mouths still moving after the line has ended.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ...core.db import tx
from ...core.logging import get_logger
from ..tts.base import Utterance
from ..tts.voices import rate_for
from .envelope import rms_envelope
from .timeline import SceneTiming, build_scene_timing

log = get_logger("audio_build")

NARRATOR = None            # dialogue.character_id IS NULL means the narrator


@dataclass
class SceneAudio:
    scene_id: int
    index: int
    timing: SceneTiming
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.timing.duration_s


@dataclass
class StoryAudio:
    scenes: list[SceneAudio] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(s.duration_s for s in self.scenes)


class VoiceDirector:
    """Maps a character + emotion onto concrete synthesis parameters.

    Voice identity lives on the character row, so it is stable for the life of the channel.
    Emotion only moves pacing, because Kokoro has no emotion axis and faking one with pitch
    makes every angry line sound like a different actor.
    """

    def __init__(self, cast: list[dict], narrator_voice: str = "bm_fable"):
        self.by_id = {c["character_id"]: c for c in cast}
        self.narrator_voice = narrator_voice

    def params(self, character_id: str | None, emotion: str | None) -> tuple[str, float, float]:
        if character_id is None or character_id not in self.by_id:
            return self.narrator_voice, rate_for(emotion) * 0.98, 0.0
        c = self.by_id[character_id]
        speed = (c.get("voice_rate") or 1.0) * rate_for(emotion)
        # Emotion pacing multiplies the character's own rate; clamp so an excited child
        # does not outrun the subtitles.
        speed = min(1.25, max(0.80, speed))
        return c["voice_id"], round(speed, 3), float(c.get("voice_pitch_semi") or 0.0)


def synthesise_story(db: Path, tts, scenes: list[dict], cast: list[dict],
                     work_dir: Path, cache_dir: Path, fps: int = 24,
                     narrator_voice: str = "bm_fable") -> StoryAudio:
    director = VoiceDirector(cast, narrator_voice)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = StoryAudio()

    for sc in scenes:
        lines = sc["dialogue"]
        pairs: list[tuple[Utterance, str | None]] = []
        utts: list[Utterance] = []
        for line in lines:
            cid = line["character_id"]
            voice, speed, pitch = director.params(cid, line["emotion"])
            path = work_dir / f"s{sc['idx']:02d}_l{line['idx']:02d}.wav"
            utt = tts.synthesize(line["line"], voice, path, speed=speed,
                                 pitch_semitones=pitch, character_id=cid,
                                 cache_dir=cache_dir)
            utt.envelope = rms_envelope(utt.path, fps=fps)
            utt.envelope_fps = fps
            utts.append(utt)
            pairs.append((utt, cid))
            _record_audio(db, sc["id"], line["id"], utt, cid)

        timing = build_scene_timing(sc["idx"], pairs)
        out.scenes.append(SceneAudio(scene_id=sc["id"], index=sc["idx"], timing=timing,
                                     utterances=utts))
        with tx(db) as con:
            con.execute("UPDATE scenes SET duration_s = ?, status = 'audio_ready' "
                        "WHERE id = ?", (timing.duration_s, sc["id"]))
        log.info("scene_audio", scene=sc["idx"], lines=len(utts),
                 duration_s=round(timing.duration_s, 2))

    log.info("story_audio_done", scenes=len(out.scenes), total_s=round(out.total_s, 1))
    return out


def _record_audio(db: Path, scene_id: int, dialogue_id: int, utt: Utterance,
                  cid: str | None) -> None:
    with tx(db) as con:
        con.execute("""
            INSERT INTO audio (scene_id, dialogue_id, kind, character_id, path, duration_s,
                               provider, voice_id, text_sha256)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (scene_id, dialogue_id, "narration" if cid is None else "dialogue", cid,
              str(utt.path), utt.duration_s, utt.provider, utt.voice_id, utt.text_sha256))
