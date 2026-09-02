"""Build a scene timeline from synthesised audio.

The ordering rule from docs/02 §6: audio is generated first and the audio determines every
duration. `duration_hint_s` from the LLM is only used for planning, never for rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..tts.base import Utterance


@dataclass
class Cue:
    """One utterance placed on the scene's local timeline."""
    utterance: Utterance
    start_s: float
    character_id: str | None

    @property
    def end_s(self) -> float:
        return self.start_s + self.utterance.duration_s


@dataclass
class SceneTiming:
    index: int
    cues: list[Cue] = field(default_factory=list)
    duration_s: float = 0.0
    pre_pad: float = 0.35
    post_pad: float = 0.55

    def speech_spans(self, character_id: str | None) -> list[tuple[float, float]]:
        return [(c.start_s, c.end_s) for c in self.cues if c.character_id == character_id]

    def envelope_for(self, character_id: str | None, fps: int) -> list[float]:
        """Stitch each cue's envelope onto the scene timeline, silence elsewhere."""
        n = int(round(self.duration_s * fps)) + 1
        env = [0.0] * n
        for c in self.cues:
            if c.character_id != character_id:
                continue
            off = int(round(c.start_s * fps))
            for i, v in enumerate(c.utterance.envelope):
                j = off + i
                if 0 <= j < n:
                    env[j] = max(env[j], v)
        return env


def build_scene_timing(index: int, utterances: list[tuple[Utterance, str | None]],
                       gap_s: float = 0.28, pre_pad: float = 0.35,
                       post_pad: float = 0.55, min_duration: float = 2.0) -> SceneTiming:
    """Lay utterances end to end with a beat between them."""
    t = SceneTiming(index=index, pre_pad=pre_pad, post_pad=post_pad)
    clock = pre_pad
    for utt, char_id in utterances:
        t.cues.append(Cue(utterance=utt, start_s=clock, character_id=char_id))
        clock += utt.duration_s + gap_s
    clock = clock - gap_s if t.cues else clock
    t.duration_s = max(min_duration, round(clock + post_pad, 3))
    return t
