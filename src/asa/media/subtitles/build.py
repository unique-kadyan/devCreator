"""Subtitles from the synthesised audio, not from the script.

The script is what we asked for; the audio is what was said and, more importantly, when.
Because every Utterance carries its true duration and its position on the scene timeline,
subtitle timing is exact without any forced alignment - and exact timing is what makes
captions usable rather than merely present.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_LINES = 2


@dataclass
class Caption:
    start_s: float
    end_s: float
    text: str
    speaker: str | None = None


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _split_long(text: str, start: float, end: float, width: int) -> list[Caption]:
    """A 14-second line is one utterance but must not be one caption.

    Split on sentence boundaries first (they are where a reader's eye rests), then
    apportion time by character count, which tracks speech duration closely enough at this
    granularity.
    """
    lines = _wrap(text, width)
    if len(lines) <= MAX_LINES:
        return [Caption(start, end, "\n".join(lines))]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()] or [text]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        cand = f"{buf} {s}".strip()
        if len(_wrap(cand, width)) <= MAX_LINES:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
            # A single sentence too long for two lines gets cut on word count.
            if len(_wrap(s, width)) > MAX_LINES:
                words = s.split()
                per = max(1, (width * MAX_LINES) // 6)
                for i in range(0, len(words), per):
                    chunks.append(" ".join(words[i:i + per]))
                buf = ""
            else:
                buf = s
    if buf:
        chunks.append(buf)

    total = sum(len(c) for c in chunks) or 1
    out, clock = [], start
    span = max(0.001, end - start)
    for c in chunks:
        d = span * len(c) / total
        out.append(Caption(clock, min(end, clock + d), "\n".join(_wrap(c, width))))
        clock += d
    return out


def captions_from_audio(story_audio, scene_offsets: dict[int, float],
                        max_chars_per_line: int = 32,
                        min_duration_s: float = 1.0) -> list[Caption]:
    """`scene_offsets` maps scene_id -> where that scene starts in the finished video."""
    caps: list[Caption] = []
    for sa in story_audio.scenes:
        base = scene_offsets.get(sa.scene_id, 0.0)
        for cue in sa.timing.cues:
            caps.extend(_split_long(cue.utterance.text, base + cue.start_s,
                                    base + cue.end_s, max_chars_per_line))
    caps.sort(key=lambda c: c.start_s)
    # Enforce a floor and stop neighbours overlapping; a caption that is still on screen
    # when the next one appears reads as a rendering bug.
    for i, c in enumerate(caps):
        if c.end_s - c.start_s < min_duration_s:
            c.end_s = c.start_s + min_duration_s
        if i + 1 < len(caps) and c.end_s > caps[i + 1].start_s:
            c.end_s = max(c.start_s + 0.4, caps[i + 1].start_s - 0.04)
    return caps


def _ts(seconds: float, comma: bool = True) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep)


def write_srt(caps: list[Caption], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for i, c in enumerate(caps, start=1):
        out.append(f"{i}\n{_ts(c.start_s)} --> {_ts(c.end_s)}\n{c.text}\n")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def write_vtt(caps: list[Caption], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ["WEBVTT", ""]
    for c in caps:
        out.append(f"{_ts(c.start_s, comma=False)} --> {_ts(c.end_s, comma=False)}")
        out.append(c.text)
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")
    return path
