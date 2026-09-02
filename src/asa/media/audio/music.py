"""Music bed selection from the local licensed library.

There is deliberately no generation here. Every good open music model ships non-commercial
weights (MusicGen is CC-BY-NC), so the library is curated by hand once and then reused.
Layout: assets/music/<mood>/<track>.{wav,mp3,flac}
"""
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...core.ledger import audit

MOODS = ["curious", "warm", "tense", "triumphant", "sad", "comedic", "neutral"]
# Story emotions -> the mood folder to draw from.
EMOTION_MOOD = {
    "curious": "curious", "neutral": "neutral", "happy": "warm", "sad": "sad",
    "scared": "tense", "angry": "tense", "surprised": "tense",
    "determined": "triumphant", "wry": "comedic", "excited": "warm",
}


@dataclass
class Track:
    path: Path
    mood: str
    duration_s: float


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


class MusicLibrary:
    def __init__(self, root: Path, db: Path | None = None):
        self.root = root
        self.db = db

    def tracks(self, mood: str | None = None) -> list[Track]:
        out: list[Track] = []
        for folder in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not folder.is_dir() or (mood and folder.name != mood):
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                    out.append(Track(p, folder.name, _duration(p)))
        return out

    def pick(self, emotion: str | None, seed: int = 0) -> Track | None:
        """Deterministic given a seed, so a re-render uses the same bed."""
        mood = EMOTION_MOOD.get((emotion or "neutral").lower(), "neutral")
        pool = self.tracks(mood) or self.tracks("neutral") or self.tracks()
        if not pool:
            return None
        if self.db:
            blocked = {p["path"] for p in audit(self.db)}
            pool = [t for t in pool
                    if not any(str(t.path).endswith(b) for b in blocked)] or pool
        return random.Random(seed).choice(pool)

    def bed_for(self, emotion: str | None, duration_s: float, out_path: Path,
                seed: int = 0, fade_s: float = 2.0) -> Path | None:
        """Loop and crossfade a track to exactly `duration_s`, with a fade in/out."""
        track = self.pick(emotion, seed)
        if track is None:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fade_out_at = max(0.0, duration_s - fade_s)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(track.path),
             "-t", f"{duration_s:.3f}",
             "-af", f"afade=t=in:st=0:d={fade_s},afade=t=out:st={fade_out_at:.3f}:d={fade_s}",
             "-c:a", "pcm_s16le", str(out_path)], check=True)
        return out_path
