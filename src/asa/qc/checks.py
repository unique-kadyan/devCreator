"""Quality control. The last gate before a human ever sees the video.

Every check returns a Finding with a severity. `fail` blocks publication outright; `warn`
routes the job to the approval queue with the reason attached. Nothing here is advisory
noise - a check that nobody would ever act on has no business slowing a render down.

The checks are deliberately mechanical: durations, loudness, sync, licences, disclosure.
Taste is the human reviewer's job and this module does not pretend otherwise.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..assemble.video import probe_stream
from ..core.db import read
from ..core.ledger import audit as licence_audit
from ..core.logging import get_logger
from ..media.audio.mixer import measure_loudness

log = get_logger("qc")

BANNED = re.compile(
    r"\b(kill yourself|suicide method|how to make a bomb|child\s*(porn|abuse)|"
    r"n[i1]gg[e3]r|f[a4]gg[o0]t)\b", re.I)
# Real trademarks that a story generator drifts toward. Their presence is not automatically
# infringement, but it needs a human to look before it ships.
STUDIO_TERMS = re.compile(
    r"\b(disney|pixar|dreamworks|studio ghibli|ghibli|nintendo|pok[eé]mon|marvel|"
    r"mickey mouse|bluey|peppa pig|paw patrol|minions)\b", re.I)


@dataclass
class Finding:
    check: str
    severity: str          # fail | warn | info
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class QCReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, check: str, severity: str, message: str, **detail) -> None:
        self.findings.append(Finding(check, severity, message, detail))

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "fail"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {"passed": self.passed,
                "fail": len(self.failures), "warn": len(self.warnings),
                "findings": [asdict(f) for f in self.findings]}


def _black_seconds(video: Path, threshold: float = 0.10) -> float:
    """Total seconds ffmpeg detects as black. A long black stretch means a scene rendered
    empty, which is invisible in the logs and glaring to a viewer."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video),
         "-vf", f"blackdetect=d=0.5:pix_th={threshold}", "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    total = sum(float(m) for m in re.findall(r"black_duration:(\d+\.?\d*)", r.stderr))
    return total


def run(db: Path, video: Path, *, expected_duration_s: float, srt: Path | None,
        thumbnail: Path | None, title: str, description: str, tags: list[str],
        story: dict, scenes: list[dict], target_lufs: float = -14.0,
        true_peak_db: float = -1.0, made_for_kids: bool | None = None,
        disclose_synthetic: bool = True, asset_paths: list[str] | None = None,
        min_minutes: float = 1.0) -> QCReport:
    rep = QCReport()

    # ---------------------------------------------------------------- container
    if not video.exists() or video.stat().st_size == 0:
        rep.add("video.exists", "fail", f"no video at {video}")
        return rep
    info = probe_stream(video)
    rep.add("video.probe", "info", "container probed", **info)

    if info["vcodec"] != "h264":
        rep.add("video.codec", "warn", f"unexpected video codec {info['vcodec']!r}")
    if info["acodec"] not in ("aac", "mp3"):
        rep.add("video.acodec", "fail", f"unusable audio codec {info['acodec']!r}")
    if (info["width"], info["height"]) != (1920, 1080):
        rep.add("video.resolution", "warn",
                f"{info['width']}x{info['height']} is not 1920x1080")
    if info["channels"] < 1:
        rep.add("audio.present", "fail", "no audio stream")

    drift = abs(info["duration"] - expected_duration_s)
    if drift > 1.5:
        rep.add("video.duration", "fail",
                f"container is {info['duration']:.2f}s but scenes total "
                f"{expected_duration_s:.2f}s (drift {drift:.2f}s)",
                container_s=info["duration"], expected_s=expected_duration_s)
    elif drift > 0.4:
        rep.add("video.duration", "warn", f"duration drift {drift:.2f}s")

    sync = abs(info["video_start"] - info["audio_start"])
    if sync > 0.08:
        rep.add("av.sync", "fail", f"audio starts {sync:.3f}s from video",
                video_start=info["video_start"], audio_start=info["audio_start"])

    if info["duration"] < min_minutes * 60:
        rep.add("video.length", "warn",
                f"{info['duration'] / 60:.1f} min is short for a long-form upload")

    # ---------------------------------------------------------------- loudness
    try:
        loud = measure_loudness(video)
        li, tp = float(loud["integrated_lufs"]), float(loud["true_peak_dbfs"])
        rep.add("audio.loudness", "info", "measured", lufs=li, true_peak_db=tp)
        if abs(li - target_lufs) > 1.0:
            rep.add("audio.lufs", "fail",
                    f"{li:.1f} LUFS is outside {target_lufs:+.0f} +/- 1.0", measured=li)
        if tp > true_peak_db + 0.3:
            rep.add("audio.peak", "fail", f"true peak {tp:.1f} dBTP exceeds "
                                          f"{true_peak_db:.1f}", measured=tp)
    except Exception as e:                                     # noqa: BLE001
        rep.add("audio.loudness", "warn", f"could not measure loudness: {e}")

    black = _black_seconds(video)
    if black > 2.0:
        rep.add("video.black", "fail", f"{black:.1f}s of black frames detected",
                seconds=black)
    elif black > 0.6:
        rep.add("video.black", "warn", f"{black:.1f}s of black frames")

    # ---------------------------------------------------------------- captions
    if srt is None or not srt.exists():
        rep.add("subtitles.exists", "warn", "no subtitle file was produced")
    else:
        text = srt.read_text(encoding="utf-8")
        cues = text.count("-->")
        if cues == 0:
            rep.add("subtitles.empty", "fail", "subtitle file has no cues")
        else:
            last = max((float(h) * 3600 + float(m) * 60 + float(s.replace(",", ".")))
                       for h, m, s in re.findall(
                           r"--> (\d\d):(\d\d):(\d\d,\d\d\d)", text))
            coverage = last / max(1e-6, info["duration"])
            if coverage < 0.55:
                rep.add("subtitles.coverage", "warn",
                        f"captions stop at {coverage:.0%} of the runtime")

    # ---------------------------------------------------------------- thumbnail
    if thumbnail is None or not thumbnail.exists():
        rep.add("thumbnail.exists", "fail", "no thumbnail")
    else:
        size = thumbnail.stat().st_size
        if size > 2 * 1024 * 1024:
            rep.add("thumbnail.size", "fail", f"{size / 1e6:.2f} MB exceeds YouTube's 2 MB")
        from PIL import Image
        with Image.open(thumbnail) as im:
            if im.size[0] < 1280 or im.size[1] < 720:
                rep.add("thumbnail.dimensions", "warn", f"{im.size} is below 1280x720")

    # ---------------------------------------------------------------- metadata
    if not 1 <= len(title) <= 100:
        rep.add("metadata.title", "fail", f"title length {len(title)} is out of range")
    if len(description) > 5000:
        rep.add("metadata.description", "fail", "description exceeds 5000 characters")
    if disclose_synthetic and "ai-assisted" not in description.lower() \
            and "ai assisted" not in description.lower():
        rep.add("metadata.disclosure", "fail",
                "synthetic-content disclosure is missing from the description")
    if made_for_kids is None:
        rep.add("metadata.made_for_kids", "fail",
                "made_for_kids has not been set; YouTube requires an explicit answer")
    if sum(len(t) for t in tags) > 500:
        rep.add("metadata.tags", "fail", "tags exceed 500 characters in total")

    # ---------------------------------------------------------------- content
    blob = " ".join([title, description, story.get("logline", ""),
                     " ".join(s.get("action", "") or "" for s in scenes),
                     " ".join(d["line"] for s in scenes for d in s.get("dialogue", []))])
    if BANNED.search(blob):
        rep.add("content.banned", "fail", "banned-term filter matched",
                terms=sorted({m.group(0).lower() for m in BANNED.finditer(blob)}))
    studio = sorted({m.group(0).lower() for m in STUDIO_TERMS.finditer(blob)})
    if studio:
        rep.add("content.trademarks", "warn",
                f"references a real studio or franchise: {', '.join(studio)}",
                terms=studio)

    # ---------------------------------------------------------------- licences
    problems = licence_audit(db, asset_paths)
    if problems:
        rep.add("licence.audit", "fail",
                f"{len(problems)} asset(s) are not clear for commercial use",
                assets=[p.get("path") for p in problems][:20])

    # ---------------------------------------------------------------- story
    dupes = _similar_stories(db, story)
    if dupes:
        rep.add("story.duplicate", "warn",
                f"beat signature matches {len(dupes)} recent stor{'y' if len(dupes)==1 else 'ies'}",
                matches=dupes[:5])

    log.info("qc_done", passed=rep.passed, failures=len(rep.failures),
             warnings=len(rep.warnings))
    return rep


def _similar_stories(db: Path, story: dict) -> list[str]:
    sig = (story.get("beat_signature") or "").strip()
    if not sig:
        return []
    with read(db) as con:
        rows = con.execute(
            "SELECT id, title, beat_signature FROM stories WHERE id <> ? "
            "ORDER BY id DESC LIMIT 60", (story.get("id", -1),)).fetchall()
    mine = set(sig.split("|"))
    out = []
    for r in rows:
        other = set((r["beat_signature"] or "").split("|"))
        if not other:
            continue
        jaccard = len(mine & other) / max(1, len(mine | other))
        if jaccard >= 0.8:
            out.append(f"#{r['id']} {r['title']} ({jaccard:.0%})")
    return out
