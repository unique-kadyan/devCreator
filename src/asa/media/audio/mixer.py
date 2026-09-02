"""Mixdown: voice bus, ducked music, SFX, and a broadcast-safe master.

Targets from config: voice -16 LUFS, music -30 LUFS, SFX -22 LUFS, master -14 LUFS / -1 dBTP.
-14 LUFS is what YouTube normalises toward, so mastering there avoids being turned down.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ...core.errors import RenderError


@dataclass
class Placed:
    path: Path
    start_s: float
    gain_db: float = 0.0


def _delay_ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _build_graph(voice: list[Placed], music: list[Placed], sfx: list[Placed],
                 duration_s: float, sample_rate: int, duck_db: float) -> tuple[list[str], str]:
    """Returns (ffmpeg input args, filtergraph ending in [premaster])."""
    inputs: list[str] = []
    parts: list[str] = []
    voice_labels: list[str] = []
    bed_labels: list[str] = []

    def add(items: list[Placed], kind: str, labels: list[str]) -> None:
        for p in items:
            i = len(inputs) // 2
            inputs.extend(["-i", str(p.path)])
            lab = f"{kind}{i}"
            parts.append(
                f"[{i}:a]aresample={sample_rate},aformat=channel_layouts=stereo,"
                f"adelay={_delay_ms(p.start_s)}:all=1,"
                f"volume={p.gain_db}dB[{lab}]")
            labels.append(lab)

    add(voice, "v", voice_labels)
    add(music, "m", bed_labels)
    add(sfx, "s", bed_labels)
    if not voice_labels and not bed_labels:
        raise ValueError("nothing to mix")

    if voice_labels:
        parts.append("".join(f"[{l}]" for l in voice_labels)
                     + f"amix=inputs={len(voice_labels)}:normalize=0:dropout_transition=0[vbus]")
    if bed_labels:
        parts.append("".join(f"[{l}]" for l in bed_labels)
                     + f"amix=inputs={len(bed_labels)}:normalize=0:dropout_transition=0[bbus]")

    if voice_labels and bed_labels:
        # duck the bed under speech; the voice bus is the sidechain key
        parts.append("[vbus]asplit=2[vout][vkey]")
        parts.append(
            f"[bbus][vkey]sidechaincompress=threshold=0.05:ratio={max(2.0, abs(duck_db)/2):.1f}"
            f":attack=12:release=320:makeup=1[bduck]")
        parts.append("[vout][bduck]amix=inputs=2:normalize=0:dropout_transition=0[premaster]")
    elif voice_labels:
        parts.append("[vbus]anull[premaster]")
    else:
        parts.append("[bbus]anull[premaster]")

    parts.append(f"[premaster]apad,atrim=0:{duration_s:.3f},aresample={sample_rate}[bed]")
    return inputs, ";".join(parts)


def _measure_for_loudnorm(path: Path, target_lufs: float, true_peak_db: float) -> dict:
    """Pass 1 of two-pass loudnorm: get the real measurements from the source."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.S)
    if not m:
        raise RuntimeError("loudnorm pass 1 produced no measurements")
    return json.loads(m.group(0))


def mix(voice: list[Placed], music: list[Placed], sfx: list[Placed],
        duration_s: float, out_path: Path, sample_rate: int = 48_000,
        duck_db: float = -12.0, target_lufs: float = -14.0,
        true_peak_db: float = -1.0) -> Path:
    """Mix, then normalise with TWO-PASS loudnorm.

    Single-pass loudnorm is only accurate to about +/-1 LU, and `alimiter` defaults to
    `level=true`, which auto-normalises the signal back UP to the ceiling - together they
    produced -13.1 LUFS at 0.0 dBFS (clipping) instead of -14 / -1. Both are fixed here.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs, graph = _build_graph(voice, music, sfx, duration_s, sample_rate, duck_db)

    stage = out_path.with_suffix(".premaster.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + inputs
                   + ["-filter_complex", graph, "-map", "[bed]",
                      "-c:a", "pcm_s24le", str(stage)], check=True)

    meas = _measure_for_loudnorm(stage, target_lufs, true_peak_db)
    ln = (f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11"
          f":measured_I={meas['input_i']}:measured_TP={meas['input_tp']}"
          f":measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}"
          f":offset={meas['target_offset']}:linear=true:print_format=summary")
    ceiling = 10 ** (true_peak_db / 20)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(stage),
         "-af", f"{ln},alimiter=limit={ceiling:.4f}:level=disabled,aresample={sample_rate}",
         "-c:a", "pcm_s16le", str(out_path)], check=True)
    stage.unlink(missing_ok=True)
    return out_path


def mux(video: Path, audio: Path, out_path: Path, audio_bitrate: str = "192k") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
         "-shortest", "-movflags", "+faststart", str(out_path)], check=True)
    return out_path


def measure_loudness(path: Path) -> dict:
    """Read back integrated loudness and true peak so QC can assert on them.

    Keys are `integrated_lufs`, `true_peak_dbfs`, `lra` - NOT loudnorm's `input_i` /
    `input_tp`, which come from `_measure_for_loudnorm` and mean something different
    (the measurement of the PRE-master). Confusing the two silently reported 0.0 LUFS for
    a master that was actually correct, and QC would have failed it.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True)
    out = {}
    for line in proc.stderr.splitlines():
        s = line.strip()
        if s.startswith("I:") and "LUFS" in s:
            out["integrated_lufs"] = float(s.split()[1])
        elif s.startswith("Peak:") and "dBFS" in s:
            # ebur128 prints both "Sample peak" and "True peak" blocks; the last wins
            out["true_peak_dbfs"] = float(s.split()[1])
        elif s.startswith("LRA:") and "LU" in s:
            out.setdefault("lra", float(s.split()[1]))
    if "integrated_lufs" not in out:
        raise RenderError(
            f"could not read loudness from {path}: ffmpeg ebur128 produced no I: line. "
            f"stderr tail: {proc.stderr[-300:]}")
    return out
