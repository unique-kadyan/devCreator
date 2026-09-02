#!/usr/bin/env python
"""Phase 2: real voice drives the timeline.

Difference from Phase 1: nothing hardcodes a duration any more. Every line is synthesised
with Kokoro, its true length sets the scene length, and its RMS envelope drives the mouth.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from asa.core.config import load_config
from asa.media.animation.grade import light_from_plate
from asa.media.animation.parallel import CharacterSpec, SceneSpec, render_scene_parallel
from asa.media.audio.envelope import rms_envelope
from asa.media.audio.mixer import Placed, measure_loudness, mix, mux
from asa.media.audio.timeline import build_scene_timing
from asa.media.tts.kokoro_tts import KokoroTTS
from asa.media.tts.voices import rate_for

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
FRAME = tuple(CFG.get("production.resolution", [1920, 1080]))
FPS = int(CFG.get("production.fps", 24))
WORLD = (int(FRAME[0] * 1.4), int(FRAME[1] * 1.4))
NARRATOR_VOICE = CFG.get("providers.tts.kokoro_local.narrator_voice", "bm_fable")
MILO = dict(character_id="milo_fox", voice="am_puck", pitch=-1.0, rate=1.02)

# Each line: (speaker, text, emotion). speaker None = narrator.
SCENES = [
    dict(idx=1, shot="wide", camera="push_in", focus=(0.46, 0.60), gesture="idle",
         facing="right", x=0.40, scale=0.72, lines=[
             (None, "Milo had walked this path a hundred times.", "curious"),
             (None, "He had never once been inside.", "curious")]),
    dict(idx=2, shot="medium", camera="pan_right", focus=(0.52, 0.58), gesture="walk_cycle",
         facing="right", x=0.44, scale=0.88, lines=[
             (None, "The bakery had been shuttered since the autumn his aunt left.", "neutral")]),
    dict(idx=3, shot="close_up", camera="static", focus=(0.50, 0.44), gesture="talk",
         facing="right", x=0.50, scale=1.32, lines=[
             ("milo_fox", "Alright, Aunt Rosa. Let's see what you left me.", "wry")]),
    dict(idx=4, shot="full", camera="pull_out", focus=(0.50, 0.58), gesture="react_shock",
         facing="right", x=0.54, scale=0.95, lines=[
             (None, "The shelves were empty. Every single one.", "surprised")]),
    dict(idx=5, shot="two_shot", camera="handheld_drift", focus=(0.48, 0.56), gesture="talk",
         facing="right", x=0.42, scale=0.86, lines=[
             ("milo_fox", "One bag of flour. That's it. That's the whole inheritance.", "wry"),
             (None, "But Milo had something better than flour. He had an idea.", "determined")]),
]


def main() -> int:
    work = ROOT / "data/work/phase2"
    (work / "audio").mkdir(parents=True, exist_ok=True)
    rig_path = ROOT / "assets/characters/milo_fox/rig.json"
    plate_path = ROOT / "assets/backgrounds/forest_village/plate.png"

    tint, strength = light_from_plate(Image.open(plate_path))
    print(f"Phase 2: {FRAME[0]}x{FRAME[1]}@{FPS}, scene light {tint} @ {strength:.2f}")
    print("Synthesising voice (Kokoro-82M, local CPU)\n")
    tts = KokoroTTS()
    t0 = time.time()
    timings, audio_total = [], 0.0
    for spec in SCENES:
        utts = []
        for j, (speaker, text, emotion) in enumerate(spec["lines"]):
            voice = MILO["voice"] if speaker else NARRATOR_VOICE
            pitch = MILO["pitch"] if speaker else 0.0
            speed = (MILO["rate"] if speaker else 1.0) * rate_for(emotion)
            out = work / "audio" / f"s{spec['idx']:02d}_l{j:02d}.wav"
            u = tts.synthesize(text, voice, out, speed=speed, pitch_semitones=pitch,
                               character_id=speaker, cache_dir=ROOT / "data/cache/tts")
            u.envelope = rms_envelope(u.path, fps=FPS)
            u.envelope_fps = FPS
            utts.append((u, speaker))
            audio_total += u.duration_s
            hit = "cache" if u.provider.endswith("+cache") else "synth"
            print(f"  s{spec['idx']} l{j}  {u.duration_s:5.2f}s  {voice:<10} "
                  f"{emotion:<11} [{hit}] {text[:46]}")
        timings.append(build_scene_timing(spec["idx"], utts))
    synth_s = time.time() - t0
    print(f"\n  synthesised {audio_total:.1f}s of speech in {synth_s:.1f}s "
          f"({audio_total/synth_s:.2f}x realtime)")

    total = sum(t.duration_s for t in timings)
    print(f"  timeline: {total:.1f}s across {len(timings)} scenes "
          f"(durations derived from audio, not estimated)\n")

    parts, stats = [], []
    for spec, tm in zip(SCENES, timings):
        out = work / f"scene_{spec['idx']:03d}.mp4"
        # the fox's mouth moves only for his own lines; narration does not move it
        env = tm.envelope_for("milo_fox", FPS)
        sspec = SceneSpec(
            plate_path=str(plate_path),
            characters=[CharacterSpec(
                rig_path=str(rig_path), x=spec["x"], y=0.93, scale=spec["scale"],
                facing=spec["facing"], gesture=spec["gesture"],
                blink_seed=spec["idx"] * 101, envelope=env, envelope_fps=FPS,
                grade_rgb=tint, grade_strength=strength)],
            duration=tm.duration_s, world=WORLD, frame=FRAME, fps=FPS,
            camera_move=spec["camera"], shot_from=spec["shot"], focus=spec["focus"])
        st = render_scene_parallel(sspec, out)
        st["scene"] = spec["idx"]
        stats.append(st)
        parts.append(out)
        print(f"  scene {spec['idx']}  {tm.duration_s:5.2f}s  {spec['shot']:<10}"
              f"{spec['camera']:<16} -> {st['seconds']:5.1f}s "
              f"({st['realtime_factor']}x realtime)")

    # ---- assemble picture, then mix one continuous audio timeline (no per-scene seams)
    concat = work / "concat.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in parts))
    silent = work / "picture.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-c", "copy", str(silent)], check=True)

    voice_bus, clock = [], 0.0
    for tm in timings:
        for cue in tm.cues:
            voice_bus.append(Placed(path=cue.utterance.path, start_s=clock + cue.start_s))
        clock += tm.duration_s

    from asa.media.audio.music import MusicLibrary
    lib = MusicLibrary(ROOT / "assets/music", db=ROOT / "data/asa.db")
    bed = lib.bed_for("curious", total, work / "bed.wav", seed=1)
    music_bus = [Placed(path=bed, start_s=0.0, gain_db=-16.0)] if bed else []
    if not music_bus:
        print("\n  NOTE: assets/music/ is empty - mixing voice only. Build the licensed "
              "music library (docs/07 §8), then `asa assets add`, before publishing.")

    mixdown = work / "mix.wav"
    mix(voice_bus, music_bus, [], duration_s=total, out_path=mixdown)
    loud = measure_loudness(mixdown)

    final = ROOT / "data/out/phase2_slice.mp4"
    mux(silent, mixdown, final)

    render_s = sum(s["seconds"] for s in stats)
    print(f"\n  loudness: {loud.get('integrated_lufs')} LUFS integrated, "
          f"true peak {loud.get('true_peak_dbfs')} dBFS  (target -14 / -1)")
    print(f"  TOTAL  {total:.1f}s video | {synth_s:.0f}s TTS + {render_s:.0f}s render "
          f"= {synth_s+render_s:.0f}s ({total/(synth_s+render_s):.2f}x realtime)")
    print(f"  OUTPUT {final}  ({final.stat().st_size/1e6:.2f} MB)")
    (work / "stats.json").write_text(json.dumps(
        {"scenes": stats, "loudness": loud, "duration_s": total,
         "tts_seconds": round(synth_s, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
