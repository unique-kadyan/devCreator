#!/usr/bin/env python
"""Phase 1 vertical slice: a hardcoded scene list -> one rendered mp4.

Proves the compositor, camera, parallax, blink/viseme automata and the ffmpeg pipe
before any of the generative stages exist. No LLM calls, no TTS yet (Phase 2).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asa.media.animation.parallel import CharacterSpec, SceneSpec, render_scene_parallel

ROOT = Path(__file__).resolve().parents[1]
WORLD = (2688, 1512)
FRAME = (1920, 1080)
FPS = 24

# --- the hand-written "story", matching the scene contract in docs/02 §3 ----------
SCENES = [
    dict(idx=1, duration=6.0, shot="wide", camera="push_in", focus=(0.46, 0.60),
         gesture="idle", facing="right", x=0.40, scale=0.72, speech=[(1.6, 5.2)],
         line="Milo had walked this path a hundred times."),
    dict(idx=2, duration=5.5, shot="medium", camera="pan_right", focus=(0.52, 0.58),
         gesture="walk_cycle", facing="right", x=0.44, scale=0.88, speech=[],
         line="(walks toward the shuttered bakery)"),
    dict(idx=3, duration=6.5, shot="close_up", camera="static", focus=(0.50, 0.44),
         gesture="talk", facing="right", x=0.50, scale=1.32, speech=[(0.7, 6.0)],
         line="Alright, Aunt Rosa. Let's see what you left me."),
    dict(idx=4, duration=5.0, shot="full", camera="pull_out", focus=(0.50, 0.58),
         gesture="react_shock", facing="left", x=0.54, scale=0.95, speech=[],
         line="(the shelves are completely empty)"),
    dict(idx=5, duration=7.0, shot="two_shot", camera="handheld_drift", focus=(0.48, 0.56),
         gesture="talk", facing="right", x=0.42, scale=0.86, speech=[(0.9, 6.4)],
         line="He had one bag of flour, and a very good idea."),
]


def build_spec(spec: dict, rig_path: Path, plate_path: Path, final: bool) -> SceneSpec:
    return SceneSpec(
        plate_path=str(plate_path),
        characters=[CharacterSpec(
            rig_path=str(rig_path), x=spec["x"], y=0.93, scale=spec["scale"],
            facing=spec["facing"], gesture=spec["gesture"],
            speech=spec["speech"], blink_seed=spec["idx"] * 101,
        )],
        duration=spec["duration"], world=WORLD, frame=FRAME, fps=FPS,
        camera_move=spec["camera"], shot_from=spec["shot"], focus=spec["focus"],
        easing="in_out", final=final,
    )


def main() -> int:
    final = "--draft" not in sys.argv
    rig_path = ROOT / "assets/characters/milo_fox/rig.json"
    plate_path = ROOT / "assets/backgrounds/forest_village/plate.png"
    work = ROOT / "data/work/phase1"
    work.mkdir(parents=True, exist_ok=True)

    total = sum(s["duration"] for s in SCENES)
    print(f"Phase 1 slice: {len(SCENES)} scenes, {total:.1f}s, "
          f"{FRAME[0]}x{FRAME[1]}@{FPS} ({'final' if final else 'draft'} resampling)\n")

    stats, parts = [], []
    for spec in SCENES:
        out = work / f"scene_{spec['idx']:03d}.mp4"
        print(f"  scene {spec['idx']}  {spec['duration']:.1f}s  {spec['shot']:<10} "
              f"{spec['camera']:<15} {spec['gesture']}")
        st = render_scene_parallel(build_spec(spec, rig_path, plate_path, final), out)
        st["scene"] = spec["idx"]
        stats.append(st)
        parts.append(out)
        print(f"           -> {st['seconds']}s render, {st['ms_per_frame']} ms/frame, "
              f"{st['realtime_factor']}x realtime ({st['workers']} workers)")

    concat = work / "concat.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in parts))
    final_path = ROOT / "data/out/phase1_slice.mp4"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-c", "copy", str(final_path)], check=True)

    render_s = sum(s["seconds"] for s in stats)
    print(f"\n  TOTAL  {total:.1f}s of video in {render_s:.1f}s of render "
          f"({total/render_s:.2f}x realtime)")
    print(f"  OUTPUT {final_path}  ({final_path.stat().st_size/1e6:.2f} MB)")
    (work / "stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
