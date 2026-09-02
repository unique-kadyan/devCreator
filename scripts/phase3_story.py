#!/usr/bin/env python
"""Phase 3: topic string -> full validated script, via the adaptive free-model router."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asa.core.config import load_config
from asa.core.logging import setup_logging
from asa.llm.factory import build_chain
from asa.llm.router import ModelRouter
from asa.story.generator import StoryGenerator

ROOT = Path(__file__).resolve().parents[1]

AVAILABLE = [dict(id="milo_fox", name="Milo", species="fox", age_band="young_adult",
                  personality="Clever, curious, kind; talks himself into trouble and out "
                              "of it again.", voice_id="am_puck")]
RECENT = ["trickster: inherit|lack|deceive|confess|repair"]


def main() -> int:
    setup_logging("INFO")
    cfg = load_config()
    topic = sys.argv[1] if len(sys.argv) > 1 else "a clever fox opens a village bakery"
    chain = build_chain(cfg)
    gen = StoryGenerator(chain,
                         target_minutes=float(cfg.get("production.target_minutes", 7)),
                         max_new_characters=int(
                             cfg.get("story.max_new_characters_per_story", 1)),
                         archetypes=cfg.get("story.archetypes"))
    t0 = time.time()
    story = gen.generate(topic=topic, keywords=["animal story", "moral", "small business"],
                         available_characters=AVAILABLE, recent_signatures=RECENT,
                         existing_locations=["forest_village"],
                         sfx_library=["leaves_rustle", "door_creak", "footsteps_dirt"])
    dt = time.time() - t0

    o = story.outline
    print(f"\n{'='*72}\n  {o.title}\n{'='*72}")
    print(f"  hook       {o.hook}")
    print(f"  logline    {o.logline}")
    print(f"  archetype  {o.archetype}   audience: {o.target_audience}")
    print(f"  moral      {o.moral}")
    print(f"  signature  {o.beat_signature}")
    print(f"  cast       " + ", ".join(
        f"{m.character_id or m.new_character_spec.name}({m.role})" for m in o.cast))
    print(f"\n  scenes: {len(story.scenes.scenes)}   words: {story.word_count}   "
          f"repairs: {story.repairs}   {dt:.1f}s")
    print(f"  models: {story.model_ids}\n")
    for s in story.scenes.scenes[:4]:
        line = s.dialogue[0].line if s.dialogue else ""
        print(f"  {s.index:2d}  {s.duration_hint_s:4.1f}s  {s.location_id:<18}"
              f"{s.shot:<12}{s.camera.move:<15}{s.emotion}")
        if s.narration:
            print(f"      narration: {s.narration[:78]}")
        if line:
            print(f"      {s.dialogue[0].character_id}: \"{line[:66]}\"")
    if len(story.scenes.scenes) > 4:
        print(f"  ... {len(story.scenes.scenes)-4} more scenes")

    out = ROOT / "data/work/phase3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "story.json").write_text(json.dumps({
        "outline": o.model_dump(), "draft": story.draft,
        "scenes": story.scenes.model_dump(), "model_ids": story.model_ids,
        "prompt_hashes": story.prompt_hashes, "repairs": story.repairs,
        "seconds": round(dt, 1)}, indent=2, default=str))
    print(f"\n  saved {out/'story.json'}")

    print("\n  model health after this run:")
    for r in ModelRouter(ROOT / "data/asa.db").report("story", limit=8):
        cold = " COLD" if r["cold_until"] else ""
        print(f"    {r['model_id']:<46} calls={r['calls']:<3} ok={r['successes']:<3} "
              f"429={r['rate_limits']:<3} err={r['errors']:<3} "
              f"{r['avg_s']:.1f}s{cold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
