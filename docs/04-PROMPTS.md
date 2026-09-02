# 04 — Prompt Architecture

## Principles

1. **Every prompt is a file in `prompts/`, versioned in git, hashed into the `prompts` table.**
   You cannot attribute a quality regression to a prompt change unless the exact prompt text and
   `model_id` are recorded with the story.
2. **Closed vocabularies, not free text.** Any field the renderer consumes (`camera.move`,
   `shot`, `gesture`, `transition`, `emotion`) is an enum. The model picks from a list; it never
   invents a value the compositor can't render.
3. **Composed from blocks**, so shared context is written once:
   `SYSTEM = channel_bible + style_bible + safety_rules`, then a task block.
4. **Structured output validated by Pydantic**, with exactly one repair attempt carrying the
   validator error back in. Two failures = stage failure, not a silent bad script.
5. **Negative constraints are data, not vibes.** The "avoid these plots" block is generated from
   the `stories` table, and the "prefer/avoid" block from the `strategy` table.
6. **Budget-aware.** Under a 50 req/day free cap, prompts are batched: 20 topics scored in one
   call, 8 titles generated in one call.

## Block layout

```
prompts/
├── _blocks/
│   ├── channel_bible.md      # audience, tone, values, what this channel is and is not
│   ├── style_bible.md        # the visual style token block — also hashed into characters.style_hash
│   ├── safety_rules.md       # originality + policy constraints, injected into every generative call
│   └── vocab.md              # the closed enums, rendered into prompts programmatically
├── story/    01_outline.md  02_draft.md  03_scenes.md  04_repair.md
├── scene/    visual_prompt.md
├── metadata/ titles.md  description.md  tags.md
├── thumbnail/ concepts.md  truthfulness_check.md
└── qc/       safety_review.md  similarity_explain.md
```

## `_blocks/safety_rules.md` (injected everywhere)

```markdown
Originality and policy constraints — these override all other instructions:

- Invent all characters, names, places and plots. Do NOT reference, imitate, parody or evoke
  any existing franchise, studio, film, series, game, or well-known character — including by
  describing them without naming them.
- Do not reuse a real person's name, likeness or identifying traits.
- Do not produce: graphic violence or injury, blood, weapons used against characters, death
  depicted on screen, cruelty played for laughs, sexual or romantic-adult content, alcohol,
  drugs, gambling, hateful or demeaning content about any group, dangerous acts a child could
  imitate, medical/financial/legal claims, or real-world political or religious commentary.
- Peril is allowed and encouraged; harm depicted on screen is not. A character may be lost,
  scared, hungry, excluded or unfairly treated. A character may not be hurt on camera.
- Emotional stakes must resolve. Do not end on despair.
- Where a character's pronouns are not specified, use they/them.
- Write nothing you would not be comfortable defending as your own original work.
```

## `story/01_outline.md` (abridged)

```markdown
{{channel_bible}}
{{safety_rules}}

Create the outline for one original animated short story.

SEED TOPIC: {{topic}}
THEME KEYWORDS: {{keywords}}
TARGET RUNTIME: {{target_minutes}} minutes  (~{{target_words}} words of narration + dialogue)

AVAILABLE CAST (prefer these — reusing them costs nothing and builds continuity):
{{#each available_characters}}
- {{id}} — {{name}}, {{species}}, {{age_band}}. {{personality}}. Voice: {{voice_id}}.
{{/each}}

You may request AT MOST {{max_new_characters}} new character(s). Requesting a new character is
expensive, so only do it if the story genuinely cannot work without one.

DO NOT retell any of these previously used plots:
{{#each recent_beat_signatures}}
- {{archetype}}: {{signature}}
{{/each}}

UNDER-USED COMBINATIONS (prefer one of these unless the seed topic clearly points elsewhere):
{{under_used_combinations}}

AUDIENCE PERFORMANCE SIGNAL (from this channel's own analytics — advisory, not binding):
  prefer: {{strategy_prefer}}
  avoid:  {{strategy_avoid}}

Return JSON matching this schema exactly:
{
  "title": str,                    // <= 60 chars, no clickbait punctuation
  "hook": str,                     // the first 3-10 seconds, as a single sentence
  "logline": str,
  "target_audience": str,
  "genre": str,
  "archetype": one of {{archetypes}},
  "moral": str,                    // a real lesson, stated plainly, not a platitude
  "setting": str,
  "beats": {"beginning": str, "conflict": str, "rising": str, "climax": str, "resolution": str},
  "ending": str,
  "beat_signature": str,           // 5 verbs, pipe-separated, e.g. "inherit|lack|deceive|confess|repair"
  "cast": [{"character_id": str|null, "role": str, "new_character_spec": object|null}]
}
```

## `story/03_scenes.md` — the renderer contract

```markdown
{{style_bible}}

Break this story into scenes for a 2D cutout-animation pipeline.

STORY: {{story_json}}
CAST RIGS (these are the ONLY poses available — do not describe motion outside this list):
  gestures: {{gesture_enum}}
  shots:    {{shot_enum}}
  camera:   {{camera_enum}}
  transitions: {{transition_enum}}
LOCATIONS ALREADY DRAWN (reuse these where possible — new locations cost image credits):
{{existing_locations}}

Rules:
- 8-16 seconds per scene; the FIRST scene must land the hook within its first 3 seconds.
- Every line of dialogue names exactly one character_id from the cast.
- `staging` gives each present character x,y in [0,1] screen space, scale, and facing.
  Characters must not overlap by more than 15%.
- `visual_prompt` describes the LOCATION ONLY. Never describe the characters — they are drawn
  from fixed assets and any description of them will be ignored.
- `sfx` uses short snake_case tags. Prefer tags already in the library: {{sfx_library}}
- Return a JSON array of Scene objects.
```

> The `visual_prompt` rule is load-bearing. It is what makes character consistency structural
> rather than aspirational: the image model is never asked to draw a character, so it can never
> draw one inconsistently.

## `metadata/titles.md` (one call, 8 candidates)

```markdown
{{channel_bible}}

Here is the finished script: {{script}}

Produce 8 title candidates. For EACH, self-score 0-1 on: curiosity, clarity, emotional_appeal,
search_relevance (overlap with: {{keywords}}), click_potential, and accuracy.

`accuracy` means: does everything the title implies actually happen in this script? A title that
implies an event not in the script must be scored 0. Do not use "you won't believe",
"gone wrong", "SHOCKING", invented numbers, ALL-CAPS beyond one word, or emoji in the title.

Return JSON: [{"title": str, "scores": {...}, "accuracy_justification": str}]
```

## Cost per video (OpenRouter free tier, 20 RPM / 50 RPD floor)

| Call | Count | Notes |
|---|---|---|
| Topic rubric scoring | 0.05 | one call per 20 candidates, amortised |
| Story outline | 1 | |
| Story draft | 1 | |
| Scene breakdown | 1 | largest output, ~2.5k tokens |
| Repair (expected) | ~0.5 | only on schema failure |
| Thumbnail concepts | 1 | |
| Thumbnail truthfulness check | 1 | |
| Title/description/tags | 1 | batched |
| QC safety review | 1 | |
| Similarity explanation | ~0.3 | only when flagged |
| **Total** | **~8–9 calls** | **≈ 5 videos/day within the 50 RPD floor** |

That is roughly 10× more headroom than this hardware can render, so the free tier is genuinely
sufficient — which is the point.
