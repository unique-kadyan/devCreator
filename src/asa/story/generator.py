"""Three-call story generation with schema repair.

Splitting outline / draft / scenes markedly improves structure adherence and keeps each
response inside free-model context limits. Each call validates against a Pydantic schema; a
failure triggers exactly one repair attempt carrying the validator error back in, and also
demotes that model for structured work so the router stops picking it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from ..core.errors import ValidationError
from ..core.logging import get_logger
from ..llm.base import Completion, parse_model
from ..llm.chain import LLMChain
from . import prompts as P
from .schema import SceneList, StoryOutline

log = get_logger("story")

# Ceiling for the truncation retry. Free models mostly top out between 8k and 16k output
# tokens, so doubling past this buys nothing but a slower failure - at that point the
# request genuinely needs to be smaller, not the budget bigger.
MAX_OUTPUT_TOKENS = 16384

# Narration pace used to turn a word count into an expected runtime. 150 wpm is the
# storytelling rate the prompts are written against; the real figure comes from TTS later,
# but by then the story is fixed and the render budget is already spent.
WORDS_PER_MINUTE = 150.0

# Below this fraction of the requested runtime the story is regenerated rather than
# shipped. Set from measurement, not taste: a run targeting 3 minutes produced 82 seconds
# of audio - 45% - and nothing anywhere noticed. QC only warns under one minute, and it
# only sees the video long after the render has been paid for.
MIN_RUNTIME_RATIO = 0.75


@dataclass
class GeneratedStory:
    outline: StoryOutline
    draft: dict
    scenes: SceneList
    model_ids: dict[str, str] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    repairs: int = 0

    @property
    def word_count(self) -> int:
        text = " ".join(str(v) for v in self.draft.get("beats", {}).values())
        return len(text.split())


class StoryGenerator:
    def __init__(self, chain: LLMChain, target_minutes: float = 7.0,
                 max_new_characters: int = 1, archetypes: list[str] | None = None):
        self.chain = chain
        self.target_minutes = target_minutes
        self.max_new_characters = max_new_characters
        self.archetypes = archetypes or [
            "underdog", "trickster", "redemption", "mystery",
            "friendship", "survival", "comedy", "family"]

    # ---------------------------------------------------------------- helpers

    def _structured(self, user: str, model: type, role: str, label: str,
                    system_extra: str = "", max_tokens: int = 4096,
                    temperature: float = 0.9,
                    post_check: Callable[[object], None] | None = None
                    ) -> tuple[object, Completion, int]:
        """One call, one repair attempt. Returns (parsed, completion, repairs_used).

        `post_check` carries constraints Pydantic cannot express because they depend on
        runtime state (how many characters already exist, which ids are in the cast). It
        raises ValidationError like any field validator, so a violation is *repairable*
        rather than fatal - the model gets told what it broke and tries once more.
        """
        system = P.system_prompt(system_extra)
        c = self.chain.complete(system, user, role=role, max_tokens=max_tokens,
                                temperature=temperature, structured=True)
        try:
            parsed = parse_model(c.text, model)
            if post_check:
                post_check(parsed)
            return parsed, c, 0
        except ValidationError as e:
            if c.meta.get("finish_reason") == "length" and max_tokens < MAX_OUTPUT_TOKENS:
                # Truncation is a BUDGET failure wearing a schema failure's clothes, and
                # the repair path actively makes it worse: the repair prompt restates the
                # original prompt plus the validator error, so it is longer, and it is sent
                # with the same ceiling - it gets cut off in the same place, twice, and the
                # stage fails with "schema failed twice" while the schema was never the
                # problem. Give the original request more room instead.
                bigger = min(max_tokens * 2, MAX_OUTPUT_TOKENS)
                log.warning("output_truncated_retrying_larger", stage=label,
                            model=c.model_id, was=max_tokens, now=bigger)
                c_big = self.chain.complete(system, user, role=role, max_tokens=bigger,
                                            temperature=temperature, structured=True)
                try:
                    parsed = parse_model(c_big.text, model)
                    if post_check:
                        post_check(parsed)
                    return parsed, c_big, 1
                except ValidationError:
                    # Still no good - fall through and let the repair path have its turn,
                    # since the second failure may genuinely be about the schema.
                    c = c_big
            log.warning("schema_failed_repairing", stage=label, model=c.model_id,
                        error=str(e)[:160])
            self.chain.note_schema_failure(c, role)
            repair = (f"{user}\n\n---\nYour previous reply was REJECTED by the schema "
                      f"validator with this error:\n\n{str(e)[:1200]}\n\n"
                      f"Return ONLY corrected JSON. No prose, no markdown fence, no "
                      f"explanation, no reasoning. Fix exactly the reported problems.")
            # The repair prompt is the ORIGINAL prompt plus the validator error, so it is
            # strictly longer than the request that already fit - or already did not. Send
            # it with the same ceiling and it truncates where the first one did. Give the
            # repair headroom from the outset rather than discovering that after the fact.
            repair_budget = min(max(max_tokens, len(repair) // 3), MAX_OUTPUT_TOKENS)
            c2 = self.chain.complete(system, repair, role=role, max_tokens=repair_budget,
                                     temperature=max(0.2, temperature - 0.4),
                                     structured=True)
            try:
                parsed = parse_model(c2.text, model)
                if post_check:
                    post_check(parsed)
                return parsed, c2, 1
            except ValidationError as e2:
                if (c2.meta.get("finish_reason") == "length"
                        and repair_budget < MAX_OUTPUT_TOKENS):
                    # Same trap as the first attempt, one level down. Without this the
                    # stage reports "schema failed twice" for what is purely a budget
                    # problem - which is exactly how job 2 died: model_truncated at 8192,
                    # then EOF-while-parsing reported as a schema failure.
                    bigger = min(repair_budget * 2, MAX_OUTPUT_TOKENS)
                    log.warning("repair_truncated_retrying_larger", stage=label,
                                model=c2.model_id, was=repair_budget, now=bigger)
                    c3 = self.chain.complete(system, repair, role=role, max_tokens=bigger,
                                             temperature=max(0.2, temperature - 0.4),
                                             structured=True)
                    try:
                        parsed = parse_model(c3.text, model)
                        if post_check:
                            post_check(parsed)
                        return parsed, c3, 2
                    except ValidationError:
                        c2 = c3
                self.chain.note_schema_failure(c2, role)
                raise ValidationError(f"{label}: schema failed twice - {e2}") from e2

    # ---------------------------------------------------------------- stages

    def outline(self, topic: str, keywords: list[str], available_characters: list[dict],
                recent_signatures: list[str], under_used: str = "",
                strategy_prefer: str = "", strategy_avoid: str = "") -> tuple:
        # A brand-new channel has nobody to reuse, so a fixed budget of 1 makes almost
        # every story impossible. Grow the budget when the library is thin, then settle
        # back to the configured steady-state ceiling once there is a cast to draw on.
        budget = self.max_new_characters
        if len(available_characters) < 3:
            budget = max(budget, 3 - len(available_characters))
        user = P.outline_prompt(
            topic=topic, keywords=keywords, target_minutes=self.target_minutes,
            available_characters=available_characters, recent_signatures=recent_signatures,
            under_used=under_used, strategy_prefer=strategy_prefer,
            strategy_avoid=strategy_avoid, max_new_characters=budget,
            archetypes=self.archetypes)
        known = {c["id"] for c in available_characters}

        def _check(o) -> None:
            new_chars = sum(1 for m in o.cast if m.new_character_spec)
            if new_chars > budget:
                raise ValidationError(
                    f"cast requests {new_chars} new characters but the budget is {budget}. "
                    f"Reuse existing cast ids ({sorted(known) or 'none available'}) or cut "
                    f"characters until only {budget} have a new_character_spec.")
            unknown = [m.character_id for m in o.cast
                       if m.character_id and m.character_id not in known]
            if unknown:
                raise ValidationError(
                    f"cast references character_id(s) {unknown} that do not exist. "
                    f"Valid ids are {sorted(known) or '(none)'}; anything else must be "
                    f"supplied as a new_character_spec with character_id set to null.")
            roles = [m.role for m in o.cast]
            if roles.count("protagonist") != 1:
                raise ValidationError(
                    f"cast must contain exactly one protagonist, found "
                    f"{roles.count('protagonist')}")

        parsed, c, rep = self._structured(user, StoryOutline, "story", "outline",
                                          max_tokens=3072, post_check=_check)
        return parsed, c, rep, P.sha(user)

    def draft(self, outline: StoryOutline) -> tuple:
        user = P.draft_prompt(outline.model_dump_json(indent=None), self.target_minutes)
        system = P.system_prompt()
        c = self.chain.complete(system, user, role="story", max_tokens=4096,
                                temperature=0.95, structured=True)
        from ..llm.base import extract_json
        try:
            data = json.loads(extract_json(c.text))
            if not isinstance(data, dict) or "beats" not in data:
                raise ValueError("missing 'beats'")
        except Exception:                                          # noqa: BLE001
            # prose is acceptable here - only the scene stage is machine-critical
            log.warning("draft_not_json_using_prose", model=c.model_id)
            data = {"beats": {"beginning": c.text.strip()}}
        return data, c, 0, P.sha(user)

    def scenes(self, outline: StoryOutline, draft: dict, cast: list[dict],
               existing_locations: list[str], sfx_library: list[str],
               expand_from: SceneList | None = None) -> tuple:
        user = P.scenes_prompt(
            story_json=outline.model_dump_json(indent=None),
            draft_json=json.dumps(draft)[:6000],
            cast=cast, existing_locations=existing_locations,
            sfx_library=sfx_library, style=P.block("style_bible"),
            target_minutes=self.target_minutes)
        if expand_from is not None:
            # Re-asking with the same prompt gets the same length back; the model has to be
            # told what it produced and by how much it missed.
            user = self._expand_prompt(user, expand_from,
                                       self.target_minutes * 60.0)
        valid_ids = {m["id"] for m in cast}

        def _check(sl) -> None:
            bad: list[str] = []
            for sc in sl.scenes:
                for d in sc.dialogue:
                    if d.character_id not in valid_ids:
                        bad.append(f"scene {sc.index} dialogue -> {d.character_id!r}")
            if bad:
                raise ValidationError(
                    "dialogue names characters that are not in the cast: "
                    + "; ".join(bad[:8])
                    + f". Every dialogue character_id must be exactly one of "
                      f"{sorted(valid_ids)}.")
            spoken = sum(len(sc.dialogue) for sc in sl.scenes)
            narrated = sum(1 for sc in sl.scenes if sc.narration.strip())
            if spoken + narrated == 0:
                raise ValidationError("no scene has dialogue or narration; the video "
                                      "would be silent")

        # Scene lists are the longest structured output in the pipeline and the one most
        # often truncated. parse_model does NOT salvage a cut-off tail - a JSON object
        # severed mid-string is unparseable - so _structured detects truncation and retries
        # with a larger budget rather than treating it as a schema error.
        parsed, c, rep = self._structured(user, SceneList, "story", "scenes",
                                          max_tokens=8192, temperature=0.7,
                                          post_check=_check)
        # Staging for an unknown id is harmless - drop it rather than burn a repair call.
        for sc in parsed.scenes:
            for cid in list(sc.staging):
                if cid not in valid_ids:
                    sc.staging.pop(cid)
            sc.characters = [c_ for c_ in sc.characters if c_ in valid_ids]
        return parsed, c, rep, P.sha(user)

    # ------------------------------------------------------- runtime shortfall

    @staticmethod
    def spoken_words(scenes: SceneList) -> int:
        """Words that will actually be spoken - narration plus dialogue, nothing else.

        `action` and `visual_prompt` are stage directions: they cost render time but no
        audio, so counting them would flatter the estimate by roughly double and defeat
        the check.
        """
        n = 0
        for sc in scenes.scenes:
            n += len((sc.narration or "").split())
            for line in sc.dialogue:
                n += len((getattr(line, "text", "") or "").split())
        return n

    def estimated_runtime_s(self, scenes: SceneList) -> float:
        return self.spoken_words(scenes) / (WORDS_PER_MINUTE / 60.0)

    def _expand_prompt(self, user: str, scenes: SceneList, want_s: float) -> str:
        have_s = self.estimated_runtime_s(scenes)
        need_words = int(max(0.0, want_s - have_s) * WORDS_PER_MINUTE / 60.0)
        return (f"{user}\n\n---\nYour previous reply was structurally valid but far too "
                f"SHORT. It contains {self.spoken_words(scenes)} spoken words, about "
                f"{have_s / 60:.1f} minutes of narration and dialogue, against a target of "
                f"{want_s / 60:.1f} minutes.\n\n"
                f"Return the SAME story, same scene order, same characters and same "
                f"locations, with roughly {need_words} MORE words of narration and "
                f"dialogue distributed across the existing scenes. Deepen what is already "
                f"there - more of what characters say to each other, more of what the "
                f"narrator observes. Do NOT invent new plot, new characters or new "
                f"locations, and do NOT pad with repetition.\n\n"
                f"Return ONLY the corrected JSON.")

    # ---------------------------------------------------------------- entry

    def generate(self, topic: str, keywords: list[str], available_characters: list[dict],
                 recent_signatures: list[str], existing_locations: list[str] | None = None,
                 sfx_library: list[str] | None = None, under_used: str = "",
                 strategy_prefer: str = "", strategy_avoid: str = "") -> GeneratedStory:
        log.info("story_start", topic=topic[:60])
        outline, c1, r1, h1 = self.outline(
            topic, keywords, available_characters, recent_signatures,
            under_used, strategy_prefer, strategy_avoid)
        log.info("outline_done", title=outline.title, archetype=outline.archetype,
                 model=c1.model_id)

        draft, c2, r2, h2 = self.draft(outline)
        log.info("draft_done", model=c2.model_id)

        cast = []
        for m in outline.cast:
            if m.character_id:
                found = next((a for a in available_characters if a["id"] == m.character_id),
                             None)
                cast.append(found or {"id": m.character_id, "name": m.character_id,
                                      "species": "unknown"})
            elif m.new_character_spec:
                spec = m.new_character_spec
                cast.append({"id": _slug(spec.name, spec.species), "name": spec.name,
                             "species": spec.species, "new": True, "spec": spec})

        scenes, c3, r3, h3 = self.scenes(
            outline, draft, cast, existing_locations or [], sfx_library or [])

        # Runtime is checked here, before art, TTS and rendering are paid for. The scene
        # prompt asks for target_minutes * 150 words and nothing downstream verifies it -
        # so a story asked for 7 minutes could ship at 1:20, which is exactly what job 1
        # did. QC cannot catch it either: it warns under a minute and never compares the
        # result against what was requested.
        want_s = self.target_minutes * 60.0
        have_s = self.estimated_runtime_s(scenes)
        if want_s > 0 and have_s < want_s * MIN_RUNTIME_RATIO:
            log.warning("story_too_short_expanding", estimated_s=round(have_s),
                        target_s=round(want_s), words=self.spoken_words(scenes))
            try:
                expanded, c3b, _, _ = self.scenes(
                    outline, draft, cast, existing_locations or [],
                    sfx_library or [], expand_from=scenes)
                if self.estimated_runtime_s(expanded) > have_s:
                    scenes, c3 = expanded, c3b
                    have_s = self.estimated_runtime_s(scenes)
            except ValidationError as e:
                # An expansion that will not validate is not worth failing the story over:
                # a short episode is a quality problem, a dead job is a worse one.
                log.warning("story_expansion_failed_keeping_original",
                            error=str(e)[:160])
        log.info("scenes_done", scenes=len(scenes.scenes), model=c3.model_id,
                 estimated_runtime_s=round(have_s), target_s=round(want_s))

        return GeneratedStory(
            outline=outline, draft=draft, scenes=scenes,
            model_ids={"outline": c1.model_id, "draft": c2.model_id, "scenes": c3.model_id},
            prompt_hashes={"outline": h1, "draft": h2, "scenes": h3},
            repairs=r1 + r2 + r3)


def _slug(name: str, species: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    return f"{base}_{species}"
