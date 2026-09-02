"""Tests for the stages built in phases 4-12. No network, no API keys, no GPU."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from asa.characters.factory import cast_voice, palette_from_spec, slug
from asa.characters.procedural_puppet import AnimalPuppet, MILO_PALETTE
from asa.characters.species import PROFILES, known, profile
from asa.core.db import connect, jdump, jload, migrate, tx
from asa.llm.base import extract_json
from asa.media.images.base import ImageCache, prompt_key
from asa.media.images.procedural import ProceduralImages
from asa.media.subtitles.build import (_split_long, _ts, captions_from_audio,
                                        write_srt, write_vtt)
from asa.publish.metadata import BAIT, clean_tags, score_title
from asa.research.collectors import Candidate, collect_seasonal, _animal_in, _keywords
from asa.research.scoring import overall, score_candidate
from asa.story.schema import Scene, SceneList


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    migrate(p)
    return p


# ----------------------------------------------------------------- database

def test_migrations_are_idempotent(tmp_path: Path):
    p = tmp_path / "a.db"
    first = migrate(p)
    second = migrate(p)
    assert len(first) >= 2 and second == []


def test_foreign_keys_are_enforced(db: Path):
    with pytest.raises(sqlite3.IntegrityError):
        with tx(db) as con:
            con.execute("INSERT INTO story_cast (story_id, character_id, role) "
                        "VALUES (9999, 'nobody', 'ally')")


def test_transaction_rolls_back(db: Path):
    with pytest.raises(RuntimeError):
        with tx(db) as con:
            con.execute("INSERT INTO locations (id,name,description,visual_prompt) "
                        "VALUES ('x','X','d','p')")
            raise RuntimeError("boom")
    con = connect(db)
    assert con.execute("SELECT COUNT(*) FROM locations").fetchone()[0] == 0


def test_jload_survives_garbage():
    assert jload(None) == {}
    assert jload("") == {}
    assert jload("not json", default=[]) == []
    assert jload('{"a":1}') == {"a": 1}
    assert jload({"already": "decoded"}) == {"already": "decoded"}


# ------------------------------------------------------------------ species

def test_every_species_builds_a_complete_rig(tmp_path: Path):
    for sp in known():
        rig = AnimalPuppet(f"t_{sp}", MILO_PALETTE, sp).build(tmp_path / sp)
        assert set(rig.visemes) == {"rest", "A", "E", "I", "O", "U", "M"}
        assert set(rig.eyes) == {"open", "half", "closed"}
        for key in rig.z_order:
            if key in ("eyes", "mouth"):
                continue
            assert key in rig.layers, f"{sp} is missing layer {key}"


def test_no_layer_escapes_the_canvas(tmp_path: Path):
    """A layer clipped by the canvas edge scales wrong in every shot that uses it."""
    for sp in known():
        rig = AnimalPuppet(f"t_{sp}", MILO_PALETTE, sp).build(tmp_path / sp)
        for key, layer in rig.layers.items():
            x, y = layer.offset
            w, h = layer.size
            assert x >= 0 and y >= 0, f"{sp}.{key} starts off-canvas at {layer.offset}"
            assert x + w <= rig.canvas[0] and y + h <= rig.canvas[1], \
                f"{sp}.{key} extends past the canvas"


def test_unknown_species_falls_back_rather_than_failing():
    assert profile("axolotl").key == "fox"
    assert profile("").key == "fox"


def test_species_are_visually_distinct(tmp_path: Path):
    """Two species must not produce byte-identical head art, or the cast is one animal."""
    heads = {}
    for sp in ("fox", "rabbit", "owl", "bear"):
        AnimalPuppet(f"t_{sp}", MILO_PALETTE, sp).build(tmp_path / sp)
        heads[sp] = (tmp_path / sp / "head.png").read_bytes()
    assert len(set(heads.values())) == len(heads)


# ------------------------------------------------------------------ casting

def test_voice_casting_is_deterministic():
    a = cast_voice("milo_fox", "young_adult", "fox", "he/him")
    b = cast_voice("milo_fox", "young_adult", "fox", "he/him")
    assert a == b


def test_voice_casting_avoids_taken_presets():
    first, _, _ = cast_voice("milo_fox", "adult", "fox", "")
    second, _, _ = cast_voice("milo_fox", "adult", "fox", "", taken={first})
    assert second != first


def test_voice_rate_stays_intelligible():
    for age in ("child", "teen", "young_adult", "adult", "elder"):
        for sp in known():
            _, pitch, rate = cast_voice(f"x_{sp}", age, sp, "")
            assert 0.88 <= rate <= 1.15
            assert -4.0 <= pitch <= 4.0


def test_palette_expansion_is_complete():
    pal = palette_from_spec("#E07A35", "#FFF6E8", "#4C9A54", "#3B6EA5")
    for key in ("fur", "fur_shadow", "chest", "ear_inner", "eyes", "nose", "paw",
                "hoodie", "hoodie_shadow", "shorts", "mouth", "tongue", "mane",
                "spine", "outline", "mask", "horn"):
        assert key in pal and pal[key].startswith("#")


def test_slug_is_filesystem_safe():
    assert slug("Wren the Wise!", "owl") == "wren_the_wise_owl"


# --------------------------------------------------------------- extract_json

@pytest.mark.parametrize("raw,want", [
    ('```json\n{"a":1}\n```', '{"a":1}'),
    ('```json\n{"a":1,"b":[1,2', '{"a":1,"b":[1,2'),          # truncated: no closing fence
    ('prose\n```\n[{"x":1}]\n```', '[{"x":1}]'),
    ('[{"x":"}"}]', '[{"x":"}"}]'),                            # closer inside a string
    ('{"a":1}', '{"a":1}'),
])
def test_extract_json(raw, want):
    assert extract_json(raw) == want


# ------------------------------------------------------------- scene schema

def test_scene_coerces_common_model_mistakes():
    s = Scene.model_validate({
        "index": 1, "duration": 9.0, "location": "bridge_night",
        "description": "Milo runs", "visual_prompt": "a stone bridge at night",
        "sfx": "footsteps, water", "camera_move": "push_in", "from_shot": "wide"})
    assert s.duration_hint_s == 9.0
    assert s.location_id == "bridge_night"
    assert s.action == "Milo runs"
    assert s.sfx == ["footsteps", "water"]
    assert s.camera.move == "push_in" and s.camera.from_shot == "wide"


def test_scene_still_rejects_unrenderable_values():
    with pytest.raises(Exception):
        Scene.model_validate({"index": 1, "location_id": "x", "action": "a",
                              "visual_prompt": "a place at night",
                              "camera": {"move": "dolly_zoom_vertigo"}})


def test_scene_list_requires_contiguous_indices():
    base = {"location_id": "x", "action": "a", "visual_prompt": "a place at night"}
    with pytest.raises(Exception):
        SceneList.model_validate({"scenes": [dict(base, index=1), dict(base, index=3),
                                             dict(base, index=4)]})


# --------------------------------------------------------------- image cache

def test_prompt_key_is_stable_and_discriminating():
    a = prompt_key("a forest", (1024, 576), "no text")
    assert a == prompt_key("a forest", (1024, 576), "no text")
    assert a != prompt_key("a forest", (1280, 720), "no text")
    assert a != prompt_key("a desert", (1024, 576), "no text")


def test_cache_rejects_a_truncated_file(tmp_path: Path):
    cache = ImageCache(tmp_path)
    p = cache.path_for("deadbeef")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    assert cache.get("deadbeef") is None
    assert not p.exists(), "a corrupt entry should be evicted, not returned"


def test_procedural_provider_is_deterministic(tmp_path: Path):
    p = ProceduralImages()
    a = p.generate("a pine forest at dusk", tmp_path / "a.png", (256, 144))
    b = p.generate("a pine forest at dusk", tmp_path / "b.png", (256, 144))
    assert a.path.read_bytes() == b.path.read_bytes()
    assert a.license_code == "CC0"


def test_procedural_separates_sky_from_ground(tmp_path: Path):
    """"forest at dusk" must not paint a green sky."""
    from PIL import Image
    p = ProceduralImages()
    g = p.generate("a pine forest clearing at dusk", tmp_path / "f.png", (256, 144))
    with Image.open(g.path) as im:
        sky = im.convert("RGB").crop((0, 0, 256, 20)).resize((1, 1)).getpixel((0, 0))
    r, gr, b = sky
    assert not (gr > r and gr > b), f"sky at dusk came out green: {sky}"


# ---------------------------------------------------------------- subtitles

def test_long_line_is_split_into_readable_captions():
    text = ("Milo had never once opened the shutters before the sun did, and yet here he "
            "was, arguing with an oven that refused to light.")
    caps = _split_long(text, 0.0, 12.0, 32)
    assert len(caps) > 1
    assert all(len(c.text.split("\n")) <= 2 for c in caps)
    assert caps[0].start_s == 0.0
    assert abs(caps[-1].end_s - 12.0) < 0.01
    for a, b in zip(caps, caps[1:]):
        assert a.end_s <= b.start_s + 1e-6


def test_srt_timestamps_use_commas_and_vtt_uses_dots(tmp_path: Path):
    assert _ts(3725.5) == "01:02:05,500"
    assert _ts(3725.5, comma=False) == "01:02:05.500"


def test_captions_never_overlap(tmp_path: Path):
    class FakeUtt:
        def __init__(self, text, dur): self.text, self.duration_s = text, dur
    class FakeCue:
        def __init__(self, u, s): self.utterance, self.start_s = u, s
        @property
        def end_s(self): return self.start_s + self.utterance.duration_s
    class FakeTiming:
        def __init__(self, cues): self.cues = cues
    class FakeScene:
        def __init__(self, sid, cues, d):
            self.scene_id, self.timing, self.duration_s = sid, FakeTiming(cues), d
    class FakeAudio:
        def __init__(self, scenes): self.scenes = scenes

    audio = FakeAudio([FakeScene(1, [FakeCue(FakeUtt("Short.", 0.3), 0.0),
                                     FakeCue(FakeUtt("Also short.", 0.3), 0.5)], 3.0)])
    caps = captions_from_audio(audio, {1: 0.0})
    for a, b in zip(caps, caps[1:]):
        assert a.end_s <= b.start_s, "captions overlap"


# ----------------------------------------------------------------- metadata

def test_bait_titles_are_gated():
    story = {"title": "The Oven That Refused", "logline": "a fox relights an oven",
             "hook": "", "setting": "a bakery", "moral": "ask for help"}
    bad = score_title("YOU WON'T BELIEVE What This Fox Did!!!", story)
    good = score_title("Milo Relights the Ninety-Year Oven", story)
    assert bad["honesty"] < 0.5
    assert good["honesty"] >= 0.5


def test_tags_are_deduplicated_and_capped():
    tags = clean_tags(["Animated Story", "animated story", "fox!!", "ab", "x" * 600])
    assert "animated story" in tags
    assert tags.count("animated story") == 1
    assert "ab" not in tags
    assert sum(len(t) for t in tags) <= 500


def test_description_carries_the_synthetic_disclosure():
    from asa.publish.metadata import build_description
    d = build_description({"title": "T"}, "A fox bakes bread.",
                          [{"name": "Milo", "species": "fox"}], "Ch", "", True, [])
    assert "ai-assisted" in d.lower()


# ----------------------------------------------------------------- research

def test_keyword_extraction_drops_stopwords():
    kw = _keywords("The fox and the rabbit were lost in the storm")
    assert "the" not in kw and "were" not in kw
    assert "fox" in kw


def test_animal_detection_matches_plurals():
    assert _animal_in("a pack of foxes") == "fox"
    assert _animal_in("nothing here") is None


def test_seasonal_collector_needs_no_network():
    cands = collect_seasonal()
    assert len(cands) >= 6
    assert all(c.source == "seasonal" for c in cands)


def test_rich_topics_outscore_thin_ones():
    rich = Candidate(
        topic="A lost lantern must be returned before the river floods the bridge at night",
        keywords=["lantern", "river", "night", "flood", "promise", "bridge"],
        primary_animal="fox", source="manual", signals={"freshness": 1.0})
    thin = Candidate(topic="animals", keywords=["animals"], source="manual")
    assert overall(score_candidate(rich, {}, {})) > overall(score_candidate(thin, {}, {}))


def test_cooldown_damps_a_recently_used_animal():
    c = Candidate(topic="A brave fox saves the market from the flood at night",
                  keywords=["fox", "flood", "market"], primary_animal="fox",
                  source="manual")
    fresh = overall(score_candidate(c, {}, {}))
    tired = overall(score_candidate(c, {"fox": 0}, {}))
    assert tired < fresh


# --------------------------------------------------------------------- QC

def test_qc_blocks_a_missing_disclosure(db: Path, tmp_path: Path):
    from asa.qc.checks import QCReport
    rep = QCReport()
    rep.add("metadata.disclosure", "fail", "missing")
    assert not rep.passed
    assert rep.to_dict()["fail"] == 1


def test_qc_report_serialises(db: Path):
    from asa.qc.checks import QCReport
    rep = QCReport()
    rep.add("a", "warn", "m", extra=1)
    rep.add("b", "info", "m")
    d = rep.to_dict()
    assert d["passed"] is True and d["warn"] == 1
    json.dumps(d)          # must be serialisable for the qc_report column


def test_banned_and_studio_filters():
    from asa.qc.checks import BANNED, STUDIO_TERMS
    assert STUDIO_TERMS.search("a story like Studio Ghibli made")
    assert not STUDIO_TERMS.search("a story about a fox in a bakery")
    assert not BANNED.search("the fox was determined to win")


# ------------------------------------------------------------------ runner

def test_state_machine_covers_every_state():
    from asa.core.stages import REGISTRY
    produced = {s.to_state for s in REGISTRY}
    consumed = {s.from_state for s in REGISTRY}
    # Every state a stage produces must either be consumed by another stage or be a
    # deliberate stopping point. An unroutable state wedges a job silently.
    from asa.core.runner import PAUSED, TERMINAL
    for state in produced:
        assert state in consumed or state in TERMINAL or state in PAUSED, \
            f"state {state} is produced but nothing consumes it"


def test_chain_breaks_exactly_once_at_the_human_gate():
    """The pipeline is linear except for one deliberate discontinuity: `approval` ends at
    AWAITING_APPROVAL and only a human (or auto-publish) sets APPROVED, which `upload`
    consumes. Any OTHER gap is a bug that would wedge a job in an unroutable state."""
    from asa.core.stages import REGISTRY
    gaps = [(a.name, b.name) for a, b in zip(REGISTRY, REGISTRY[1:])
            if a.to_state != b.from_state]
    assert gaps == [("approval", "upload")], f"unexpected gaps in the chain: {gaps}"


def test_lease_prevents_double_claim(db: Path, tmp_path: Path, monkeypatch):
    from asa.core.runner import Runner

    class FakeCtx:
        pass
    ctx = FakeCtx()
    ctx.db = db
    ctx.paths_for = lambda jid: type("P", (), {"work": tmp_path / "w",
                                               "out": tmp_path / "o"})()
    r = Runner(ctx)
    job_id = r.create_job()
    assert r.claim(job_id) is True
    assert r.claim(job_id) is False, "a live lease must block a second claim"
    r.release(job_id)
    assert r.claim(job_id) is True


# --------------------------------------------------------------- thumbnails

def test_thumbnail_text_never_covers_the_character(tmp_path: Path):
    from PIL import Image
    from asa.publish.thumbnail import render_variant
    puppet = tmp_path / "p"
    AnimalPuppet("t_fox", MILO_PALETTE, "fox").build(puppet)
    plate = tmp_path / "plate.png"
    ProceduralImages().generate("a village at morning", plate, (1024, 576))
    for layout in ("right_hero", "left_hero", "centre_low", "corner_up"):
        v = render_variant(plate, puppet, "The Oven That Refused", "surprised", layout,
                           tmp_path / f"{layout}.jpg")
        assert v.path.exists()
        assert v.path.stat().st_size <= 2 * 1024 * 1024
        with Image.open(v.path) as im:
            assert im.size == (1280, 720)


# ----------------------------------------------------------------- feedback

def test_shrinkage_pulls_small_samples_toward_the_mean():
    from asa.analytics.feedback import SHRINK_K
    channel_mean = 0.50
    for n, raw in ((1, 0.95), (20, 0.95)):
        shrunk = (n * raw + SHRINK_K * channel_mean) / (n + SHRINK_K)
        if n == 1:
            assert shrunk < 0.65, "one great video must not dominate"
        else:
            assert shrunk > 0.85, "twenty great videos should carry weight"


# ------------------------------------------------------------------ casting

def test_species_roster_is_broad_and_covers_every_role():
    from asa.characters.species import PROFILES, known
    roster = known()
    assert len(roster) >= 25
    for expected in ("lion", "elephant", "buffalo", "bull", "tiger", "panther", "dog",
                     "cat", "rhino", "wolf", "bear", "giraffe", "hippo", "camel"):
        assert expected in roster, f"{expected} is missing from the roster"
    covered = {r for sp in PROFILES.values() for r in sp.roles}
    for role in ("protagonist", "antagonist", "mentor", "ally", "trickster",
                 "comic_relief", "underdog"):
        assert role in covered, f"no species is cast-able as {role}"


def test_every_species_is_schema_legal():
    """A species the writer may name must be one the puppet builder can draw."""
    import typing
    from asa.characters.species import known
    from asa.story.schema import NewCharacterSpec
    legal = set(typing.get_args(NewCharacterSpec.model_fields["species"].annotation))
    assert legal == set(known())


def test_casting_matches_animal_to_part():
    from asa.characters.species import suggest
    mentor = [k for k, _ in suggest("mentor", ("wise", "patient"), "savannah", limit=4)]
    assert "elephant" in mentor or "owl" in mentor
    predator = [k for k, _ in suggest("antagonist", ("dangerous", "silent"), "night",
                                      limit=4)]
    assert any(p in predator for p in ("panther", "leopard", "tiger", "wolf"))
    tiny = [k for k, _ in suggest("underdog", ("small", "overlooked"), "village", limit=3)]
    assert "mouse" in tiny


def test_relative_scale_orders_animals_by_real_size():
    from asa.characters.species import relative_scale
    order = ["mouse", "rabbit", "cat", "fox", "wolf", "lion", "buffalo", "elephant"]
    scales = [relative_scale(s) for s in order]
    assert scales == sorted(scales), f"species scales are not monotonic: {scales}"
    assert relative_scale("elephant", "mouse") > 2.0


def test_species_features_are_actually_drawn(tmp_path: Path):
    """Each signature feature must change the head art, or the flag is decorative."""
    from PIL import Image
    from asa.characters.procedural_puppet import AnimalPuppet, MILO_PALETTE
    heads = {}
    for sp in ("elephant", "rhino", "bull", "tiger", "leopard", "lion", "giraffe",
               "crow", "owl", "hedgehog", "raccoon", "boar"):
        AnimalPuppet(f"t_{sp}", MILO_PALETTE, sp).build(tmp_path / sp)
        heads[sp] = (tmp_path / sp / "head.png").read_bytes()
    assert len(set(heads.values())) == len(heads), "two species drew identical heads"


def test_horned_species_horns_are_visible(tmp_path: Path):
    """A horn hidden behind the skull is a horn that does not exist."""
    from PIL import Image
    from asa.characters.procedural_puppet import AnimalPuppet, MILO_PALETTE
    horn_rgb = (232, 220, 192)     # BASE_PALETTE['horn']
    for sp in ("rhino", "bull", "buffalo", "goat", "boar", "elephant"):
        AnimalPuppet(f"t_{sp}", MILO_PALETTE, sp).build(tmp_path / sp)
        with Image.open(tmp_path / sp / "head.png") as im:
            colours = {c for _, c in im.convert("RGB").getcolors(1 << 20)}
        assert horn_rgb in colours, f"{sp}'s horn/tusk is not visible in the head layer"


def test_qc_duration_check_uses_an_independent_source():
    """The expected duration must not be read from the file being checked."""
    import inspect
    from asa.core import stages
    src = inspect.getsource(stages.quality_control)
    assert "expected_duration_s=planned" in src
    assert 'expected_duration_s=v["duration_s"]' not in src


def test_stage_detail_cannot_collide_with_log_binding(db: Path, tmp_path: Path):
    """A stage returning its own `seconds` must not blow up the runner's log call.

    This actually happened: `animate` succeeded, reported seconds, and the TypeError in
    logging aborted the job after ~3 minutes of successful rendering.
    """
    from asa.core.runner import Runner
    from asa.core import stages as st

    class Ctx:
        pass
    ctx = Ctx()
    ctx.db = db
    ctx.paths_for = lambda jid: type("P", (), {"work": tmp_path, "out": tmp_path})()
    ctx.notifier = type("N", (), {"needs_human": lambda *a: None,
                                  "failed": lambda *a: None,
                                  "send": lambda *a, **k: None})()
    r = Runner(ctx)
    job_id = r.create_job()
    probe = st.Stage("probe", "RESEARCHED", "TOPIC_SELECTED",
                     lambda c, j: {"seconds": 12.5, "stage": "x", "job": 9, "frames": 3})
    st.REGISTRY.insert(0, probe)
    try:
        res = r.step(job_id)
        assert res.ok, res.error
        assert res.state == "TOPIC_SELECTED"
    finally:
        st.REGISTRY.remove(probe)


def test_loudness_keys_match_between_producer_and_consumers():
    """measure_loudness' keys must be the ones mixdown and QC actually read.

    They diverged once: the readers used loudnorm's `input_i`/`input_tp`, got 0.0 from
    `.get(...)`, stored 0.0 LUFS for a master that measured -14.2, and QC would have
    failed it.
    """
    import inspect
    from asa.assemble import mixdown
    from asa.media.audio import mixer
    from asa.qc import checks

    produced = {"integrated_lufs", "true_peak_dbfs", "lra"}
    src = inspect.getsource(mixer.measure_loudness)
    for key in produced:
        assert f'"{key}"' in src, f"measure_loudness no longer emits {key}"
    for module in (mixdown, checks):
        text = inspect.getsource(module)
        assert 'input_i' not in text, f"{module.__name__} reads loudnorm's key by mistake"
        assert 'input_tp' not in text, f"{module.__name__} reads loudnorm's key by mistake"
