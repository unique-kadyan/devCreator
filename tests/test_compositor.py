"""Golden-frame and geometry tests. No network, no API keys."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.characters.procedural_puppet import FoxPuppet, MILO_PALETTE   # noqa: E402
from asa.characters.rig import Rig, VISEMES, EYE_STATES                # noqa: E402
from asa.media.animation.camera import Camera, ease                    # noqa: E402
from asa.media.animation.compositor import (                           # noqa: E402
    BackgroundLayer, CharacterInstance, SceneRenderer, SpeechSpan)
from asa.media.animation.parallax import multiplane                    # noqa: E402

WORLD = (960, 540)
FRAME = (640, 360)


@pytest.fixture(scope="module")
def puppet(tmp_path_factory) -> tuple[Rig, Path]:
    d = tmp_path_factory.mktemp("milo")
    rig = FoxPuppet("milo_fox", MILO_PALETTE).build(d)
    return rig, d


def test_rig_has_every_viseme_and_eye_state(puppet):
    rig, _ = puppet
    assert set(rig.visemes) == set(VISEMES)
    assert set(rig.eyes) == set(EYE_STATES)
    for key in list(rig.visemes.values()) + list(rig.eyes.values()):
        assert key in rig.layers


def test_rig_roundtrips(puppet, tmp_path):
    rig, _ = puppet
    p = tmp_path / "rig.json"
    rig.save(p)
    assert Rig.load(p).model_dump() == rig.model_dump()


def test_ease_is_monotonic_and_bounded():
    for kind in ("linear", "in", "out", "in_out"):
        vals = [ease(i / 20, kind) for i in range(21)]
        assert vals[0] == pytest.approx(0.0, abs=1e-6)
        assert vals[-1] == pytest.approx(1.0, abs=1e-6)
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_camera_viewport_never_leaves_the_world():
    for move in ("static", "push_in", "pull_out", "pan_left", "pan_right",
                 "tilt_up", "tilt_down", "handheld_drift"):
        cam = Camera(WORLD, FRAME, move=move, shot_from="wide", focus=(0.1, 0.9))
        for i in range(11):
            x0, y0, x1, y1 = cam.at(i / 10).box()
            assert x0 >= -0.5 and y0 >= -0.5
            assert x1 <= WORLD[0] + 0.5 and y1 <= WORLD[1] + 0.5


def test_multiplane_returns_three_ordered_planes():
    plate = Image.new("RGB", (480, 270), (120, 90, 60))
    layers = multiplane(plate, WORLD)
    assert [round(l.parallax, 2) for l in layers] == [0.30, 0.62, 1.00]
    assert all(l.image.size == WORLD for l in layers)


def _renderer(rig, base, duration=2.0, **kw):
    plate = Image.new("RGB", (480, 270), (120, 90, 60))
    ch = CharacterInstance(rig=rig, base_dir=base, x=0.5, y=0.93, scale=1.0,
                           speech=[SpeechSpan(0.2, 1.5)], blink_seed=7, **kw)
    return SceneRenderer(WORLD, FRAME, multiplane(plate, WORLD), [ch],
                         Camera(WORLD, FRAME, move="push_in", shot_from="wide"),
                         duration, fps=12, final=False)


def test_frames_are_correct_size_and_opaque(puppet):
    rig, base = puppet
    r = _renderer(rig, base)
    f = r.render_frame(0)
    assert f.size == FRAME
    assert f.getchannel("A").getextrema() == (255, 255)


def test_render_is_deterministic(puppet):
    """Same inputs must give byte-identical frames, or resume-after-crash reveals seams."""
    rig, base = puppet
    a = _renderer(rig, base).render_frame(9).tobytes()
    b = _renderer(rig, base).render_frame(9).tobytes()
    assert a == b


def test_character_actually_changes_the_frame(puppet):
    """Guards against a silently mis-transformed puppet landing off-screen."""
    rig, base = puppet
    plate = Image.new("RGB", (480, 270), (120, 90, 60))
    bg = multiplane(plate, WORLD)
    cam = Camera(WORLD, FRAME, move="static", shot_from="wide")
    empty = SceneRenderer(WORLD, FRAME, bg, [], cam, 2.0, 12, final=False).render_frame(0)
    withch = _renderer(rig, base).render_frame(0)
    assert empty.tobytes() != withch.tobytes()


def test_visemes_move_the_mouth(puppet):
    rig, base = puppet
    ch = CharacterInstance(rig=rig, base_dir=base, speech=[SpeechSpan(0.0, 5.0)], blink_seed=3)
    seen = {ch.viseme(i / 12) for i in range(60)}
    assert len(seen) > 2 and "rest" not in {ch.viseme(0.5)}
    silent = CharacterInstance(rig=rig, base_dir=base, speech=[], blink_seed=3)
    assert {silent.viseme(i / 12) for i in range(24)} == {"rest"}


def test_blinks_happen_and_are_deterministic(puppet):
    rig, base = puppet
    ch = CharacterInstance(rig=rig, base_dir=base, blink_seed=11)
    states = [ch.eye_state(i / 24) for i in range(24 * 20)]
    assert "closed" in states and "open" in states
    again = CharacterInstance(rig=rig, base_dir=base, blink_seed=11)
    assert [again.eye_state(i / 24) for i in range(24 * 20)] == states


# ---------------------------------------------------------------- phase 2: audio

def test_envelope_shape_and_bounds(tmp_path):
    """A gated, smoothed envelope must stay in [0,1] and be silent on silence."""
    import numpy as np
    import soundfile as sf
    from asa.media.audio.envelope import envelope_to_viseme, rms_envelope

    sr = 24000
    sig = np.concatenate([
        np.zeros(sr // 2, dtype=np.float32),
        (0.6 * np.sin(2 * np.pi * 180 * np.arange(sr) / sr)).astype(np.float32),
        np.zeros(sr // 2, dtype=np.float32),
    ])
    p = tmp_path / "s.wav"
    sf.write(p, sig, sr)
    env = rms_envelope(p, fps=24)
    assert len(env) == pytest.approx(48, abs=2)
    assert all(0.0 <= v <= 1.0 for v in env)
    assert max(env[:8]) < 0.2 and max(env[16:28]) > 0.7
    assert envelope_to_viseme(0.0) == "rest" and envelope_to_viseme(0.95) == "A"


def test_timeline_duration_comes_from_audio_not_estimates():
    from asa.media.audio.timeline import build_scene_timing
    from asa.media.tts.base import Utterance

    def u(d):
        return Utterance(text="x", path=Path("x.wav"), duration_s=d, sample_rate=24000,
                         character_id=None, voice_id="v", provider="t", text_sha256="h")

    t = build_scene_timing(1, [(u(2.0), None), (u(3.0), "milo_fox")],
                           gap_s=0.25, pre_pad=0.3, post_pad=0.5)
    assert t.duration_s == pytest.approx(0.3 + 2.0 + 0.25 + 3.0 + 0.5, abs=1e-6)
    assert t.cues[1].start_s == pytest.approx(0.3 + 2.0 + 0.25)
    assert t.speech_spans("milo_fox") == [(pytest.approx(2.55), pytest.approx(5.55))]
    assert len(t.speech_spans(None)) == 1


def test_utterance_hash_is_stable_and_parameter_sensitive():
    from asa.media.tts.base import Utterance
    a = Utterance.hash_text("hello there", "af_heart", 1.0, 0.0)
    assert a == Utterance.hash_text("hello there", "af_heart", 1.0, 0.0)
    assert a != Utterance.hash_text("hello there", "af_heart", 1.05, 0.0)
    assert a != Utterance.hash_text("hello there", "am_puck", 1.0, 0.0)
    assert a != Utterance.hash_text("hello  there!", "af_heart", 1.0, 0.0)


def test_envelope_drives_mouth_instead_of_random_cycle(puppet):
    rig, base = puppet
    loud = CharacterInstance(rig=rig, base_dir=base, envelope=[0.0] * 12 + [0.9] * 12,
                             envelope_fps=24, blink_seed=1)
    assert loud.viseme(0.1) == "rest"
    assert loud.viseme(0.7) == "A"
    assert loud.is_speaking(0.7) and not loud.is_speaking(0.1)


# ---------------------------------------------------------------- gap-closing pass

def test_scene_grading_changes_pixels_but_preserves_alpha(puppet):
    rig, base = puppet
    plain = CharacterInstance(rig=rig, base_dir=base)
    lit = CharacterInstance(rig=rig, base_dir=base, grade_rgb=(255, 190, 140),
                            grade_strength=0.4)
    a, b = plain.layer("head"), lit.layer("head")
    assert a.size == b.size
    assert a.tobytes() != b.tobytes()
    assert a.getchannel("A").tobytes() == b.getchannel("A").tobytes()


def test_zero_grade_is_a_no_op(puppet):
    rig, base = puppet
    a = CharacterInstance(rig=rig, base_dir=base).layer("body")
    b = CharacterInstance(rig=rig, base_dir=base, grade_rgb=(255, 100, 0),
                          grade_strength=0.0).layer("body")
    assert a.tobytes() == b.tobytes()


def test_light_from_plate_returns_a_usable_tint():
    from asa.media.animation.grade import light_from_plate
    warm = Image.new("RGB", (200, 200), (200, 140, 70))
    tint, strength = light_from_plate(warm)
    assert 0 < strength <= 1
    assert all(0 <= c <= 255 for c in tint)
    assert tint[0] > tint[2]              # warm plate stays warm
    cool = Image.new("RGB", (200, 200), (70, 120, 200))
    assert light_from_plate(cool)[0][2] > light_from_plate(cool)[0][0]


def test_ledger_blocks_unlicensed_and_noncommercial(tmp_path):
    import sqlite3
    from asa.core.ledger import add_asset, audit, attribution_block
    db = tmp_path / "t.db"
    sqlite3.connect(db).executescript(
        (ROOT / "migrations/001_initial.sql").read_text())
    good = tmp_path / "ok.wav"; good.write_bytes(b"RIFF0000WAVE")
    nc = tmp_path / "nc.wav"; nc.write_bytes(b"RIFF0000WAVE")
    add_asset(db, good, kind="music", source="yt_audio_library", license_code="YT-AUDIO-LIB")
    add_asset(db, nc, kind="music", source="somewhere", license_code="CC-BY-NC",
              attribution="x")
    problems = {Path(p["path"]).name: p["reason"] for p in audit(db)}
    assert "nc.wav" in problems and "noncommercial" in problems["nc.wav"]
    assert "ok.wav" not in problems


def test_ledger_refuses_ccby_without_attribution(tmp_path):
    import sqlite3
    from asa.core.ledger import add_asset
    db = tmp_path / "t.db"
    sqlite3.connect(db).executescript((ROOT / "migrations/001_initial.sql").read_text())
    f = tmp_path / "s.wav"; f.write_bytes(b"RIFF0000WAVE")
    with pytest.raises(ValueError, match="attribution"):
        add_asset(db, f, kind="sfx", source="freesound", license_code="CC-BY")
    add_asset(db, f, kind="sfx", source="freesound", license_code="CC-BY",
              attribution='"Leaves" by someone - freesound.org - CC BY 4.0')


def test_unregistered_finds_files_with_no_licence_row(tmp_path):
    import sqlite3
    from asa.core.ledger import unregistered
    db = tmp_path / "data" / "t.db"
    db.parent.mkdir()
    sqlite3.connect(db).executescript((ROOT / "migrations/001_initial.sql").read_text())
    music = tmp_path / "assets" / "music" / "warm"
    music.mkdir(parents=True)
    (music / "orphan.wav").write_bytes(b"RIFF0000WAVE")
    found = unregistered(db, [tmp_path / "assets" / "music"])
    assert any("orphan.wav" in f for f in found)


def test_error_taxonomy_distinguishes_retry_from_advance():
    from asa.core.errors import QuotaExhausted, RateLimited
    assert RateLimited("x", "p").retryable is True
    assert QuotaExhausted("x", "p").retryable is False


def test_sfx_rejects_unknown_licences():
    """A sound whose licence is not explicitly allow-listed must never be used."""
    from asa.media.audio.sfx import ALLOWED
    assert "Attribution Noncommercial" not in ALLOWED
    assert "Sampling+" not in ALLOWED
    assert set(ALLOWED.values()) <= {"CC0", "CC-BY"}


def test_cli_doctor_and_help_are_wired():
    from asa.core import cli
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert callable(cli.cmd_doctor) and callable(cli.cmd_bench)


def test_assets_scan_does_not_pass_on_an_empty_library(tmp_path, monkeypatch, capsys):
    """An empty library must FAIL, not report 'all clear' - that reads as a pass."""
    from asa.core import cli
    (tmp_path / "assets/music").mkdir(parents=True)
    (tmp_path / "assets/sfx").mkdir(parents=True)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.cmd_assets_scan(None) == 1
    assert "NO audio files" in capsys.readouterr().out
