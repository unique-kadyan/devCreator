"""The join between shot planning, the images on disk and the clip renderer."""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.assemble.shot_render import (                                  # noqa: E402
    _slice_envelope, build_shot_jobs, shot_image_name)
from asa.assemble.shotlist import image_keys, plan_shots                # noqa: E402
from asa.core.errors import RenderError                                 # noqa: E402
from asa.media.images.scene_image import (                              # noqa: E402
    CINEMATIC_STYLE, scene_prompt, shot_prompt)
from asa.media.video.factory import build_video_chain                   # noqa: E402


@dataclass
class FakeUtterance:
    duration_s: float
    path: Path = Path("/dev/null")


@dataclass
class FakeCue:
    start_s: float
    utterance: FakeUtterance
    character_id: str | None

    @property
    def end_s(self) -> float:
        return self.start_s + self.utterance.duration_s


@dataclass
class FakeTiming:
    cues: list
    duration_s: float

    def envelope_for(self, character_id, fps):
        n = int(round(self.duration_s * fps)) + 1
        env = [0.0] * n
        for c in self.cues:
            if c.character_id != character_id:
                continue
            for k in range(int(c.start_s * fps), min(n, int(c.end_s * fps))):
                env[k] = 0.8
        return env


SCENE = {"idx": 4, "shot": "medium", "emotion": "neutral", "camera_move": "static",
         "staging": {"a": {}, "b": {}}, "action": "they argue by the gate",
         "dialogue": [{"idx": 0, "character_id": "a", "emotion": "angry"},
                      {"idx": 1, "character_id": "b", "emotion": "happy"}]}
TIMING = FakeTiming([FakeCue(0.35, FakeUtterance(2.0), "a"),
                     FakeCue(2.65, FakeUtterance(2.0), "b")], 5.2)


def test_a_missing_image_names_the_stage_that_should_have_made_it(tmp_path):
    with pytest.raises(RenderError) as e:
        build_shot_jobs(SCENE, TIMING, tmp_path, fps=24, size=(320, 180))
    assert "art stage" in str(e.value)


def test_every_shot_gets_an_envelope_the_length_of_its_clip(tmp_path):
    for shot in plan_shots(SCENE):
        img = tmp_path / shot_image_name(SCENE["idx"], shot.image_key)
        Image.new("RGB", (64, 36)).save(img)
    jobs = build_shot_jobs(SCENE, TIMING, tmp_path, fps=24, size=(320, 180))
    assert jobs
    for shot, job in jobs:
        assert job.frames == shot.frames
        assert len(job.envelope) == shot.frames


def test_the_envelope_is_the_speakers_own_not_the_scenes():
    """An envelope built from the whole mix opens the mouth of whoever is on screen
    whenever ANYONE is talking, which is what makes a two-hander look dubbed."""
    shots = plan_shots(SCENE)
    whole_scene = replace(shots[0], start_frame=0, end_frame=int(5.2 * 24))
    env = _slice_envelope(TIMING, whole_scene, 24)
    speaking = [i for i, v in enumerate(env) if v > 0]
    # 'a' speaks 0.35-2.35s; 'b' speaks 2.65-4.65s and must leave 'a' silent.
    assert speaking, "the speaker's own line should be in the envelope"
    assert max(speaking) / 24 < 2.5
    assert all(env[i] == 0 for i in range(int(2.6 * 24), int(4.6 * 24)))


def test_narration_shots_carry_no_envelope_and_no_face(tmp_path):
    scene = dict(SCENE, dialogue=[{"idx": 0, "character_id": None, "emotion": "neutral"}])
    timing = FakeTiming([FakeCue(0.35, FakeUtterance(3.0), None)], 4.0)
    for shot in plan_shots(scene):
        Image.new("RGB", (64, 36)).save(tmp_path / shot_image_name(4, shot.image_key))
    (shot, job), = build_shot_jobs(scene, timing, tmp_path, fps=24, size=(320, 180))
    assert job.face is None and job.envelope == []


def test_shot_image_names_are_stable_and_distinct():
    assert shot_image_name(4, "a|close_up|warm") == shot_image_name(4, "a|close_up|warm")
    assert shot_image_name(4, "a|close_up|warm") != shot_image_name(4, "b|close_up|warm")
    assert shot_image_name(4, "a|close_up|warm") != shot_image_name(5, "a|close_up|warm")


def test_an_unimplemented_video_provider_falls_through_to_local():
    """A chain that can run out is a chain that fails a render at 3am."""
    class Cfg:
        def get(self, key, default=None):
            return ["kling", "local"] if key == "providers.video.chain" else default

    chain = build_video_chain(Cfg())
    assert [p.name for p in chain.providers] == ["local"]
    assert chain.providers[0].available


def test_the_art_stage_buys_exactly_the_images_the_animate_stage_asks_for(tmp_path):
    """The two stages run in separate processes - and after a crash, on separate days - and
    never share state. They agree only because both re-derive the shot list from the same
    dialogue. This is the test that fails first if that ever stops being true.

    It matters most for long speeches, which are planned as several DIFFERENT setups. An
    earlier version split them only at render time, when the art stage had already been and
    gone, so the extra setups named images nobody had bought.
    """
    long_line = " ".join(["shabd"] * 90)                       # ~39s: forces a split
    scene = {"idx": 7, "shot": "medium", "emotion": "neutral", "camera_move": "static",
             "staging": {"a": {}, "b": {}}, "action": "she explains the whole plan",
             "dialogue": [{"idx": 0, "character_id": None, "emotion": "neutral",
                           "line": long_line},
                          {"idx": 1, "character_id": "a", "emotion": "happy",
                           "line": long_line}]}
    timing = FakeTiming([FakeCue(0.0, FakeUtterance(39.0), None),
                         FakeCue(39.5, FakeUtterance(39.0), "a")], 79.0)

    # What the ART stage would buy, from the dialogue alone.
    bought = {shot_image_name(scene["idx"], k)
              for k in image_keys(plan_shots(scene, max_images=6))}
    for name in bought:
        Image.new("RGB", (64, 36)).save(tmp_path / name)

    # What the ANIMATE stage asks for, having measured the audio.
    jobs = build_shot_jobs(scene, timing, tmp_path, fps=24, size=(64, 36), max_images=6)
    assert jobs, "a scene with dialogue must produce shots"
    assert {j.image_path.name for _, j in jobs} <= bought
    assert len(jobs) > 2, "a 39-second speech must not be one shot"


def test_a_long_speech_is_covered_by_more_than_one_picture(tmp_path):
    """The 52%-of-the-episode-is-a-still defect, at the level that produces the files."""
    long_line = " ".join(["shabd"] * 90)
    scene = {"idx": 2, "shot": "medium", "emotion": "neutral", "camera_move": "static",
             "staging": {"a": {}, "b": {}}, "action": "he talks",
             "dialogue": [{"idx": 0, "character_id": "a", "emotion": "neutral",
                           "line": long_line}]}
    timing = FakeTiming([FakeCue(0.0, FakeUtterance(39.0), "a")], 39.0)
    for k in image_keys(plan_shots(scene, max_images=6)):
        Image.new("RGB", (64, 36)).save(tmp_path / shot_image_name(scene["idx"], k))
    jobs = build_shot_jobs(scene, timing, tmp_path, fps=24, size=(64, 36), max_images=6)
    assert len({j.image_path.name for _, j in jobs}) > 1
    # And no single picture may carry the bulk of it.
    longest = max(sum(j.frames for _, j in jobs if j.image_path.name == n)
                  for n in {j.image_path.name for _, j in jobs})
    assert longest / sum(j.frames for _, j in jobs) < 0.6


def test_every_framing_the_planner_can_emit_asks_for_a_different_picture():
    """Two framings that build the same prompt are one picture with two filenames.

    The image chain caches on the prompt, so identical prompts return identical pixels -
    and cutting between them changes nothing on screen while still costing a generation and
    still reporting, truthfully but uselessly, that the scene now has more setups.
    `close_up` and `extreme_close_up` shipped with byte-identical composition strings.
    """
    from asa.assemble.shotlist import _NARRATION_FRAMINGS, _SPEAKER_FRAMINGS

    scene = {"idx": 1, "shot": "medium", "emotion": "neutral", "camera_move": "static",
             "staging": {"a": {}, "b": {}}, "action": "they talk", "visual_prompt": "a room"}
    cast = {"a": {"character_id": "a", "name": "A", "species": "fox"}}

    speaking = {f: shot_prompt(scene, cast, speaker="a", framing=f, speaking=True)
                for f in _SPEAKER_FRAMINGS}
    assert len(set(speaking.values())) == len(speaking), \
        f"duplicate speaking prompts: {sorted(speaking)}"

    narration = {f: shot_prompt(scene, cast, speaker=None, framing=f)
                 for f in _NARRATION_FRAMINGS}
    assert len(set(narration.values())) == len(narration), \
        f"duplicate narration prompts: {sorted(narration)}"


def test_each_part_of_a_speech_gets_its_own_window_into_the_voice(tmp_path):
    """A hosted lip-sync model is driven by the wav, so the wav has to be the right stretch
    of it. Every part of a long speech carries the same line index; without a window they
    would all be synced to the line's opening words."""
    long_line = " ".join(["shabd"] * 90)
    scene = {"idx": 3, "shot": "medium", "emotion": "neutral", "camera_move": "static",
             "staging": {"a": {}, "b": {}}, "action": "he explains",
             "dialogue": [{"idx": 0, "character_id": "a", "emotion": "neutral",
                           "line": long_line}]}
    timing = FakeTiming([FakeCue(0.0, FakeUtterance(39.0), "a")], 39.0)
    for k in image_keys(plan_shots(scene, max_images=6)):
        Image.new("RGB", (64, 36)).save(tmp_path / shot_image_name(scene["idx"], k))
    jobs = build_shot_jobs(scene, timing, tmp_path, fps=24, size=(64, 36), max_images=6)

    offsets = [j.voice_offset_s for _, j in jobs]
    assert len(set(offsets)) == len(offsets), f"parts share a window: {offsets}"
    assert offsets == sorted(offsets), "windows must advance through the line"
    assert offsets[0] < 1.0, "the first part starts at the start of the line"
    for shot, j in jobs:
        assert j.voice_path is not None
        assert j.voice_duration_s > 0
        # The window may not run past the end of the recording.
        assert j.voice_offset_s + j.voice_duration_s <= 39.0 + 0.01


# ------------------------------------------------------------------ prompt hygiene
#
# `scenes.visual_prompt` is authored for the PUPPET renderer, which wants an empty
# background plate in a flat storybook style. Pasting it into a cinematic prompt puts two
# contradictory art directions in one string, and the image model obeys whichever it likes -
# which is why one scene came back part flat 2D illustration and part photoreal.

STYLED = {"idx": 1, "shot": "medium", "emotion": "neutral", "camera_move": "static",
          "staging": {"a": {}}, "action": "वीरू अपने कमरे में बैठा है, लैपटॉप के सामने",
          "visual_prompt": ("A small rented room at night, a desk with a glowing laptop, "
                            "storybook flat-vector illustration, clean bold shapes, "
                            "muted natural palette, painterly texture, no characters, "
                            "no people, no animals, no text, no watermark")}
CAST = {"a": {"character_id": "a", "name": "A", "species": "fox"}}


def test_the_puppet_paths_flat_style_never_reaches_a_photoreal_prompt():
    p = shot_prompt(STYLED, CAST, speaker="a", framing="close_up", speaking=True).lower()
    for banned in ("storybook", "flat-vector", "illustration", "painterly", "bold shapes"):
        assert banned not in p, f"{banned!r} contradicts {CINEMATIC_STYLE[:24]!r}"


def test_the_puppet_paths_no_characters_clause_never_reaches_a_character_shot():
    """It occasionally won the argument and returned an empty room."""
    p = shot_prompt(STYLED, CAST, speaker="a", framing="close_up", speaking=True).lower()
    assert "no characters" not in p and "no animals" not in p


def test_the_scene_description_itself_survives():
    """Stripping the styling must not strip the room."""
    p = shot_prompt(STYLED, CAST, speaker="a", framing="close_up", speaking=True).lower()
    assert "rented room" in p and "laptop" in p


def test_non_english_prose_is_not_sent_to_an_english_image_model():
    """A Hindi channel writes `action` in Hindi; FLUX has no useful response to Devanagari
    and it crowds out the parts of the prompt that do steer it."""
    p = shot_prompt(STYLED, CAST, speaker="a", framing="close_up", speaking=True)
    assert all(c.isascii() for c in p), [c for c in p if not c.isascii()][:8]


def test_english_action_is_still_used():
    scene = dict(STYLED, action="he stares at the empty inbox")
    p = shot_prompt(scene, CAST, speaker="a", framing="close_up", speaking=True)
    assert "empty inbox" in p


def test_the_same_hygiene_applies_to_whole_scene_prompts():
    """`scene_prompt` is the non-performance path and had the identical defect."""
    p = scene_prompt(STYLED, CAST).lower()
    assert "storybook" not in p and "no characters" not in p
    assert all(c.isascii() for c in p)
