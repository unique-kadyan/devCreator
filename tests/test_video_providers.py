"""The shot-video seam: conforming a hosted clip, and deciding who renders what.

Two things are worth guarding here, and they are the two that fail silently.

The frame count is the first. Scene clips are stream-copy concatenated and the mixdown is
laid against a scene duration measured from the audio, so a clip that comes back a little
long is not a slightly long clip - it is every later scene drifting out of sync, with every
individual file still perfectly valid.

Which provider pays is the second. Hosted video is billed per second of output, so a
provider that quietly accepts every shot - including the two-second insert of a letterbox
that has no mouth in it - is a bill rather than a bug report.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.core.errors import (                                     # noqa: E402
    AllProvidersExhausted, ProviderError, QuotaExhausted)
from asa.media.video.base import RenderedShot, ShotJob             # noqa: E402
from asa.media.video.conform import RETIME_TOLERANCE, conform_clip, probe  # noqa: E402
from asa.media.video.factory import VideoChain                     # noqa: E402
from asa.media.video.replicate import ReplicateVideo, _voice_slice  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH")


def make_clip(path: Path, seconds: float, fps: int, size=(320, 180)) -> Path:
    """A clip deliberately unlike what we ask for: wrong size, wrong rate, wrong length."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=size={size[0]}x{size[1]}:rate={fps}:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def job(**kw) -> ShotJob:
    base = dict(image_path=Path("/tmp/x.png"), frames=48, fps=24, size=(640, 360))
    base.update(kw)
    return ShotJob(**base)


# ---------------------------------------------------------------- conforming

@needs_ffmpeg
def test_a_clip_is_conformed_to_the_exact_frame_count(tmp_path):
    src = make_clip(tmp_path / "src.mp4", 5.0, 25, (320, 180))
    conform_clip(src, tmp_path / "out.mp4", frames=48, fps=24, size=(640, 360),
                 preserve_timing=True)
    got = probe(tmp_path / "out.mp4")
    assert got["packets"] == 48
    assert (got["width"], got["height"]) == (640, 360)


@needs_ffmpeg
def test_a_short_clip_is_padded_rather_than_left_short(tmp_path):
    """Ending short is the dangerous direction: the concat succeeds and the audio slides."""
    src = make_clip(tmp_path / "src.mp4", 0.5, 24)
    conform_clip(src, tmp_path / "out.mp4", frames=48, fps=24, size=(320, 180),
                 preserve_timing=True)
    assert probe(tmp_path / "out.mp4")["packets"] == 48


@needs_ffmpeg
def test_lip_synced_output_is_never_retimed(tmp_path):
    """Stretching a lip-synced clip slides the mouth off the voice the model was paid to
    match, so a mismatch is trimmed or padded instead."""
    src = make_clip(tmp_path / "src.mp4", 5.0, 24)
    stats = conform_clip(src, tmp_path / "out.mp4", frames=24, fps=24, size=(320, 180),
                         preserve_timing=True)
    assert stats["retimed"] is False


@needs_ffmpeg
def test_image_to_video_output_is_retimed_to_fit_the_edit(tmp_path):
    """Nothing in an i2v clip is synchronised to anything, so a model that only emits
    five-second clips must not get to dictate how long the shot is."""
    src = make_clip(tmp_path / "src.mp4", 5.0, 24)
    stats = conform_clip(src, tmp_path / "out.mp4", frames=48, fps=24, size=(320, 180),
                         preserve_timing=False)
    assert stats["retimed"] is True
    assert probe(tmp_path / "out.mp4")["packets"] == 48


@needs_ffmpeg
def test_a_near_exact_clip_is_not_retimed(tmp_path):
    """Padding a rounding difference is right; resampling every frame for it is not."""
    src = make_clip(tmp_path / "src.mp4", 2.0, 24)
    stats = conform_clip(src, tmp_path / "out.mp4",
                         frames=int(round(2.0 * 24 * (1 + RETIME_TOLERANCE / 2))),
                         fps=24, size=(320, 180), preserve_timing=False)
    assert stats["retimed"] is False


# ---------------------------------------------------------------- who pays

def test_a_hosted_provider_without_a_token_is_unavailable():
    assert ReplicateVideo(token="").available is False


def test_a_hosted_provider_only_pays_for_shots_with_a_mouth_in_them():
    p = ReplicateVideo(token="k", speaking_only=True)
    talking = job(voice_path=Path("/tmp/v.wav"), face=(0.1, 0.0, 0.8, 1.0),
                  envelope=[0.5] * 48)
    assert p.accepts(talking) is True
    # An establishing wide: no face, no envelope, nothing to sync.
    assert p.accepts(job()) is False
    # A face but no voice - a reaction shot while somebody else talks.
    assert p.accepts(job(face=(0.1, 0.0, 0.8, 1.0))) is False


def test_speaking_only_can_be_turned_off():
    assert ReplicateVideo(token="k", speaking_only=False).accepts(job()) is True


def test_the_shot_budget_is_a_hard_stop():
    """A retry loop against a metered API is a bill, not a crash."""
    p = ReplicateVideo(token="k", speaking_only=False, max_shots=2)
    assert p.accepts(job()) and p.accepts(job())
    p.rendered = 2
    assert p.accepts(job()) is False


# ---------------------------------------------------------------- the chain

class Stub:
    def __init__(self, name, *, available=True, accepts=True, boom=False):
        self.name = name
        self.available = available
        self._accepts = accepts
        self.boom = boom
        self.calls = 0

    def accepts(self, job):
        return self._accepts

    def render(self, job, dest):
        self.calls += 1
        if self.boom:
            raise ProviderError("nope", provider=self.name)
        return RenderedShot(path=dest, provider=self.name)


def test_a_declined_shot_falls_through_to_the_next_provider():
    """Declining is routine - it is how an episode comes out part hosted, part local."""
    hosted, local = Stub("hosted", accepts=False), Stub("local")
    assert VideoChain([hosted, local]).render(job(), Path("/tmp/o.mp4")).provider == "local"
    assert hosted.calls == 0


def test_a_failing_provider_falls_through_too():
    hosted, local = Stub("hosted", boom=True), Stub("local")
    assert VideoChain([hosted, local]).render(job(), Path("/tmp/o.mp4")).provider == "local"
    assert hosted.calls == 1


def test_an_unavailable_provider_is_skipped_without_being_called():
    hosted, local = Stub("hosted", available=False), Stub("local")
    VideoChain([hosted, local]).render(job(), Path("/tmp/o.mp4"))
    assert hosted.calls == 0


def test_an_empty_chain_raises_rather_than_returning_nothing():
    with pytest.raises(AllProvidersExhausted):
        VideoChain([Stub("a", accepts=False)]).render(job(), Path("/tmp/o.mp4"))


# ------------------------------------------------- what we actually send the model
#
# Every case below was found by preflighting the live Replicate schemas rather than by
# reading the code, which is the point: these are the inputs a model silently rejects or
# quietly misinterprets, and each one would have failed or degraded every hosted shot.


def test_a_fixed_length_model_is_asked_for_a_length_it_offers():
    """Kling accepts a `duration` of 5 or 10 and rejects everything else, so a 4.5-second
    shot asking for 4 fails the prediction outright."""
    p = ReplicateVideo(token="k", i2v_durations=[5, 10])
    offered = p.i2v_specs[0].durations
    assert p._clip_duration(job(frames=108, fps=24), offered) == 5      # 4.5s
    assert p._clip_duration(job(frames=24, fps=24), offered) == 5       # 1.0s
    assert p._clip_duration(job(frames=147, fps=24), offered) == 10     # 6.1s


def test_a_fixed_length_model_is_never_asked_for_a_clip_shorter_than_the_shot():
    """Rounding down looks tidier and freezes the end of every shot: the missing frames get
    filled by cloning the last one."""
    p = ReplicateVideo(token="k", i2v_durations=[5, 10])
    for frames in range(24, 24 * 11, 7):
        asked = p._clip_duration(job(frames=frames, fps=24), p.i2v_specs[0].durations)
        assert asked >= min(frames / 24, 10)


def test_a_free_form_model_is_asked_for_the_shots_own_length():
    p = ReplicateVideo(token="k", i2v_durations=[])
    assert p._clip_duration(job(frames=108, fps=24), p.i2v_specs[0].durations) == 4


def test_lipsync_and_i2v_extras_do_not_leak_into_each_other(tmp_path):
    """Replicate validates against the chosen model's schema and rejects unknown fields, so
    one shared bag of extras posts SadTalker's `preprocess` to Kling and fails the run."""
    img = tmp_path / "s.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    p = ReplicateVideo(token="k", speaking_only=False,
                       lipsync_extra={"preprocess": "full"},
                       i2v_extra={"cfg_scale": 0.7})
    _, i2v, _ = p._payload(job(image_path=img), tmp_path)
    assert "preprocess" not in i2v and i2v["cfg_scale"] == 0.7


@needs_ffmpeg
def test_a_shot_is_lip_synced_to_its_own_slice_of_the_speech(tmp_path):
    """A long speech is covered by several setups. Sending each of them the whole line and
    trimming the video afterwards syncs every part to the line's opening words - the mouth
    moves convincingly and says the wrong thing.
    """
    wav = tmp_path / "line.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=9", str(wav)], check=True)
    j = job(voice_path=wav, voice_offset_s=3.0, voice_duration_s=3.0,
            face=(0.1, 0.0, 0.8, 1.0), envelope=[0.4] * 48, seed=11)
    out = _voice_slice(j, tmp_path)
    assert out != wav, "the shot's own window must be cut out, not the whole line"
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip())
    assert abs(dur - 3.0) < 0.15


@needs_ffmpeg
def test_a_job_with_no_window_uses_the_whole_line(tmp_path):
    """What a caller that predates windows looks like: the whole line is right for it."""
    wav = tmp_path / "line.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", str(wav)], check=True)
    assert _voice_slice(job(voice_path=wav), tmp_path) == wav


def test_a_throttled_creation_is_retried_rather_than_downgraded(monkeypatch):
    """429 is the one failure that must NOT fall through to the local renderer.

    Replicate throttles prediction creation to a handful a minute. Treating that like any
    other provider failure loses the lip-sync the shot was routed here to get, for a reason
    that clears itself in seconds - and it does so silently, because falling through to
    local is a successful render.
    """
    import asa.media.video.replicate as R

    p = ReplicateVideo(token="k")
    calls = {"n": 0}

    class Resp:
        def __init__(self, code):
            self.status_code, self.headers, self.text = code, {}, "throttled"

        def json(self):
            return {"id": "ok"}

    class Client:
        def post(self, url, **kw):
            calls["n"] += 1
            return Resp(429 if calls["n"] == 1 else 201)

    monkeypatch.setattr(R, "CREATE_RETRY",
                        R.RetryPolicy(attempts=3, base_delay_s=0, max_delay_s=0, jitter=0))
    assert p._create(Client(), "owner/official", {}) == {"id": "ok"}
    assert calls["n"] == 2, "the throttled attempt must be retried, not abandoned"


def test_a_community_model_is_run_by_version_id():
    """`GET /models/{slug}` answering 200 does not mean the slug can be RUN: Replicate
    serves the model route for official models only and 404s it for community ones. That
    gap is what made a green preflight sit above a shot that failed every time."""
    p = ReplicateVideo(token="k")
    p._versions["cjwbw/sadtalker"] = "vvv"
    seen = []

    class Resp:
        def __init__(self, code):
            self.status_code, self.headers, self.text = code, {}, ""

        def json(self):
            return {"id": "ok"}

    class Client:
        def post(self, url, **kw):
            seen.append((url, kw.get("json", {})))
            return Resp(404 if "/models/" in url else 201)

    assert p._create(Client(), "cjwbw/sadtalker", {"a": 1}) == {"id": "ok"}
    assert seen[-1][0].endswith("/predictions")
    assert seen[-1][1]["version"] == "vvv"


def test_endless_throttling_is_reported_as_a_billing_problem(monkeypatch):
    """The diagnosis that cost four minutes of backoff to learn the first time.

    An account with no credit is throttled to a trickle and only returns the honest 402 once
    a request gets under the rate limit - so the retries, which are the thing generating the
    traffic, are the least likely to ever see it. All-429 therefore means "probably out of
    credit", and the error should say that rather than "rate limited".
    """
    import asa.media.video.replicate as R

    class Resp:
        status_code, headers, text = 429, {}, "throttled"

        def json(self):
            return {}

    class Client:
        def post(self, url, **kw):
            return Resp()

    monkeypatch.setattr(R, "CREATE_RETRY",
                        R.RetryPolicy(attempts=2, base_delay_s=0, max_delay_s=0, jitter=0))
    with pytest.raises(QuotaExhausted) as e:
        ReplicateVideo(token="k")._create(Client(), "owner/official", {})
    assert "out of credit" in str(e.value)


def test_running_out_of_credit_does_not_retry():
    """402 is not transient. Retrying it just spends the render's wall clock."""
    class Resp:
        status_code, headers, text = 402, {}, "insufficient credit"

    with pytest.raises(QuotaExhausted):
        ReplicateVideo(token="k")._raise_for(Resp(), "owner/model")


def test_an_account_that_cannot_pay_is_asked_only_once(tmp_path):
    """The forty minutes this exists to stop spending.

    Learning that the account has no credit costs a full CREATE_RETRY cycle - five attempts
    and about three and a half minutes of measured backoff - because an uncredited account
    is throttled to a trickle and answers 429 to the very retries trying to reach the honest
    402. That is a fair price once. It is not a fair price twelve times, which is what a
    twelve-speaking-shot episode paid before this, to be told the same thing each time and
    fall back to the local renderer anyway.

    The chain is built once per animate stage and this instance is reused across every
    scene, so the flag lives exactly as long as it should.
    """
    p = ReplicateVideo(token="k", speaking_only=False)
    asked = []

    def boom(*a, **kw):
        asked.append(1)
        raise QuotaExhausted("Replicate billing exhausted", provider="replicate")

    p._render = boom
    assert p.accepts(job()) is True
    with pytest.raises(QuotaExhausted):
        p.render(job(), tmp_path / "o.mp4")

    # Every later shot is declined outright rather than buying the same diagnosis again.
    assert p.accepts(job()) is False
    assert p.accepts(job()) is False
    assert len(asked) == 1


def test_a_chain_falls_through_quietly_once_the_account_is_known_to_be_broke(tmp_path):
    """And the fallback still renders. Declining is not failing."""
    hosted = ReplicateVideo(token="k", speaking_only=False)
    hosted._render = lambda *a, **kw: (_ for _ in ()).throw(
        QuotaExhausted("Replicate billing exhausted", provider="replicate"))
    local = Stub("local")
    chain = VideoChain([hosted, local])
    for _ in range(3):
        assert chain.render(job(), tmp_path / "o.mp4").provider == "local"
    assert local.calls == 3


def test_a_prompt_taking_lipsync_model_is_given_the_scenes_action(tmp_path):
    """Audio-driven models split on the prompt, and both halves fail loudly.

    Wan-S2V requires one and rejects the prediction without it; OmniHuman's schema is
    exactly image and audio, and Replicate rejects unknown fields, so a prompt sent
    unconditionally would fail every OmniHuman shot instead. Keying it off the mapping is
    what keeps switching between them a config edit.
    """
    # Bytes, not media: `_data_uri` reads the file and guesses the mime from the suffix,
    # and a job naming no voice window is handed its wav untouched, so nothing here decodes.
    wav, img = tmp_path / "v.wav", tmp_path / "s.png"
    wav.write_bytes(b"RIFF----WAVE")
    img.write_bytes(b"\x89PNG\r\n----")
    spoken = dict(image_path=img, voice_path=wav, envelope=[0.4] * 48,
                  face=(0.1, 0.0, 0.8, 1.0), motion_prompt="a fox pushes a laptop away")

    wan = ReplicateVideo(token="k", lipsync_model="wan-video/wan-2.2-s2v",
                         lipsync_inputs={"image": "image", "audio": "audio",
                                         "prompt": "prompt"})
    _, payload, _ = wan._payload(job(**spoken), tmp_path)
    assert payload["prompt"] == "a fox pushes a laptop away"

    # The default mapping declares no prompt, so none is sent.
    _, payload, _ = ReplicateVideo(token="k")._payload(job(**spoken), tmp_path)
    assert set(payload) == {"image", "audio"}


def test_a_prompt_taking_lipsync_model_still_gets_one_on_a_wordless_shot(tmp_path):
    """A required field cannot be left to the scene having written an action line."""
    wav, img = tmp_path / "v.wav", tmp_path / "s.png"
    wav.write_bytes(b"RIFF----WAVE")
    img.write_bytes(b"\x89PNG\r\n----")
    p = ReplicateVideo(token="k", lipsync_inputs={"image": "image", "audio": "audio",
                                                  "prompt": "prompt"})
    _, payload, _ = p._payload(
        job(image_path=img, voice_path=wav, envelope=[0.4] * 48,
            face=(0.1, 0.0, 0.8, 1.0), motion_prompt=""), tmp_path)
    assert payload["prompt"]


# ------------------------------------------------- falling back between MODELS
#
# A different axis from the provider chain. Dropping to the local renderer when a hosted
# model errors gives up lip-sync and gesture - the two things the hosted call was made for -
# over something as ordinary as a slug renamed overnight.


def spoken_job(tmp_path, **kw):
    """A speaking shot. Bytes, not media: `_data_uri` reads the file and guesses the mime
    from the suffix, and a job naming no voice window is handed its wav untouched."""
    wav, img = tmp_path / "v.wav", tmp_path / "s.png"
    wav.write_bytes(b"RIFF----WAVE")
    img.write_bytes(b"\x89PNG\r\n----")
    return job(image_path=img, voice_path=wav, envelope=[0.4] * 48,
               face=(0.1, 0.0, 0.8, 1.0), **kw)


def wire(p, fails: set, seen: list, tmp_path):
    """Drive the model walk without a network: record what each model was sent, and let the
    named ones fail the way a renamed slug or a choked prediction does."""
    def create(client, model, payload):
        seen.append((model, sorted(payload)))
        if model in fails:
            raise ProviderError(f"{model} exploded", provider="replicate")
        return {"id": "ok"}

    p._create = create
    p._await = lambda client, pred, model: "https://example.invalid/clip.mp4"
    p._fetch = lambda client, url, raw: make_clip(raw, 2.0, 24)
    return p


@needs_ffmpeg
def test_a_failed_model_falls_back_to_the_next_one_with_its_OWN_mapping(tmp_path):
    """The reason a fallback list of bare slugs would be a trap.

    Field names are a property of the model: OmniHuman takes `image`/`audio`, SadTalker calls
    the same two files `source_image`/`driven_audio`. A second model inheriting the first's
    mapping would fail every prediction it was added to rescue - and it would do so on the
    day the primary was already broken, which is the worst day to discover it.
    """
    p = wire(ReplicateVideo(token="k", lipsync_fallbacks=[
        {"model": "backup/talking-head",
         "inputs": {"image": "source_image", "audio": "driven_audio"},
         "extra": {"preprocess": "full"}}]),
        fails={"bytedance/omni-human"}, seen=(seen := []), tmp_path=tmp_path)

    out = p.render(spoken_job(tmp_path), tmp_path / "o.mp4")
    assert out.model_id == "backup/talking-head"
    assert seen == [("bytedance/omni-human", ["audio", "image"]),
                    ("backup/talking-head", ["driven_audio", "preprocess", "source_image"])]
    assert out.stats["fell_back_from"]          # and it says so, for the log


@needs_ffmpeg
def test_the_primary_is_not_charged_for_a_fallback_that_was_never_needed(tmp_path):
    p = wire(ReplicateVideo(token="k", lipsync_fallbacks=[
        {"model": "backup/x", "inputs": {"image": "image", "audio": "audio"}}]),
        fails=set(), seen=(seen := []), tmp_path=tmp_path)
    out = p.render(spoken_job(tmp_path), tmp_path / "o.mp4")
    assert out.model_id == "bytedance/omni-human"
    assert len(seen) == 1
    assert "fell_back_from" not in out.stats


def test_no_credit_does_not_walk_the_model_list(tmp_path):
    """Every slug on the account fails a 402 identically, so trying the rest would spend the
    shot's wall clock to be told the same thing three times."""
    p = ReplicateVideo(token="k", lipsync_fallbacks=[
        {"model": "backup/x", "inputs": {"image": "image", "audio": "audio"}}])
    seen = []

    def create(client, model, payload):
        seen.append(model)
        raise QuotaExhausted("Replicate billing exhausted", provider="replicate")

    p._create = create
    with pytest.raises(QuotaExhausted):
        p.render(spoken_job(tmp_path), tmp_path / "o.mp4")
    assert seen == ["bytedance/omni-human"]
    assert p.accepts(spoken_job(tmp_path)) is False       # and the latch still fires


def test_every_model_failing_names_every_model(tmp_path):
    """The chain then drops to the local renderer, and the log has to say why - one error
    naming the last model would hide that two others were tried and how they failed."""
    p = wire(ReplicateVideo(token="k", lipsync_fallbacks=[
        {"model": "backup/x", "inputs": {"image": "image", "audio": "audio"}}]),
        fails={"bytedance/omni-human", "backup/x"}, seen=[], tmp_path=tmp_path)
    with pytest.raises(ProviderError) as e:
        p.render(spoken_job(tmp_path), tmp_path / "o.mp4")
    assert "bytedance/omni-human" in str(e.value) and "backup/x" in str(e.value)


def test_a_fallback_missing_its_mapping_is_dropped_not_obeyed():
    """Skipped rather than raised - a mistyped fallback must not take down an episode whose
    primary works - but it is never silently given the primary's field names."""
    p = ReplicateVideo(token="k", lipsync_fallbacks=[
        {"model": "backup/x"},                       # no inputs at all
        {"inputs": {"image": "image"}},              # no model
        {"model": "good/x", "inputs": {"image": "image", "audio": "audio"}}])
    assert [s.model for s in p.lipsync_specs] == ["bytedance/omni-human", "good/x"]


def test_clip_length_is_asked_per_model_not_per_provider():
    """Kling offers 5 or 10 and rejects everything else; Wan has no duration field at all,
    so the enum belongs to the model. A fallback inheriting the primary's would fail on the
    very shot it was added to rescue."""
    p = ReplicateVideo(token="k", i2v_durations=[5, 10], i2v_fallbacks=[
        {"model": "free/form", "inputs": {"image": "image", "prompt": "prompt"}}])
    primary, fallback = p.i2v_specs
    assert primary.durations == (5, 10) and fallback.durations == ()
    shot = job(frames=108, fps=24)                                  # 4.5s
    assert p._clip_duration(shot, primary.durations) == 5
    assert p._clip_duration(shot, fallback.durations) == 4


def test_the_default_lipsync_model_takes_a_whole_figure_not_just_a_head():
    """The gap this provider was pointed at second: SadTalker moves a head and freezes the
    body, so a character explaining something for forty seconds never moves an arm. The
    default is an audio-driven whole-figure model, and its input mapping must be the plain
    two fields that model actually declares - Replicate rejects unknown fields, so a
    leftover `preprocess` from the old model fails the prediction outright."""
    p = ReplicateVideo(token="k")
    assert p.lipsync_model == "bytedance/omni-human"
    assert set(p.lipsync_inputs) == {"image", "audio"}
    assert p.lipsync_extra == {}
