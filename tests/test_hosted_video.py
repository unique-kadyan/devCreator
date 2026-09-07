"""Several metered services in one chain, and what happens as each runs out.

The point of more than one hosted provider is that free allowances are small and separate.
So the behaviour worth guarding is not "does it render" - the per-model tests next door
cover that - but what the chain does at the two moments an allowance ends: the shot must
move to the next service that still has credit, and the spent one must go quiet instead of
paying to rediscover the same wall on every remaining shot.

The second half is about profiles. fal and WaveSpeed put their fields in different places,
and a profile that reads the wrong one fails in the worst way available: it downloads
something that is not the clip, or polls a URL that never changes, and neither says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asa.core.errors import (                                          # noqa: E402
    AllProvidersExhausted, AuthError, ProviderError, QuotaExhausted)
from asa.media.video.base import RenderedShot, ShotJob                 # noqa: E402
from asa.media.video.factory import VideoChain                         # noqa: E402
from asa.media.video.hosted import (                                   # noqa: E402
    SERVICES, HostedVideo, ModelSpec, clip_url, data_uri)


def job(**kw) -> ShotJob:
    base = dict(image_path=Path("s.png"), frames=48, fps=24, size=(64, 36),
                envelope=[0.5] * 48, face=(0.1, 0.0, 0.8, 1.0),
                voice_path=Path("v.wav"))
    base.update(kw)
    return ShotJob(**base)


class Stub:
    """A provider that answers one way, to test what the CHAIN does about it."""

    def __init__(self, name, outcome=None):
        self.name, self.outcome, self.calls = name, outcome, 0
        self.available = True

    def accepts(self, job):
        return self.outcome is not None or True

    def render(self, job, dest):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return RenderedShot(path=dest, provider=self.name)


# ------------------------------------------------------- spending several allowances


def test_a_spent_allowance_moves_to_the_next_service_not_to_local():
    """The whole reason there is more than one hosted entry in the chain.

    Free allowances are small and separate, so the first one running out is expected rather
    than exceptional. If that dropped the shot to the local renderer it would lose lip-sync
    and gesture - the two things the hosted call exists for - while another account with
    credit sat unused directly below it.
    """
    broke = Stub("fal", QuotaExhausted("allowance spent", provider="fal"))
    funded = Stub("wavespeed")
    local = Stub("local")
    out = VideoChain([broke, funded, local]).render(job(), Path("out.mp4"))
    assert out.provider == "wavespeed", "a spent allowance must advance, not give up"
    assert local.calls == 0, "the free renderer is a floor, not the next step down"


def test_a_rejected_key_at_one_service_does_not_end_the_episode():
    """AuthError used to escape the chain and fail the render outright.

    That was defensible with a single hosted provider - the alternative was silently
    downgrading a whole episode to the local renderer. With several, a key that is missing,
    revoked or mistyped at ONE service is precisely the case the others are there to cover,
    and aborting would throw away allowances that are still good.
    """
    bad_key = Stub("fal", AuthError("key rejected"))
    funded = Stub("wavespeed")
    out = VideoChain([bad_key, funded, Stub("local")]).render(job(), Path("out.mp4"))
    assert out.provider == "wavespeed"


def test_the_chain_still_fails_loudly_when_every_service_is_out():
    """Advancing past each failure must not become swallowing all of them."""
    with pytest.raises(AllProvidersExhausted) as e:
        VideoChain([Stub("fal", QuotaExhausted("spent", provider="fal")),
                    Stub("wavespeed", AuthError("nope")),
                    Stub("local", ProviderError("ffmpeg gone"))]).render(
                        job(), Path("out.mp4"))
    assert "fal" in str(e.value) and "wavespeed" in str(e.value) and "local" in str(e.value)


def test_a_service_that_cannot_pay_is_asked_only_once():
    """Learning an account is broke costs a full retry cycle - minutes, on a throttled one.

    Paying that on every shot is what made a twelve-shot episode wait forty minutes to be
    told the same thing twelve times. Once the answer is known, `accepts` declines for free
    and the chain reaches the next service immediately.
    """
    p = HostedVideo(SERVICES["fal"], key="k", lipsync_model="m")

    def broke(client, spec, inputs):
        raise QuotaExhausted("no credit", provider="fal")

    p._produce = broke
    p._asset = lambda client, path: "data:,"
    with pytest.raises(QuotaExhausted):
        p.render(job(voice_offset_s=0.0, voice_duration_s=0.0), Path("out.mp4"))
    assert p.accepts(job()) is False, "the second shot must not pay to learn this again"


# --------------------------------------------------------------- reading a profile


class FakeResp:
    def __init__(self, payload, code=200):
        self.status_code, self._payload = code, payload
        self.headers, self.text = {}, str(payload)

    def json(self):
        return self._payload


class FakeClient:
    """Answers by URL, and records every URL it was asked for."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def _answer(self, url):
        self.seen.append(url)
        for pattern, resp in self.routes.items():
            if pattern in url:
                return resp
        raise AssertionError(f"nothing configured for {url}")

    def post(self, url, **kw):
        self.last_post = kw
        return self._answer(url)

    def get(self, url, **kw):
        return self._answer(url)


def test_the_fal_profile_polls_one_url_and_collects_from_another():
    """fal splits status from result, and the split is easy to miss.

    The status endpoint says COMPLETED and contains no clip; the clip is only in the
    response the SECOND url serves. A profile that read the output from the status payload
    would poll forever or return nothing, having been told the job succeeded.
    """
    p = HostedVideo(SERVICES["fal"], key="k", lipsync_model="fal-ai/x", poll_s=0)
    client = FakeClient({
        "queue.fal.run/fal-ai/x": FakeResp({"request_id": "r1",
                                            "status_url": "https://q/st",
                                            "response_url": "https://q/re"}),
        "https://q/st": FakeResp({"status": "COMPLETED"}),
        "https://q/re": FakeResp({"video": {"url": "https://cdn/clip.mp4"}})})
    got = p._produce(client, ModelSpec("fal-ai/x", {"image": "image_url"}), {})
    assert got == "https://cdn/clip.mp4"
    assert "https://q/st" in client.seen, "the status url must actually be polled"
    assert "https://q/re" in client.seen, "the clip lives behind the response url"


def test_the_wavespeed_profile_builds_its_poll_url_and_reads_through_data():
    """WaveSpeed hands back an id rather than a url, and wraps everything in `data`.

    Two chances to read the wrong field. Getting the id path wrong submits fine and then
    cannot find the job; getting the envelope wrong finds a status of `None`, which is not
    terminal, so the shot burns its whole timeout before failing.
    """
    p = HostedVideo(SERVICES["wavespeed"], key="k", lipsync_model="ws/s2v", poll_s=0)
    client = FakeClient({
        "api/v3/ws/s2v": FakeResp({"code": 200, "data": {"id": "t1"}}),
        "predictions/t1/result": FakeResp(
            {"code": 200, "data": {"status": "completed",
                                   "outputs": ["https://cdn/w.mp4"]}})})
    got = p._produce(client, ModelSpec("ws/s2v", {"image": "image"}), {})
    assert got == "https://cdn/w.mp4"
    assert any("predictions/t1/result" in u for u in client.seen)


def test_an_unknown_status_keeps_polling_but_a_failed_one_stops():
    """These services add queue states more often than they remove them.

    So an unrecognised status has to mean "still working" - treating it as failure would
    break the provider the first time a service renamed `IN_QUEUE`. A status the profile
    knows is terminal must still stop immediately, or a permanently failed job is polled
    until the shot times out.
    """
    p = HostedVideo(SERVICES["wavespeed"], key="k", lipsync_model="ws/s2v", poll_s=0)
    client = FakeClient({
        "api/v3/ws/s2v": FakeResp({"data": {"id": "t1"}}),
        "predictions/t1/result": FakeResp(
            {"data": {"status": "failed", "error": "model blew up"}})})
    with pytest.raises(ProviderError) as e:
        p._produce(client, ModelSpec("ws/s2v", {"image": "image"}), {})
    assert "model blew up" in str(e.value)


def test_a_clip_url_is_not_whatever_http_string_comes_first():
    """The payloads carry log, seed-image and thumbnail urls next to the video.

    Returning the first http string found downloads a JPEG and hands it to ffmpeg as a clip,
    which fails as a conform error about a file that looks perfectly valid.
    """
    assert clip_url({"thumbnail_url": "https://cdn/thumb.jpg",
                     "video": {"url": "https://cdn/real.mp4"}}) == "https://cdn/real.mp4"
    assert clip_url({"logs": "https://cdn/log.txt",
                     "result": {"file": "https://cdn/real.mp4"}}) == "https://cdn/real.mp4"
    assert clip_url({"logs": "https://cdn/log.txt"}) is None


# ------------------------------------------------------------------ handing over files


def test_a_service_that_wants_urls_is_not_sent_a_data_uri(tmp_path):
    """WaveSpeed rejects data URIs, so the file is uploaded and the returned URL is sent.

    Worth a test because the failure is silent in the confusing direction: sending a data
    URI to a service that wants a URL is not an upload error, it is a schema error on every
    model in the chain, reported as though the mapping were wrong.
    """
    p = HostedVideo(SERVICES["wavespeed"], key="k", lipsync_model="ws/s2v")
    still = tmp_path / "s.png"
    still.write_bytes(b"\x89PNG\r\n----")
    client = FakeClient({"media/upload/binary":
                         FakeResp({"data": {"download_url": "https://cdn/s.png"}})})
    assert p._asset(client, still) == "https://cdn/s.png"
    assert "files" in client.last_post, "the file must be posted, not inlined"


def test_a_service_that_takes_data_uris_does_not_make_an_upload_round_trip(tmp_path):
    """fal accepts data URIs, and an upload it never needed is a request per asset per shot."""
    p = HostedVideo(SERVICES["fal"], key="k", lipsync_model="fal-ai/x")
    still = tmp_path / "s.png"
    still.write_bytes(b"\x89PNG\r\n----")
    client = FakeClient({})            # any request at all would raise
    assert p._asset(client, still).startswith("data:image/png;base64,")
    assert client.seen == []


def test_a_failed_upload_is_an_error_rather_than_a_quiet_data_uri(tmp_path):
    """Falling back to inlining would turn one clear error into a schema error per model."""
    p = HostedVideo(SERVICES["wavespeed"], key="k", lipsync_model="ws/s2v")
    still = tmp_path / "s.png"
    still.write_bytes(b"\x89PNG\r\n----")
    client = FakeClient({"media/upload/binary": FakeResp({"data": {}}, code=200)})
    with pytest.raises(ProviderError) as e:
        p._asset(client, still)
    assert "no URL" in str(e.value)


# ------------------------------------------------------------------ status mapping


@pytest.mark.parametrize("code,expected", [
    (401, AuthError), (403, AuthError), (402, QuotaExhausted), (404, ProviderError),
    (500, ProviderError)])
def test_status_codes_map_onto_the_class_the_chain_routes_on(code, expected):
    """402 and 429 must not be confused: one advances the chain, the other waits and retries
    the same provider. Getting them the wrong way round either burns an episode backing off
    from a permanent condition, or abandons a service that was merely busy."""
    p = HostedVideo(SERVICES["fal"], key="k", lipsync_model="m")
    with pytest.raises(expected):
        p._status_error(FakeResp({}, code=code), "m")


def test_a_quota_message_behind_a_400_is_still_a_quota_problem():
    """Not every service answers 402. One that says `insufficient balance` under a 400 would
    otherwise be retried as an ordinary failure, on every shot, for the whole episode."""
    p = HostedVideo(SERVICES["fal"], key="k", lipsync_model="m")
    r = FakeResp({}, code=400)
    r.text = "insufficient balance for this request"
    with pytest.raises(QuotaExhausted):
        p._status_error(r, "m")


def test_an_inline_asset_over_the_cap_is_named_rather_than_truncated(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * (9 * 1024 * 1024))
    with pytest.raises(ProviderError) as e:
        data_uri(big, "fal")
    assert "big.png" in str(e.value)


# ----------------------------------------------------- profiles that reshape the request


def test_the_dashscope_profile_names_the_model_in_the_body_and_nests_the_input():
    """Wan first-party posts to ONE route and names the model in the body.

    Which breaks the assumption every earlier profile shared - that the model is a path
    segment - in a way that fails confusingly rather than loudly: submitting to
    `{base}/wan2.2-s2v` is a 404 on a route that does not exist, reported as "no such
    model", sending you to check a slug that was right all along.

    The envelope matters just as much. DashScope splits the body into `input` (the assets)
    and `parameters` (everything else) and ignores what is in the wrong half, so a
    resolution sent inside `input` is not an error - it is a 720p bill for a 480p ask.
    """
    svc = SERVICES["dashscope"]
    assert svc.submit_url("wan2.2-s2v").endswith("/video-generation/video-synthesis")
    body = svc.wire_body("wan2.2-s2v", {"image_url": "data:image/png;base64,AA",
                                        "prompt": "the fox speaks",
                                        "parameters.resolution": "480P"})
    assert body["model"] == "wan2.2-s2v", "the model is a body field here, not a path"
    assert body["input"] == {"image_url": "data:image/png;base64,AA",
                             "prompt": "the fox speaks"}
    assert body["parameters"] == {"resolution": "480P"}, "a dotted name goes to the root"
    assert svc.auth_headers("k")["X-DashScope-Async"] == "enable", (
        "without this header the service tries to answer synchronously and times out")


def test_the_dashscope_profile_polls_its_task_and_reads_the_uppercase_status():
    """A task id under `output`, a poll route of its own, and SUCCEEDED rather than
    succeeded. An unrecognised status is treated as still-working, so reading the wrong
    field here does not fail - it polls until the shot's timeout expires."""
    p = HostedVideo(SERVICES["dashscope"], key="k", lipsync_model="wan2.2-s2v", poll_s=0)
    client = FakeClient({
        "video-synthesis": FakeResp({"output": {"task_id": "t9",
                                                "task_status": "PENDING"}}),
        "/tasks/t9": FakeResp({"output": {"task_status": "SUCCEEDED",
                                          "results": {"video_url": "https://cdn/w.mp4"}}})})
    got = p._produce(client, ModelSpec("wan2.2-s2v", {"image": "image_url"}), {})
    assert got == "https://cdn/w.mp4"
    assert any("/tasks/t9" in u for u in client.seen)


def test_the_google_profile_sends_a_batch_of_one_and_an_asset_as_an_object():
    """Veo takes a list of instances and refuses a data URI for the image.

    Both are silent-ish failures in opposite directions: a bare dict where a list belongs is
    an INVALID_ARGUMENT that names a field rather than the shape, and a data URI in
    `bytesBase64Encoded` is accepted as base64 - of the string "data:image/png;base64,..."
    - which decodes to nothing recognisable as an image.
    """
    svc = SERVICES["google"]
    assert svc.submit_url("veo-3.0-fast-generate-001").endswith(
        "/models/veo-3.0-fast-generate-001:predictLongRunning")
    body = svc.wire_body("veo-3.0-fast-generate-001", {
        "image": "data:image/png;base64,QUJD", "prompt": "a slow push in",
        "parameters.durationSeconds": 8, "parameters.aspectRatio": "16:9"})
    assert body["instances"] == [{"image": {"bytesBase64Encoded": "QUJD",
                                            "mimeType": "image/png"},
                                  "prompt": "a slow push in"}]
    assert body["parameters"] == {"durationSeconds": 8, "aspectRatio": "16:9"}
    assert "model" not in body, "Google names the model in the route, not the body"
    assert svc.auth_headers("k") == {"x-goog-api-key": "k"}, (
        "an API key is not a Bearer token here - that header is for OAuth")


def test_the_google_profile_treats_a_boolean_done_as_terminal_and_finds_a_bare_uri():
    """Two Google-shaped hazards in one exchange.

    The operation reports completion with `done: true` rather than with a status string, and
    the clip's URI is a Files endpoint with no `.mp4` on the end - so the generic hunt that
    protects every other service from downloading a thumbnail would reject the real clip.
    """
    p = HostedVideo(SERVICES["google"], key="k", i2v_model="veo-3.0-fast-generate-001",
                    speaking_only=False, poll_s=0)
    op = "models/veo-3.0-fast-generate-001/operations/abc"
    uri = "https://generativelanguage.googleapis.com/v1beta/files/xyz:download?alt=media"
    client = FakeClient({
        ":predictLongRunning": FakeResp({"name": op}),
        "/operations/abc": FakeResp(
            {"name": op, "done": True,
             "response": {"generateVideoResponse": {
                 "generatedSamples": [{"video": {"uri": uri}}]}}})})
    got = p._produce(client, ModelSpec("veo-3.0-fast-generate-001", {"image": "image"}), {})
    assert got == uri


def test_the_google_clip_is_fetched_with_the_key_and_the_others_without_it():
    """Veo's returned URI is on the API itself and answers an anonymous GET with 401.

    Which arrives as a corrupt mp4 rather than as an error, because by then we are streaming
    a response body to a file and calling it a clip. Every other service hands back a signed
    URL, and sending our key to their CDN is not something to do by default.
    """
    google = HostedVideo(SERVICES["google"], key="gk", i2v_model="veo",
                         speaking_only=False)
    assert google._download_headers() == {"x-goog-api-key": "gk"}
    assert HostedVideo(SERVICES["fal"], key="k", lipsync_model="m")._download_headers() == {}


def test_the_reseller_bodies_are_untouched_by_the_reshaping():
    """The profile fields the first-party services needed must be no-ops for the others.

    This is the regression that would be found late and blamed on the model: fal and
    WaveSpeed take our mapped fields AS the body, and a stray `model` key or an `input`
    wrapper is a schema error on every shot, for providers that were working.
    """
    flat = {"image_url": "data:image/png;base64,AA", "prompt": "p", "duration": 5}
    for name in ("fal", "wavespeed"):
        assert SERVICES[name].wire_body("some/model", flat) == flat
        assert SERVICES[name].submit_url("some/model").endswith("/some/model")


def test_an_i2v_only_provider_declines_a_speaking_shot_rather_than_posting_an_empty_slug():
    """Veo has no audio-driven mode, so a Google entry is i2v-only by nature.

    Before this it was indistinguishable from a half-configured provider: the empty string
    went out as a model slug on every speaking shot and came back 404, spending the shot's
    wall clock to reach the local renderer it should have declined to immediately.
    """
    p = HostedVideo(SERVICES["google"], key="k", i2v_model="veo", speaking_only=False)
    assert p.accepts(job()) is False, "nothing configured for a speaking shot"
    assert p.accepts(job(face=None)) is True, "the shots it CAN render are still its own"
    assert p.lipsync_specs == [], "an empty model must not become a spec"


def test_a_lipsync_only_provider_declines_the_other_half_the_same_way():
    """The mirror case, and the one that already exists in config: several entries name a
    lipsync model and no i2v model, and with `speaking_only: false` they would have posted
    an empty slug for every insert."""
    p = HostedVideo(SERVICES["fal"], key="k", lipsync_model="fal-ai/x",
                    speaking_only=False)
    assert p.accepts(job()) is True
    assert p.accepts(job(face=None)) is False


def test_the_novita_profile_polls_a_query_string_and_reads_a_nested_task_status():
    """The id goes into a query parameter rather than into the path, and the status lives
    one level down under `task`. Reading the top level finds no status at all, which is not
    terminal - so the shot polls until its timeout instead of failing."""
    p = HostedVideo(SERVICES["novita"], key="k", lipsync_model="wan2.7-i2v", poll_s=0)
    client = FakeClient({
        "/async/wan2.7-i2v": FakeResp({"task_id": "n1"}),
        "task-result?task_id=n1": FakeResp(
            {"task": {"status": "TASK_STATUS_SUCCEED"},
             "videos": [{"video_url": "https://cdn/n.mp4"}]})})
    got = p._produce(client, ModelSpec("wan2.7-i2v", {"image": "image_url"}), {})
    assert got == "https://cdn/n.mp4"
    assert any("task-result?task_id=n1" in u for u in client.seen)


def test_the_siliconflow_profile_polls_by_posting_the_id_to_a_fixed_route():
    """Every other service serves the job AT a url. SiliconFlow POSTs the id to one status
    route, so the id has to survive as far as the poll - and a GET there is a 404 that
    reads like the job never existed."""
    p = HostedVideo(SERVICES["siliconflow"], key="k", i2v_model="Wan-AI/Wan2.2-I2V-A14B",
                    speaking_only=False, poll_s=0)
    client = FakeClient({
        "/video/submit": FakeResp({"requestId": "r7"}),
        "/video/status": FakeResp({"status": "Succeed",
                                   "results": {"videos": [{"url": "https://cdn/s.mp4"}]}})})
    got = p._produce(client, ModelSpec("Wan-AI/Wan2.2-I2V-A14B", {"image": "image"}), {})
    assert got == "https://cdn/s.mp4"
    assert client.last_post["json"] == {"requestId": "r7"}, (
        "the poll body carries the id; without it the route has nothing to look up")


def test_every_profile_can_name_a_submit_route_and_present_a_key():
    """A smoke test over the whole table, because a new entry is a dict literal and the way
    one goes wrong is a typo in a field name that nothing references until a render."""
    for name, svc in SERVICES.items():
        assert svc.submit_url("m/1").startswith("https://"), name
        assert svc.auth_headers("k"), name
        assert svc.poll_path or svc.poll_template, f"{name} has nothing to poll"
        if svc.poll_method == "POST":
            assert svc.poll_body_field, f"{name} polls by POST with no field for the id"
