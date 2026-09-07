"""Hosted image-to-video and lip-sync, through Replicate's prediction API.

Why this exists: the local 2.5D renderer cannot do lip-sync, and that is a property of the
approach rather than a bug to be tuned out. Measured on a finished episode, its jaw warp
moved the entire picture - glasses, hoodie, laptop and the bookshelf behind the character -
because it places the mouth from the framing word in the prompt, and the image model does
not reliably obey that word. One shot asked for a close-up of an owl and came back as a
wide two-shot of a rooftop; the warp then animated the skyline for forty seconds. The
attempted rescue - generating mouth-open and mouth-closed variants on a fixed seed and
blending between them - does not work either: with FLUX.1-schnell, changing one clause at a
fixed seed changed 63% of the pixels, so the pair are two different pictures, not two
expressions of one.

A model that was trained to move a face is the answer to moving a face. This provider is
the seam for one.

The same argument applies a second time, one level out. Nothing on the local path moves a
character's arms or legs either, and no amount of work on a jaw band ever will - there is no
gesture, pose or limb concept anywhere below this line, only a vertical displacement of a
photograph. So the default here is an audio-driven WHOLE-FIGURE model rather than a talking
head: one signal, one prediction, mouth and hands together.

WHAT IS ACTUALLY IN THIS FILE. Only the parts that are about Replicate: its two
incompatible prediction routes, its throttling behaviour, and the shape of its prediction
object. Everything else a metered renderer does - which shots are worth spending on, the
per-mode model chain, the out-of-credit latch, conforming the returned clip - lives in
`hosted.HostedShotProvider`, because none of it turned out to be about Replicate and there
is now more than one service in the chain.

MODEL IDS ARE CONFIG, NOT CODE. The defaults below are the shapes this was written
against, but hosted catalogues churn, and a wrong slug here would fail every shot at 3am
for a reason nobody could see. `lipsync_inputs` / `i2v_inputs` map our logical fields onto
whatever the chosen model calls its parameters, so pointing this at a different model is a
config change. NEEDS VERIFICATION on your own account before a paid run.

Cost is the reason for `speaking_only` and `max_shots`. Hosted video is billed per second
of output, an episode is a hundred-odd seconds, and a runaway retry loop is a bill rather
than a crash. Non-speaking shots - establishing wides, inserts, cutaways - have no mouth to
sync and are left to the local renderer, which is free and good enough for a slow push on a
landscape.
"""
from __future__ import annotations

import time

import httpx

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.logging import get_logger
from ...core.retry import RetryPolicy, with_retry
from .hosted import (  # noqa: F401
    MAX_INLINE_BYTES,
    HostedShotProvider,
    ModelSpec,
    data_uri,
    model_spec,
    voice_slice,
)

log = get_logger("video_replicate")

API = "https://api.replicate.com/v1"

# image + audio -> a performance. The one that actually matters for the complaint.
#
# OmniHuman rather than SadTalker, and the difference is the body. SadTalker animates a head
# and nothing else - with `still_mode` on, which is where its artefacts live, the character's
# hands and legs are as frozen as they are on the local path, so a character explaining
# something for forty seconds does it without moving an arm. OmniHuman drives the whole
# figure from the same audio: mouth, head, shoulders, arms and hands.
#
# Required inputs are exactly `image` and `audio`, so this is the plain shape below with no
# extras. NEEDS VERIFICATION on your own account before a paid run: the schema was read from
# the live catalogue, but no clip has been generated through it here - this account has no
# credit and every creation attempt is answered 402.
DEFAULT_LIPSYNC_MODEL = "bytedance/omni-human"
DEFAULT_LIPSYNC_INPUTS = {"image": "image", "audio": "audio"}
# image + prompt -> motion. Used for non-speaking shots when speaking_only is off.
DEFAULT_I2V_MODEL = "kwaivgi/kling-v1.6-standard"
DEFAULT_I2V_INPUTS = {"image": "start_image", "prompt": "prompt", "duration": "duration"}
# Some image-to-video models take an ENUM of clip lengths, not a number of seconds. Kling
# accepts 5 or 10 and rejects anything else, so a 4.5-second shot asking for `duration: 4`
# fails the whole prediction. Verified against the live schema. An empty list means the
# model takes a free number and the shot's own length is sent.
DEFAULT_I2V_DURATIONS = [5, 10]

# Replicate throttles prediction CREATION hard - six a minute on this account. Falling
# through to the local renderer on a 429 would be wrong twice over: the shot silently loses
# the lip-sync it was routed here for, and it does so for a reason that fixes itself in
# seconds. Backing off is the whole answer. Long delays are cheap next to a shot that takes
# a minute to generate anyway.
CREATE_RETRY = RetryPolicy(attempts=5, base_delay_s=12.0, max_delay_s=90.0)

# The names these helpers had while they lived in this module. Kept so that a traceback or
# an import written against the old layout still lands on the same function.
_spec = model_spec
_data_uri = data_uri
_voice_slice = voice_slice


class ReplicateVideo(HostedShotProvider):
    name = "replicate"
    default_lipsync_inputs = DEFAULT_LIPSYNC_INPUTS
    default_i2v_inputs = DEFAULT_I2V_INPUTS

    def __init__(self, token: str | None, *, lipsync_model: str = DEFAULT_LIPSYNC_MODEL,
                 i2v_model: str = DEFAULT_I2V_MODEL,
                 lipsync_inputs: dict | None = None, i2v_inputs: dict | None = None,
                 i2v_durations: list | None = None,
                 lipsync_extra: dict | None = None, i2v_extra: dict | None = None,
                 lipsync_fallbacks: list | None = None, i2v_fallbacks: list | None = None,
                 speaking_only: bool = True, max_shots: int = 40,
                 timeout_s: float = 900.0, poll_s: float = 3.0, crf: int = 20):
        # `i2v_durations` is resolved here rather than in the base class because an empty
        # list is a meaningful answer - "this model takes a free number of seconds" - and
        # only `None`, meaning nobody said, should inherit Kling's enum.
        super().__init__(
            token, lipsync_model=lipsync_model, i2v_model=i2v_model,
            lipsync_inputs=lipsync_inputs, i2v_inputs=i2v_inputs,
            i2v_durations=(DEFAULT_I2V_DURATIONS if i2v_durations is None
                           else i2v_durations),
            lipsync_extra=lipsync_extra, i2v_extra=i2v_extra,
            lipsync_fallbacks=lipsync_fallbacks, i2v_fallbacks=i2v_fallbacks,
            speaking_only=speaking_only, max_shots=max_shots, timeout_s=timeout_s,
            poll_s=poll_s, crf=crf)
        self._versions: dict[str, str] = {}

    # ------------------------------------------------------------------ api

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"}

    def _status_error(self, r: httpx.Response, model: str) -> None:
        """Replicate's wording over the shared status mapping in `hosted`.

        Only the codes whose message needs to name a Replicate concept are handled here -
        the config key to change, the billing page to visit. Everything else, including the
        429 handling that decides retry-versus-advance, is the shared behaviour.
        """
        if r.status_code == 402:
            raise QuotaExhausted("Replicate billing exhausted", provider=self.name)
        if r.status_code == 404:
            raise ProviderError(
                f"Replicate has no model '{model}' available to this account; set "
                f"providers.video.replicate.*_model to one you can run",
                provider=self.name, retryable=False)
        if r.status_code in (401, 403):
            raise AuthError(f"Replicate token rejected ({r.status_code}) for {model}")
        HostedShotProvider._status_error(self, r, model)

    # This module called it `_raise_for` before the shared base existed. One implementation,
    # two names, rather than two that can drift apart.
    _raise_for = _status_error

    def _latest_version(self, client: httpx.Client, model: str) -> str:
        """The id of a model's current version, cached for the life of this provider."""
        cached = self._versions.get(model)
        if cached:
            return cached
        r = client.get(f"{API}/models/{model}", headers=self._headers())
        self._raise_for(r, model)
        version = ((r.json().get("latest_version") or {}).get("id") or "")
        if not version:
            raise ProviderError(f"Replicate model '{model}' publishes no version to run",
                                provider=self.name, retryable=False)
        self._versions[model] = version
        log.info("replicate_version_resolved", model=model, version=version[:12])
        return version

    def _create(self, client: httpx.Client, model: str, payload: dict) -> dict:
        """Start a prediction, by whichever of Replicate's two routes this model needs.

        There are two, and which one works is a property of the model rather than of
        anything we can see in its slug:

          /models/{owner}/{name}/predictions   OFFICIAL models only
          /predictions  {"version": ...}       everything else

        A community model answers 200 to `GET /models/{slug}` and 404 to a prediction on the
        model route - so "the slug resolves" is not the same question as "the slug runs",
        and checking only the first is how a preflight goes green on a model that cannot be
        run. Rather than ask the caller to know the difference, fall back on the 404 and
        resolve the version.
        """
        try:
            return with_retry(lambda: self._create_once(client, model, payload),
                              CREATE_RETRY, label=f"replicate:{model}")
        except RateLimited as e:
            # Every attempt throttled. On this API that is more often a billing wall than
            # congestion: an account with no credit is squeezed to a trickle and only admits
            # to HTTP 402 once a request gets under the rate limit - so the honest 402 is
            # the one answer the retries are least likely to see. Say so in the error, or
            # the next person spends four minutes of backoff learning it again.
            raise QuotaExhausted(
                f"{e} - every attempt was throttled, which usually means the account is "
                f"out of credit rather than busy; check `asa video preflight`",
                provider=self.name) from e

    def _create_once(self, client: httpx.Client, model: str, payload: dict) -> dict:
        if ":" in model:
            _, _, version = model.partition(":")
            return self._create_by_version(client, model, version, payload)
        r = client.post(f"{API}/models/{model}/predictions",
                        headers=self._headers(), json={"input": payload})
        if r.status_code == 404:
            return self._create_by_version(
                client, model, self._latest_version(client, model), payload)
        self._raise_for(r, model)
        return r.json()

    def _create_by_version(self, client: httpx.Client, model: str, version: str,
                           payload: dict) -> dict:
        r = client.post(f"{API}/predictions", headers=self._headers(),
                        json={"version": version, "input": payload})
        self._raise_for(r, model)
        return r.json()

    def _cancel(self, client: httpx.Client, pred: dict, model: str) -> None:
        """Stop a prediction we are walking away from. Never raises."""
        pid = pred.get("id")
        cancel = (pred.get("urls") or {}).get("cancel")
        if not (pid or cancel):
            return
        try:
            r = client.post(cancel or f"{API}/predictions/{pid}/cancel",
                            headers=self._headers())
            log.info("replicate_prediction_cancelled", model=model, prediction=pid,
                     status=r.status_code)
        except Exception as e:                                        # noqa: BLE001
            log.warning("replicate_cancel_failed", model=model, prediction=pid,
                        error=str(e)[:120])

    def _await(self, client: httpx.Client, pred: dict, model: str) -> str:
        """Poll to completion and return the URL of the produced clip."""
        url = (pred.get("urls") or {}).get("get")
        deadline = time.time() + self.timeout_s
        while True:
            status = pred.get("status")
            if status == "succeeded":
                break
            if status in ("failed", "canceled"):
                raise ProviderError(
                    f"Replicate prediction {status} for {model}: "
                    f"{str(pred.get('error'))[:200]}", provider=self.name)
            if time.time() > deadline or not url:
                # CANCEL IT. Abandoning a prediction does not stop it - Replicate bills the
                # time it runs, and giving up locally just means paying for a clip nobody
                # will ever collect. Measured 2026-09-05: one OmniHuman prediction was left
                # behind by a killed render and accrued 1176 GPU-seconds (~$1.79) before it
                # was cancelled by hand, for a shot the pipeline had already moved past.
                #
                # Best effort, and deliberately so: this path is already an error, and a
                # failure to cancel must not replace the timeout with a confusing one.
                self._cancel(client, pred, model)
                raise ProviderError(
                    f"Replicate prediction timed out after {self.timeout_s:.0f}s "
                    f"for {model} (cancelled)", provider=self.name)
            time.sleep(self.poll_s)
            r = client.get(url, headers=self._headers())
            self._raise_for(r, model)
            pred = r.json()

        out = pred.get("output")
        if isinstance(out, list):
            out = out[-1] if out else None
        if isinstance(out, dict):
            out = out.get("video") or out.get("url")
        if not isinstance(out, str) or not out.startswith("http"):
            raise ProviderError(f"Replicate returned no clip URL for {model}: "
                                f"{str(out)[:120]}", provider=self.name)
        return out

    # ------------------------------------------------------------------ render

    def _produce(self, client: httpx.Client, spec: ModelSpec, inputs: dict) -> str:
        return self._await(client, self._create(client, spec.model, inputs), spec.model)
