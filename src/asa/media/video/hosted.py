"""Metered shot rendering, and the several services that will do it for you.

Two things live here. The first is everything a hosted video provider does that is NOT its
HTTP dialect: deciding which shots are worth spending on, mapping our fields onto a model's
schema, walking a per-mode model chain, remembering that the account cannot pay, and forcing
whatever comes back into the frame count the timeline promised. All of that was written for
Replicate and none of it is about Replicate, so it is a base class (`HostedShotProvider`)
and Replicate is one subclass of it.

The second is the observation that the rest - submit a job, poll it, collect a URL - is the
same three steps at every one of these services, differing only in where each puts its
fields. fal answers with a `status_url` to poll and a separate `response_url` to collect;
WaveSpeed wraps everything in `data` and builds the poll URL from an id. That is a
difference in JSON paths, not in behaviour, so a service is a `Service` profile rather than
a module, and `HostedVideo` renders through any of them.

WHY MORE THAN ONE SERVICE. Every one of these bills GPU-seconds, and each gives new
accounts a small free allowance. One allowance does not run this pipeline; several, walked
in order as each is spent, render meaningfully more shots than the first one alone before
anything has to be paid for. `QuotaExhausted` from a provider already advances the chain in
`factory.VideoChain`, so the whole mechanism is: implement the protocol, appear in
`providers.video.chain`, and go quiet when the credit runs out.

Read the limits honestly. Free allowances are sized for evaluation, not operation - the
total across every service here is a handful of clips, not a day's episodes - and some
services restrict what their free tier's output may be used for. This code makes the
allowances reachable and makes running out a non-event. It does not make hosted video free.
"""
from __future__ import annotations

import base64
import mimetypes
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx

from ...core.errors import AuthError, ProviderError, QuotaExhausted, RateLimited
from ...core.logging import get_logger
from .base import RenderedShot, ShotJob
from .conform import conform_clip

log = get_logger("video_hosted")

# Above this, an asset is too big to inline as a data URI. Services that would rather be
# handed a URL declare an `upload` on their profile and never reach this.
MAX_INLINE_BYTES = 8 * 1024 * 1024

# Extensions that make a URL look like the clip rather than like a log or a thumbnail.
CLIP_SUFFIXES = (".mp4", ".webm", ".mov", ".m4v", ".mkv")

# Keys worth trying before falling back to a recursive hunt, most specific first. These
# services return model-shaped payloads rather than one agreed envelope, and the alternative
# to a list like this is a per-model output path in config that nobody would fill in.
URL_KEYS = ("video", "url", "uri", "download_url", "video_url", "output", "outputs",
             "data")


@dataclass(frozen=True)
class ModelSpec:
    """One hosted model, with the mapping and settings that make our fields fit ITS schema.

    Grouped rather than held as parallel lists because every field here is a property of the
    model rather than of us. OmniHuman takes `image` and `audio` and rejects anything else;
    Wan-S2V wants a `prompt` on top and refuses to run without one; SadTalker calls the same
    two files `source_image` and `driven_audio` and needs `preprocess` set or it crops our
    composed frame down to a floating head; Kling's `duration` is an enum of two values while
    Wan has no duration field at all.

    Which is exactly why a fallback list of bare slugs would be a trap: the second model
    would inherit the first's mapping and fail every prediction it was added to rescue,
    reporting a schema error at the moment the primary was already broken.
    """

    model: str
    inputs: dict
    extra: dict = field(default_factory=dict)
    # Clip lengths the model offers, when it offers a fixed set. Empty = a free number.
    durations: tuple = ()


def model_spec(entry: dict, role: str, provider: str = "") -> ModelSpec | None:
    """One configured fallback, or None with a reason logged.

    Skipped rather than raised: a mistyped fallback should not take down an episode whose
    primary model is working. `asa video preflight` is where a bad one is meant to be found,
    and it reports every model in the chain, not just the first.
    """
    model = str(entry.get("model") or "").strip()
    inputs = entry.get("inputs") or {}
    if not model or not inputs.get("image"):
        log.warning("video_fallback_ignored", provider=provider, role=role,
                    entry=str(entry)[:120],
                    reason="needs a `model` and an `inputs` mapping naming at least `image`")
        return None
    return ModelSpec(model=model, inputs=dict(inputs), extra=dict(entry.get("extra") or {}),
                     durations=tuple(entry.get("durations") or ()))


def data_uri(path: Path, provider: str = "") -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_INLINE_BYTES:
        raise ProviderError(
            f"{path.name} is {len(raw) // 1024}KB, over the {MAX_INLINE_BYTES // 1024}KB "
            f"inline limit for this provider", provider=provider, retryable=False)
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def voice_slice(job: ShotJob, workdir: Path, provider: str = "") -> Path:
    """The wav a lip-sync model should actually be driven by: this SHOT's audio.

    `voice_path` is a whole synthesised line, and a shot is often one part of a speech that
    several setups cover between them. Sending the whole line and trimming the returned
    video would lip-sync every part of the speech to the line's opening words - the mouth
    moves convincingly and says the wrong thing, which is worse than not moving at all.

    Returns the file untouched only when the job names no window at all - which is what a
    caller that has not been taught about windows looks like, and the whole line is the right
    answer for it. A window that happens to span the whole line is still cut, because nothing
    here knows the file's length without probing it, and cutting a few seconds of PCM is
    cheaper than finding out.

    PADDED TO THE SHOT, not to the speech, and that difference is visible. An audio-driven
    model generates as much video as it is given audio: hand it the 3.78s of speech that a
    4.47s shot contains and it returns 3.84s, leaving 0.63s the conformer can only fill by
    holding the last frame. Measured on the first hosted clip of scene 1 - the final 19 of
    134 frames had literally zero pixel change, so a performance that was working ended in a
    freeze on every shot whose voice stops before the cut does.

    Trailing silence is part of the shot: it is the beat where the character has finished
    speaking and has not yet been cut away from, and a model given that silence animates it
    as a closed mouth and a settling body, which is what it should look like.
    """
    src = job.voice_path
    if src is None:
        raise ProviderError("no voice for a speaking shot", provider=provider,
                            retryable=False)
    if job.voice_offset_s <= 0.02 and job.voice_duration_s <= 0.0:
        return src
    dur = job.voice_duration_s or job.duration_s
    # Never SHORTER than the shot. `apad` runs forever, so the output `-t` is what actually
    # sets the length; taking the max keeps a voice that overruns its shot intact rather
    # than clipping a word off the end of it.
    pad_to = max(dur, job.duration_s)
    if shutil.which("ffmpeg") is None:
        raise ProviderError("ffmpeg not found on PATH", provider=provider, retryable=False)
    out = workdir / f"voice_{job.seed}.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{job.voice_offset_s:.3f}"]
    if dur > 0:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(src), "-af", "apad", "-t", f"{pad_to:.3f}",
            "-c:a", "pcm_s16le", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise ProviderError(f"could not cut {src.name} for this shot: "
                            f"{r.stderr.strip()[:200]}", provider=provider)
    return out


def dig(obj, path: tuple):
    """Walk a tuple of keys into nested dicts, returning None rather than raising."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def clip_url(obj) -> str | None:
    """The URL of the produced clip, wherever this service decided to put it.

    Preferred keys first, then a recursive hunt restricted to things that look like video
    files. The restriction matters: several of these payloads also carry log, thumbnail and
    seed-image URLs, and returning the first http string found would download a JPEG and
    hand it to ffmpeg as a clip.
    """
    if isinstance(obj, str):
        return obj if obj.startswith("http") else None
    if isinstance(obj, list):
        for item in obj:
            found = clip_url(item)
            if found:
                return found
        return None
    if isinstance(obj, dict):
        for key in URL_KEYS:
            if key in obj:
                found = clip_url(obj[key])
                if found:
                    return found
        for value in obj.values():
            found = clip_url(value)
            if found and found.lower().split("?")[0].endswith(CLIP_SUFFIXES):
                return found
    return None


def place(body: dict, path, value) -> None:
    """Set a nested key, creating the dicts on the way down.

    The counterpart to `dig`, and here for the same reason: these services differ in where
    a field goes, not in which fields exist. A mapping may name its target with dots -
    `duration: parameters.durationSeconds` - and land it outside the input envelope without
    a profile field per case.
    """
    node = body
    for key in list(path)[:-1]:
        node = node.setdefault(key, {})
    node[list(path)[-1]] = value


def as_asset_object(value, keys: tuple):
    """A data URI as the {bytes, mime} object some services want instead.

    Google's is the API that insists: it takes an image as `bytesBase64Encoded` beside a
    `mimeType` rather than as a URI, and rejects the data URI our other services accept.
    Anything that is not a data URI - an uploaded file's URL, a prompt, a number - is
    returned untouched, so a profile can set this without auditing every field it sends.
    """
    if not (isinstance(value, str) and value.startswith("data:") and ";base64," in value):
        return value
    head, _, b64 = value.partition(";base64,")
    return {keys[0]: b64, keys[1]: head[len("data:"):]}


class HostedShotProvider:
    """A metered shot renderer, minus the HTTP dialect a particular service speaks.

    Subclasses implement `_produce` - submit one model's input and come back with the URL of
    a clip - and inherit the parts that are the same wherever the rendering happens: the
    spend policy, the model chain, the credit latch and the frame-count contract.
    """

    name = "hosted"
    # Default field mappings, overridden per subclass where the house style differs. Only
    # ever read through `dict(... or ...)` in the constructor, so no instance can reach in
    # and mutate the class's copy for every other provider in the chain.
    default_lipsync_inputs: ClassVar[dict] = {"image": "image", "audio": "audio"}
    default_i2v_inputs: ClassVar[dict] = {"image": "image", "prompt": "prompt"}

    def __init__(self, key: str | None, *, lipsync_model: str = "",
                 i2v_model: str = "",
                 lipsync_inputs: dict | None = None, i2v_inputs: dict | None = None,
                 i2v_durations: list | None = None,
                 lipsync_extra: dict | None = None, i2v_extra: dict | None = None,
                 lipsync_fallbacks: list | None = None, i2v_fallbacks: list | None = None,
                 speaking_only: bool = True, max_shots: int = 40,
                 timeout_s: float = 900.0, poll_s: float = 3.0, crf: int = 20):
        self.token = (key or "").strip()
        self.lipsync_model = lipsync_model
        self.i2v_model = i2v_model
        self.lipsync_inputs = dict(lipsync_inputs or self.default_lipsync_inputs)
        self.i2v_inputs = dict(i2v_inputs or self.default_i2v_inputs)
        self.i2v_durations = list(i2v_durations or [])
        # Kept apart per mode on purpose. These services validate the input against the
        # chosen model's schema and reject unknown fields, so one shared bag of extras means
        # SadTalker's `preprocess` gets posted to Kling and fails the prediction.
        self.lipsync_extra = dict(lipsync_extra or {})
        self.i2v_extra = dict(i2v_extra or {})
        self.speaking_only = speaking_only
        self.max_shots = max_shots
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.crf = crf
        self.rendered = 0
        # Set once the account has told us it cannot pay. See `_latch_no_credit`.
        self._no_credit: str | None = None

        # The per-mode model chain: the configured primary, then any fallbacks, tried in
        # order for one shot. This is a different axis from the PROVIDER chain in
        # factory.py, and both are needed. Falling straight to the local renderer when a
        # hosted model errors gives up lip-sync and gesture for the shot - the two things
        # the hosted call was made for - over something as ordinary as a slug that was
        # renamed overnight or a model that choked on one image. Another hosted model is a
        # much smaller step down than no model at all, and the mode's list is ordered by how
        # much of the performance survives: whole figure, then whole figure from a prompt,
        # then a talking head, and only then the local camera move.
        #
        # A mode with no model configured contributes NO specs, and `accepts` then declines
        # that shot to whoever is next in the provider chain. Which is not hypothetical
        # tidiness: Veo has no audio-driven mode at all, so a Google entry is an i2v-only
        # provider by nature, and the alternative to declining is posting the empty string
        # as a model slug and failing every speaking shot on a 404.
        self.lipsync_specs = ([ModelSpec(self.lipsync_model, self.lipsync_inputs,
                                         self.lipsync_extra)] if self.lipsync_model else [])
        self.i2v_specs = ([ModelSpec(self.i2v_model, self.i2v_inputs, self.i2v_extra,
                                     tuple(self.i2v_durations))] if self.i2v_model else [])
        for entries, specs, role in ((lipsync_fallbacks, self.lipsync_specs, "lipsync"),
                                     (i2v_fallbacks, self.i2v_specs, "i2v")):
            for entry in (entries or []):
                spec = model_spec(entry, role, self.name)
                if spec is not None:
                    specs.append(spec)

    # ------------------------------------------------------------------ policy

    @property
    def available(self) -> bool:
        return bool(self.token)

    def accepts(self, job: ShotJob) -> bool:
        """Whether this provider should be spending its allowance on this particular shot.

        Declining is not failing: the chain moves to the next provider - another service
        with credit left, or eventually the local renderer. That is the whole point of
        answering per shot rather than per episode: the talking close-ups get a real model,
        the landscape gets a free slow push.
        """
        if not self.available:
            return False
        if self._no_credit:
            return False
        if self.rendered >= self.max_shots:
            log.warning("hosted_budget_reached", provider=self.name,
                        max_shots=self.max_shots)
            return False
        if self.speaking_only and not (job.voice_path and job.speaking):
            return False
        # Configured for the OTHER mode only. An i2v-only service asked for a speaking shot
        # has nothing to answer with, and the local renderer - or the next provider - does.
        if not self._candidates(job)[0]:
            return False
        return True

    def _latch_no_credit(self, reason: str) -> None:
        """Remember, for the rest of this process, that this account cannot pay.

        Learning it can cost a full retry cycle - several attempts and minutes of backoff,
        measured on Replicate - because an uncredited account is throttled to a trickle and
        answers 429 to the very retries trying to reach the honest 402. Paying that once is
        the price of being sure it is not ordinary congestion.

        Paying it again on every remaining shot is not. The chain is built once per animate
        stage and these instances are reused across every scene, so without this flag a
        twelve-speaking-shot episode waits the full cycle twelve times to be told the same
        thing, and all twelve shots move on to the next provider regardless.

        With several free allowances in the chain this is what makes running out a
        non-event: the exhausted service goes quiet after one shot pays to discover it, and
        every later shot goes straight to whichever service still has credit.
        """
        if self._no_credit:
            return
        self._no_credit = reason
        log.warning("hosted_provider_out_of_credit", provider=self.name,
                    reason=reason[:200],
                    detail="skipping every remaining shot rather than re-learning this")

    # ------------------------------------------------------------------ inputs

    def _clip_duration(self, job: ShotJob, durations: tuple) -> int:
        """What to ask an i2v model for, given it may only offer fixed lengths.

        Per model, not per provider: Kling accepts 5 or 10 and rejects everything else,
        while Wan takes no duration at all. A fallback that inherited the primary's enum
        would fail on the very shot it was added to rescue.

        Never round DOWN to a shorter clip than the shot: the missing frames would be filled
        by cloning the last one, which freezes the picture at the end of every shot. Ask for
        the shortest offered length that covers it and let `conform_clip` retime.
        """
        want = max(1.0, job.duration_s)
        if not durations:
            return max(1, round(want))
        longer = [d for d in sorted(durations) if d >= want]
        return int(longer[0] if longer else max(durations))

    def _candidates(self, job: ShotJob) -> tuple[list[ModelSpec], bool]:
        """(models to try in order, preserve_timing) for this shot."""
        if job.voice_path and job.speaking:
            # Lip-synced output must never be retimed: stretching it slides the mouth off
            # the very voice this provider was paid to match.
            return self.lipsync_specs, True
        return self.i2v_specs, False

    def _inputs_for(self, spec: ModelSpec, job: ShotJob, image_uri: str,
                    audio_uri: str | None) -> dict:
        """This model's input dict, built from ITS mapping rather than the primary's."""
        m = spec.inputs
        payload = {m["image"]: image_uri}
        if audio_uri is not None and m.get("audio"):
            payload[m["audio"]] = audio_uri
        # Only when the mapping declares one. The audio-driven models split on this:
        # OmniHuman takes image and audio and works the gesture out for itself, while
        # Wan-S2V requires a prompt and REJECTS the prediction without one - and these
        # services reject unknown fields just as hard in the other direction, so a prompt
        # sent unconditionally would fail every OmniHuman shot. Keyed off the mapping, both
        # are a config edit.
        #
        # Worth sending where it is accepted: `motion_prompt` is the scene's own action
        # text, so the motion is directed by what the scene says is happening rather than
        # left to the model's idea of a person talking.
        if m.get("prompt"):
            default = ("the character speaks to camera, natural gesture"
                       if audio_uri is not None else "subtle natural motion, cinematic")
            payload[m["prompt"]] = job.motion_prompt or default
        if m.get("duration"):
            payload[m["duration"]] = self._clip_duration(job, spec.durations)
        payload.update(spec.extra)
        return payload

    def _payload(self, job: ShotJob, workdir: Path,
                 spec: ModelSpec | None = None) -> tuple[str, dict, bool]:
        """(model, input, preserve_timing) for ONE model of this shot's chain - the primary
        unless `spec` names another.

        Kept as a single entry point even though `_render` walks the chain itself: a failed
        prediction has to be reproducible from somewhere, and "what exactly did we send that
        model" is the first question every schema bug asks.
        """
        specs, preserve = self._candidates(job)
        spec = spec or specs[0]
        image_uri = data_uri(job.image_path, self.name)
        audio_uri = (data_uri(voice_slice(job, workdir, self.name), self.name)
                     if preserve else None)
        return spec.model, self._inputs_for(spec, job, image_uri, audio_uri), preserve

    # ------------------------------------------------------------------ http

    def _headers(self) -> dict:
        raise NotImplementedError

    def _asset(self, client: httpx.Client, path: Path) -> str:
        """How this service wants to be handed a local file. Inline unless told otherwise."""
        return data_uri(path, self.name)

    def _produce(self, client: httpx.Client, spec: ModelSpec, inputs: dict) -> str:
        """Run one model to completion and return the URL of the clip it made."""
        raise NotImplementedError

    def _status_error(self, r: httpx.Response, model: str) -> None:
        """Map one HTTP response onto the taxonomy the chain routes on.

        The distinction that earns its keep here is 402 from 429. Out of credit means this
        provider is finished for the run and the next one should be tried immediately; rate
        limited means wait and ask this one again. Getting them the wrong way round either
        burns an episode's wall clock backing off from a permanent condition, or gives up a
        provider that was only busy.
        """
        code = r.status_code
        if code in (401, 403):
            raise AuthError(f"{self.name} rejected the API key ({code}) for {model}")
        if code == 429:
            # Honour the server's own number when it gives one - guessing shorter just
            # spends another attempt to be told the same thing.
            try:
                after = float(r.headers.get("retry-after") or 0.0)
            except ValueError:
                after = 0.0
            raise RateLimited(f"{self.name} rate limited: {r.text[:120]}",
                              provider=self.name, retry_after_s=max(after, 12.0))
        if code == 402:
            raise QuotaExhausted(f"{self.name} allowance exhausted", provider=self.name)
        if code == 404:
            raise ProviderError(
                f"{self.name} has no model '{model}' available to this account; set the "
                f"provider's *_model to one you can run",
                provider=self.name, retryable=False)
        if code >= 400:
            # Some of these services answer 200 with an error body and 400 with a quota
            # message, so the text is checked as well as the code. Cheap, and the cost of
            # missing it is the whole chain backing off from a permanent condition.
            if self._looks_broke(r.text):
                raise QuotaExhausted(f"{self.name} allowance exhausted: {r.text[:120]}",
                                     provider=self.name)
            raise ProviderError(f"{self.name} {code} for {model}: {r.text[:200]}",
                                provider=self.name)

    @staticmethod
    def _looks_broke(text: str) -> bool:
        low = (text or "").lower()
        return any(s in low for s in ("insufficient", "quota", "out of credit",
                                      "no credit", "balance", "exceeded your",
                                      "payment required"))

    def _download_headers(self) -> dict:
        """Headers for the GET that collects the finished clip.

        Empty for the services that hand back a signed URL anyone may fetch. Google does
        not: its `uri` is a Files endpoint on the API itself, and an anonymous GET is
        answered 401 - which arrives as a corrupt download rather than as an error, because
        by then we are streaming bytes to a file and calling it an mp4.
        """
        return {}

    def _fetch(self, client: httpx.Client, url: str, raw: Path) -> None:
        raw.parent.mkdir(parents=True, exist_ok=True)
        with client.stream("GET", url, headers=self._download_headers(),
                           timeout=300.0) as resp:
            resp.raise_for_status()
            with raw.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 16):
                    fh.write(chunk)

    # ------------------------------------------------------------------ render

    def render(self, job: ShotJob, dest: Path) -> RenderedShot:
        if not self.accepts(job):
            raise ProviderError(f"{self.name} declined this shot", provider=self.name)
        t0 = time.time()
        tmp = Path(tempfile.mkdtemp(prefix="asa_shot_"))
        try:
            return self._render(job, dest, tmp, t0)
        except QuotaExhausted as e:
            self._latch_no_credit(str(e))
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _render(self, job: ShotJob, dest: Path, tmp: Path, t0: float) -> RenderedShot:
        specs, preserve = self._candidates(job)
        mode = "lipsync" if preserve else "i2v"
        raw = dest.with_suffix(".raw.mp4")
        used: ModelSpec | None = None
        tried: list[str] = []
        with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as client:
            # Prepared once, not once per candidate: the inputs are the same picture and the
            # same slice of speech whichever model is asked to animate them, and neither
            # base64 of a 1024px still nor an upload round trip is free.
            image_uri = self._asset(client, job.image_path)
            audio_uri = (self._asset(client, voice_slice(job, tmp, self.name))
                         if preserve else None)
            for i, spec in enumerate(specs):
                last = i == len(specs) - 1
                try:
                    url = self._produce(
                        client, spec, self._inputs_for(spec, job, image_uri, audio_uri))
                    self._fetch(client, url, raw)
                    used = spec
                    break
                except (QuotaExhausted, AuthError):
                    # Not this model's fault and not fixable by picking another one: an
                    # account with no credit and a rejected key fail identically on every
                    # slug in the list. Trying the rest would spend the shot's wall clock to
                    # be told the same thing three times. The PROVIDER chain is the right
                    # place to recover from this, and it does.
                    raise
                except (ProviderError, httpx.HTTPError) as e:
                    tried.append(f"{spec.model}: {str(e)[:100]}")
                    if last:
                        raise ProviderError(
                            f"every {mode} model failed: " + " | ".join(tried),
                            provider=self.name) from e
                    log.warning("video_model_failed_falling_back", provider=self.name,
                                mode=mode, failed=spec.model, next=specs[i + 1].model,
                                error=str(e)[:160])

        assert used is not None
        try:
            stats = conform_clip(raw, dest, frames=job.frames, fps=job.fps,
                                 size=job.size, preserve_timing=preserve, crf=self.crf)
        finally:
            raw.unlink(missing_ok=True)

        self.rendered += 1
        elapsed = round(time.time() - t0, 2)
        stats.update({"seconds": elapsed, "model": used.model, "mode": mode})
        if tried:
            stats["fell_back_from"] = tried
        log.info("hosted_shot_ok", provider=self.name, model=used.model, seconds=elapsed,
                 mode=mode, frames=job.frames, spent_shots=self.rendered,
                 fell_back=len(tried) or None)
        return RenderedShot(path=dest, provider=self.name, model_id=used.model, stats=stats)


# ---------------------------------------------------------------------------- services


@dataclass(frozen=True)
class Service:
    """Where one service puts the fields that every one of them has.

    A profile rather than a subclass because the differences really are this small. Adding a
    service that speaks one of these dialects - and the API-compatible resellers all do - is
    a `SERVICES` entry, or a `base_url` override in config with no code at all.

    The first-party APIs stretched that further than the resellers did, in one direction
    worth naming. fal and WaveSpeed name the model in the URL and take our mapped fields as
    the whole request body; DashScope and Google post to ONE fixed route, name the model in
    the body, and want those fields nested inside an envelope. That is still a difference in
    shape rather than in behaviour - submit, poll, collect is unchanged - so it is three
    more profile fields (`submit_path`, `model_field`, `input_path`) rather than a subclass.
    """

    name: str
    base: str
    # Env var the key comes from, and how it is presented. fal says `Key`, most say `Bearer`,
    # and Google does not use the Authorization header at all.
    env: str
    scheme: str = "Bearer"
    auth_header: str = "Authorization"
    # Sent with every request on top of auth. DashScope will not QUEUE a video job without
    # `X-DashScope-Async`; it tries to answer synchronously instead and times out.
    headers: dict = field(default_factory=dict)
    # Where the submit goes, and where the model is named. The resellers put the model in
    # the path; the first-party APIs have one route and take it in the body.
    submit_path: str = "{base}/{model}"
    model_field: str = ""
    # Where our mapped fields sit in the body. Empty = they ARE the body. `input_list` wraps
    # them in a one-element list, for a route whose parameter is a batch.
    input_path: tuple = ()
    input_list: bool = False
    # (bytes key, mime key) for a service that wants an asset as an object rather than as a
    # data URI. Set only where the API insists: it rewrites every data URI in the body.
    asset_object: tuple = ()
    # Where the submit response hides the things needed to collect the result. A service
    # gives either a ready-made poll URL (`poll_path`) or an id to build one from
    # (`id_path` + `poll_template`).
    poll_path: tuple = ()
    id_path: tuple = ("id",)
    poll_template: str = ""
    # How the poll is made. Most serve the job at a URL; SiliconFlow POSTs the id to one
    # fixed status route instead, so the id has to survive as far as the poll rather than
    # being baked into a URL.
    poll_method: str = "GET"
    poll_body_field: str = ""
    # Where the polled response keeps its status, and which values are terminal.
    status_path: tuple = ("status",)
    done: frozenset = frozenset({"completed", "succeeded", "success"})
    failed: frozenset = frozenset({"failed", "error", "canceled", "cancelled", "timeout"})
    # Some services answer the poll with the result inline; others hand back a second URL to
    # collect it from once the status says it is ready.
    result_path: tuple = ()
    # Where to start looking for the clip URL, before the generic hunt in `clip_url`.
    output_path: tuple = ()
    error_path: tuple = ("error",)
    # Set when the service will not take a data URI and wants files uploaded first.
    upload: str = ""
    # Set when the finished clip is served from behind the same key as the API. Google's is:
    # the returned URI is a Files endpoint that answers 401 to an anonymous GET.
    download_auth: bool = False

    # ---------------------------------------------------------------- request shape

    def auth_headers(self, token: str) -> dict:
        value = f"{self.scheme} {token}".strip() if self.scheme else token
        return {self.auth_header: value, **self.headers}

    def submit_url(self, model: str) -> str:
        return self.submit_path.format(base=self.base, model=model)

    def wire_body(self, model: str, payload: dict) -> dict:
        """The JSON this service wants, from the flat {field: value} a mapping produces.

        Three transformations, each of which is some service's hard requirement and every
        one of them a no-op on a profile that does not ask for it - which is why the fal and
        WaveSpeed bodies come out of here byte for byte what they were before this existed:

          * a dotted target name is nested at the ROOT rather than in the input envelope, so
            `duration: parameters.durationSeconds` reaches Google's `parameters` without a
            profile field for every such case;
          * a data URI becomes an object where `asset_object` says the API wants one;
          * what is left is placed at `input_path`, and the model named at `model_field`.

        Built here rather than in `_inputs_for` because none of it is a property of the
        model, which is what that mapping describes. Two services host the same Wan-S2V and
        disagree only about the envelope around it.
        """
        body: dict = {}
        inner: dict = {}
        for name, value in payload.items():
            if self.asset_object:
                value = as_asset_object(value, self.asset_object)
            if "." in name:
                place(body, name.split("."), value)
            else:
                inner[name] = value
        if self.input_path:
            place(body, self.input_path, [inner] if self.input_list else inner)
        else:
            body.update(inner)
        if self.model_field:
            body[self.model_field] = model
        return body


SERVICES: dict[str, Service] = {
    # Submit returns a status_url to poll and a response_url to collect from. Accepts data
    # URIs for file inputs, so no upload step. Hosts OmniHuman and Wan-S2V, which is why it
    # is the first alternative worth having: the same models as Replicate, a separate
    # allowance.
    "fal": Service(
        name="fal", base="https://queue.fal.run", env="FAL_KEY", scheme="Key",
        poll_path=("status_url",), result_path=("response_url",),
        done=frozenset({"COMPLETED"}),
        failed=frozenset({"FAILED", "ERROR", "CANCELED", "CANCELLED"}),
        output_path=("video",), error_path=("error",)),

    # Everything is wrapped in `data`, and the poll URL is built from an id rather than
    # handed over. Wants real URLs rather than data URIs, hence the upload endpoint. The
    # cheapest paid rate of the three once the free credit is gone, which is why it is worth
    # keeping after the allowance rather than only during it.
    "wavespeed": Service(
        name="wavespeed", base="https://api.wavespeed.ai/api/v3",
        env="WAVESPEED_API_KEY", scheme="Bearer",
        id_path=("data", "id"),
        poll_template="{base}/predictions/{id}/result",
        status_path=("data", "status"),
        done=frozenset({"completed"}),
        failed=frozenset({"failed", "cancelled", "canceled", "timeout", "deleted"}),
        output_path=("data", "outputs"), error_path=("data", "error"),
        upload="{base}/media/upload/binary"),

    # Alibaba Model Studio - Wan FIRST PARTY. The three services above are resellers of the
    # same open-weights Wan-2.2, so this is not a fourth model, it is the model without a
    # middleman: another free allowance on signup, and the cheapest per second of the four
    # once that is gone. Worth having for exactly the reason `chain` exists.
    #
    # INTERNATIONAL endpoint. The mainland one (dashscope.aliyuncs.com) is a separate
    # region with separate accounts, and a key from the wrong one authenticates nowhere -
    # which arrives as 401 and reads like a mistyped key.
    #
    # Everything is async by header rather than by route: without `X-DashScope-Async` the
    # service tries to answer a video job synchronously and the request times out.
    "dashscope": Service(
        name="dashscope", base="https://dashscope-intl.aliyuncs.com/api/v1",
        env="DASHSCOPE_API_KEY", scheme="Bearer",
        headers={"X-DashScope-Async": "enable"},
        submit_path="{base}/services/aigc/video-generation/video-synthesis",
        model_field="model", input_path=("input",),
        id_path=("output", "task_id"), poll_template="{base}/tasks/{id}",
        status_path=("output", "task_status"),
        done=frozenset({"SUCCEEDED"}),
        failed=frozenset({"FAILED", "CANCELED", "CANCELLED", "UNKNOWN"}),
        output_path=("output",), error_path=("output", "message")),

    # Novita. Free credit on signup ($1 at the time of writing), and the only service in
    # this table carrying a Wan NEWER than 2.2 - `wan2.7-i2v`, where the two-versions-old
    # model everywhere else is the actual quality ceiling of the shot list.
    #
    # Its i2v route documents an `audio_url` input, which if it behaves like Wan-S2V makes
    # this a lip-sync provider as well. TREAT THAT AS UNPROVEN until one shot has been
    # watched: the failure mode is not a broken render, it is a confident performance of
    # the WRONG words, and lip-synced clips are never retimed so nothing downstream catches
    # it. `speech_to_video` is not a separate route here, unlike everywhere else.
    "novita": Service(
        name="novita", base="https://api.novita.ai/v3", env="NOVITA_API_KEY",
        scheme="Bearer",
        submit_path="{base}/async/{model}",
        id_path=("task_id",),
        poll_template="{base}/async/task-result?task_id={id}",
        status_path=("task", "status"),
        done=frozenset({"TASK_STATUS_SUCCEED"}),
        failed=frozenset({"TASK_STATUS_FAILED", "TASK_STATUS_CANCELED"}),
        output_path=("videos",), error_path=("task", "reason")),

    # SiliconFlow. Wan again, i2v only - no audio-driven model at all - and the one service
    # here that polls by POST: the id goes in the body of a fixed `/video/status` route
    # rather than into a URL. Takes the still as a data URI, so no upload step.
    #
    # A signed URL that expires in ONE HOUR, which is fine for us (the clip is fetched
    # within seconds of the poll succeeding) and worth knowing before anyone caches a
    # payload for later.
    "siliconflow": Service(
        name="siliconflow", base="https://api.siliconflow.com/v1",
        env="SILICONFLOW_API_KEY", scheme="Bearer",
        submit_path="{base}/video/submit", model_field="model",
        id_path=("requestId",),
        poll_template="{base}/video/status",
        poll_method="POST", poll_body_field="requestId",
        status_path=("status",),
        done=frozenset({"Succeed"}),
        failed=frozenset({"Failed"}),
        output_path=("results", "videos"), error_path=("reason",)),

    # Google, via the Gemini API. Veo, and NOT Google Flow: Flow is the subscription web
    # app built on this model and has no API to call, so "use Flow" means this endpoint.
    #
    # It is the odd one out here in the way that matters most to a shot list. Veo has no
    # audio-driven mode - it generates its own dialogue rather than performing ours - so it
    # cannot answer a speaking shot at all, and this profile is i2v only. It also charges
    # several times what Wan does. See the config block before putting it in the chain.
    #
    # Its dialect is the furthest from the resellers': the model is in the path but with a
    # method suffix, the key is not a Bearer token, the image is an object rather than a
    # data URI, the input is a batch, completion is a boolean, and the finished clip is
    # behind the same key as the API. All six are profile fields.
    "google": Service(
        name="google", base="https://generativelanguage.googleapis.com/v1beta",
        env="GEMINI_API_KEY", scheme="", auth_header="x-goog-api-key",
        submit_path="{base}/models/{model}:predictLongRunning",
        input_path=("instances",), input_list=True,
        asset_object=("bytesBase64Encoded", "mimeType"),
        id_path=("name",), poll_template="{base}/{id}",
        # An operation reports completion with a boolean rather than with a status string,
        # and `dig` hands it over as one. A FAILED operation is also `done: true` - with an
        # `error` and no video - so it arrives as "no clip URL", carrying Google's own
        # message in the text. Less tidy than a failed status, and it fails the shot at the
        # same point either way.
        status_path=("done",), done=frozenset({"True"}), failed=frozenset(),
        output_path=("response", "generateVideoResponse", "generatedSamples"),
        error_path=("error", "message"), download_auth=True),
}


class HostedVideo(HostedShotProvider):
    """Any service whose dialect a `Service` profile describes."""

    def __init__(self, service: Service, key: str | None, **kw):
        self.service = service
        self.name = service.name
        super().__init__(key, **kw)

    def _headers(self) -> dict:
        return {**self.service.auth_headers(self.token),
                "Content-Type": "application/json"}

    def _download_headers(self) -> dict:
        return self.service.auth_headers(self.token) if self.service.download_auth else {}

    def _asset(self, client: httpx.Client, path: Path) -> str:
        """Inline, or uploaded first where the service insists on a URL.

        A failed upload is a ProviderError rather than a fall-through to a data URI: the
        services that publish an upload endpoint are the ones that reject data URIs, so
        "try it inline anyway" would turn one clear error into a schema error on every model
        in the chain.
        """
        svc = self.service
        if not svc.upload:
            return data_uri(path, self.name)
        url = svc.upload.format(base=svc.base)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        r = client.post(url, headers=svc.auth_headers(self.token),
                        files={"file": (path.name, path.read_bytes(), mime)})
        self._status_error(r, "media/upload")
        found = clip_url(r.json())
        if not found:
            raise ProviderError(f"{self.name} upload of {path.name} returned no URL: "
                                f"{r.text[:160]}", provider=self.name)
        return found

    def _produce(self, client: httpx.Client, spec: ModelSpec, inputs: dict) -> str:
        svc = self.service
        r = client.post(svc.submit_url(spec.model), headers=self._headers(),
                        json=svc.wire_body(spec.model, inputs))
        self._status_error(r, spec.model)
        submitted = r.json()

        ident = dig(submitted, svc.id_path)
        poll = dig(submitted, svc.poll_path) if svc.poll_path else None
        if not poll and svc.poll_template:
            if not ident:
                raise ProviderError(f"{self.name} submit returned no id for {spec.model}: "
                                    f"{r.text[:160]}", provider=self.name)
            poll = svc.poll_template.format(base=svc.base, id=ident)
        if not poll:
            raise ProviderError(f"{self.name} submit returned nothing to poll for "
                                f"{spec.model}: {r.text[:160]}", provider=self.name)

        payload = self._await(client, poll, spec.model, ident)
        if svc.result_path:
            # fal splits status from result: the status endpoint says COMPLETED and the
            # clip is only in the response the second URL serves.
            result_url = dig(submitted, svc.result_path)
            if result_url:
                rr = client.get(result_url, headers=self._headers())
                self._status_error(rr, spec.model)
                payload = rr.json()

        found = clip_url(dig(payload, svc.output_path) if svc.output_path else payload)
        if not found:
            found = clip_url(payload)
        if not found:
            raise ProviderError(f"{self.name} returned no clip URL for {spec.model}: "
                                f"{str(payload)[:160]}", provider=self.name)
        return found

    def _await(self, client: httpx.Client, poll: str, model: str, ident=None) -> dict:
        """Poll until the job reaches a terminal state, and return the last payload."""
        svc = self.service
        deadline = time.time() + self.timeout_s
        while True:
            if svc.poll_method == "POST":
                r = client.post(poll, headers=self._headers(),
                                json={svc.poll_body_field: ident})
            else:
                r = client.get(poll, headers=self._headers())
            self._status_error(r, model)
            payload = r.json()
            status = str(dig(payload, svc.status_path) or "")
            if status in svc.done:
                return payload
            if status in svc.failed:
                raise ProviderError(
                    f"{self.name} job {status} for {model}: "
                    f"{str(dig(payload, svc.error_path))[:200]}", provider=self.name)
            # An unrecognised status is treated as still working rather than as a failure:
            # these services add queue states more often than they remove them, and timing
            # out is the honest way to give up on one that never finishes.
            if time.time() > deadline:
                raise ProviderError(
                    f"{self.name} job timed out after {self.timeout_s:.0f}s for {model} "
                    f"(last status {status or 'unknown'})", provider=self.name)
            time.sleep(self.poll_s)
