"""Build the shot-video provider chain from config.

`local` is always appended, whether or not it is listed: it needs no key and no network, so
it is the floor under any hosted video model that is out of credit, rate limited or slow.

The chain is walked per SHOT, not per episode, because the providers differ in what they
are worth spending on. A hosted lip-sync model is the only thing that answers the "the
character is static and the voice comes from the background" complaint, and it is billed by
the second; the local renderer is free and perfectly good for a slow push across a
landscape. So a provider may decline an individual shot (`accepts`) and the next one picks
it up - an episode routinely comes out of here part hosted, part local.

More than one HOSTED provider is worth having for a different reason: each is a separate
account with its own small free allowance, and the interesting failure is running out. A
provider that is spent raises QuotaExhausted, which advances this chain rather than failing
the shot, so the allowances are consumed one after another and only what none of them could
render falls to the local path. Its own model chain is a level below this one - see
`hosted.HostedShotProvider` - and the two do different jobs: that one recovers from a bad
model, this one from a bad account.

Which is why a rejected KEY advances the chain here too. With a single hosted provider that
was worth aborting for, since the alternative was silently downgrading a whole episode. With
several, a key that is missing or revoked at one service is exactly what the others cover.
"""
from __future__ import annotations

from ...core.errors import AllProvidersExhausted, AuthError, ProviderError
from ...core.logging import get_logger
from .base import RenderedShot, ShotJob
from .hosted import SERVICES, HostedVideo
from .local import LocalPerformance
from .replicate import DEFAULT_I2V_MODEL, DEFAULT_LIPSYNC_MODEL, ReplicateVideo

log = get_logger("video")


class VideoChain:
    def __init__(self, providers: list):
        self.providers = [p for p in providers if p is not None]

    def render(self, job: ShotJob, dest) -> RenderedShot:
        failures: list[str] = []
        for p in self.providers:
            if not getattr(p, "available", False):
                failures.append(f"{p.name}: unavailable")
                continue
            accepts = getattr(p, "accepts", None)
            if accepts is not None and not accepts(job):
                failures.append(f"{p.name}: declined this shot")
                continue
            try:
                return p.render(job, dest)
            except AuthError as e:
                # Advance rather than abort. With one hosted provider a rejected key was
                # worth stopping for, because the alternative was a silent downgrade to the
                # local renderer for the whole episode. With several, a key that has been
                # revoked or typed wrong at ONE service is exactly the case the others are
                # here to cover, and failing the episode over it would throw away the
                # allowances that are still good. Logged at its own name because, unlike
                # running out of credit, it never fixes itself.
                failures.append(f"{p.name}: key rejected")
                log.warning("shot_provider_key_rejected", provider=p.name,
                            error=str(e)[:160])
            except (ProviderError, RuntimeError, OSError) as e:
                failures.append(f"{p.name}: {str(e)[:100]}")
                log.warning("shot_provider_failed", provider=p.name, error=str(e)[:160])
        raise AllProvidersExhausted("every video provider failed: " + " | ".join(failures))


def build_video_chain(cfg, workers: int | None = None, crf: int = 20) -> VideoChain:
    order = cfg.get("providers.video.chain", ["local"]) or ["local"]
    built, skipped = [], []
    for name in order:
        if name == "local":
            continue                      # appended below; listing it changes nothing
        if name == "replicate":
            base = "providers.video.replicate"
            p = ReplicateVideo(
                token=cfg.secret("REPLICATE_API_TOKEN", required=False),
                # Defaults come from the provider module, not from a second copy of the
                # slug here. Two places to change a model id is one place to forget: a
                # config missing the key would have kept pointing at the old talking-head
                # model long after the module default moved to a whole-figure one.
                lipsync_model=cfg.get(f"{base}.lipsync_model", DEFAULT_LIPSYNC_MODEL),
                i2v_model=cfg.get(f"{base}.i2v_model", DEFAULT_I2V_MODEL),
                lipsync_inputs=cfg.get(f"{base}.lipsync_inputs"),
                i2v_inputs=cfg.get(f"{base}.i2v_inputs"),
                i2v_durations=cfg.get(f"{base}.i2v_durations"),
                lipsync_extra=cfg.get(f"{base}.lipsync_extra"),
                i2v_extra=cfg.get(f"{base}.i2v_extra"),
                lipsync_fallbacks=cfg.get(f"{base}.lipsync_fallbacks"),
                i2v_fallbacks=cfg.get(f"{base}.i2v_fallbacks"),
                speaking_only=bool(cfg.get(f"{base}.speaking_only", True)),
                max_shots=int(cfg.get(f"{base}.max_shots", 40)),
                timeout_s=float(cfg.get(f"{base}.timeout_s", 900)),
                poll_s=float(cfg.get(f"{base}.poll_s", 3.0)),
                crf=crf)
        elif name in SERVICES:
            service = SERVICES[name]
            base = f"providers.video.{name}"
            p = HostedVideo(
                service, key=cfg.secret(service.env, required=False),
                # No module-level default model per service, unlike Replicate: these are
                # here to be swapped as free allowances run out, and a slug guessed in code
                # for a service the user may never have signed up to would fail every shot
                # with a 404 instead of being skipped for having no key.
                lipsync_model=cfg.get(f"{base}.lipsync_model", ""),
                i2v_model=cfg.get(f"{base}.i2v_model", ""),
                lipsync_inputs=cfg.get(f"{base}.lipsync_inputs"),
                i2v_inputs=cfg.get(f"{base}.i2v_inputs"),
                i2v_durations=cfg.get(f"{base}.i2v_durations"),
                lipsync_extra=cfg.get(f"{base}.lipsync_extra"),
                i2v_extra=cfg.get(f"{base}.i2v_extra"),
                lipsync_fallbacks=cfg.get(f"{base}.lipsync_fallbacks"),
                i2v_fallbacks=cfg.get(f"{base}.i2v_fallbacks"),
                speaking_only=bool(cfg.get(f"{base}.speaking_only", True)),
                max_shots=int(cfg.get(f"{base}.max_shots", 40)),
                timeout_s=float(cfg.get(f"{base}.timeout_s", 900)),
                poll_s=float(cfg.get(f"{base}.poll_s", 3.0)),
                crf=crf)
            if p.available and not (p.lipsync_model or p.i2v_model):
                # A key with nothing to spend it on. Skipped loudly rather than left in the
                # chain to fail every shot on an empty slug.
                #
                # EITHER model is enough, and one of them alone is a legitimate provider
                # rather than a half-configured one: Veo has no audio-driven mode, so a
                # Google entry is i2v-only by nature. A provider configured for one mode
                # declines the other in `accepts` and the chain carries that shot on.
                skipped.append(f"{name}(no lipsync_model or i2v_model configured)")
                continue
        else:
            skipped.append(f"{name}(unknown)")
            continue
        if p.available:
            built.append(p)
        else:
            skipped.append(f"{name}(no credentials)")
    built.append(LocalPerformance(
        workers=workers, crf=crf,
        lipsync=bool(cfg.get("production.performance.lipsync", True)),
        blink=bool(cfg.get("production.performance.blink", True)),
        min_face_confidence=float(
            cfg.get("production.performance.min_face_confidence", 0.5))))
    if skipped:
        log.warning("video_providers_skipped", skipped=skipped)
    log.info("video_chain_built", active=[p.name for p in built])
    return VideoChain(built)
