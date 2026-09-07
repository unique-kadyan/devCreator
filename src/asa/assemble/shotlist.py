"""Cutting a scene into shots, so the picture changes when the speaker changes.

The cinematic path used to render one still per SCENE and hold it for the whole scene -
often twenty or thirty seconds - while the voices played over it. Nothing in the frame
belonged to the person speaking, so the result read as a photograph with a voice-over
rather than as characters talking to each other. That is a shot problem, not a lip-sync
problem: even a perfect mouth is unconvincing if the camera never cuts to whoever is
speaking.

A shot here is a contiguous run of lines by ONE speaker. Cuts land on speaker changes,
which is where an editor would put them, and a long single-speaker run is split so the
picture still moves.

Two properties this module is built around:

1. **It runs twice and must agree with itself.** The art stage plans shots before any
   audio exists (to know which images to generate); the animate stage plans them again
   with measured durations. `plan_shots` is therefore a pure function of the scene's
   dialogue, and `time_shots` is the only thing that needs the audio.

2. **Shots are cheap; images are not.** Several shots share one generated image via
   `image_key`, so a six-line exchange between two characters cuts six times while costing
   two image generations. Cutting back to the same setup is shot/reverse-shot grammar, not
   a mistake - it is what every dialogue scene ever filmed does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..core.logging import get_logger

log = get_logger("shotlist")

# Framings whose subject is a face big enough to animate. A wide shot of a character
# thirty metres away has no mouth to move, and pretending otherwise is what produces the
# rubbery smear that gives 2.5D animation a bad name.
FACE_FRAMINGS = ("close_up", "extreme_close_up", "medium", "over_shoulder")

# Rotations. Deterministic, so the art stage and the animate stage pick the same framing
# for the same shot without sharing state.
# `medium` is FIRST, and it was absent until 2026-09-04. The old reason was sound for the
# renderer that existed then: medium is the framing an image model interprets most loosely,
# so it is the one whose head box cannot be predicted, and an unpredictable head box is
# fatal to a warp that must be told where the mouth is.
#
# It stopped being the right trade when a hosted audio-driven model became the primary
# renderer for speaking shots (`providers.video.<service>.speaking_only`). These models are
# not given a head box and do not want one - they locate the figure themselves - so the
# objection does not apply to the path these shots now take. What DOES apply is that they
# animate the whole figure, and a head cropped "from the top edge to the bottom edge" has no
# shoulders, arms or hands for them to move.
#
# CONFIRMED 2026-09-05 on the first hosted clips this repo has ever rendered
# (wavespeed-ai/wan-2.2/speech-to-video, job 10 scene 7). The earlier note here reported
# "whole-frame motion 1.18 against local 0.71, lower-third 0.82 against 0.87 - i.e. no
# gesture at all", and that reading was wrong because the metric was confounded: a
# whole-frame pixel delta counts the local renderer's Ken Burns move, which displaces the
# BACKGROUND as much as the subject, so a still under a zoom scores like a performance.
#
# Measured instead as subject motion against a background the camera should leave alone:
#
#                              subject   background   ratio
#   hosted medium (gesture)       4.47         0.67    6.66
#   hosted close-up               4.68         0.88    5.33
#   local renderer                4.05         4.21    0.96
#   the reference video           6.24         4.45    1.40
#
# A ratio near 1.0 is the signature of nothing being animated - the subject moves no more
# than the wall behind it. The hosted medium shot really does gesture: the character looks
# down at a paper, lifts its head and raises a hand. The reference sits at 1.40 because it
# moves the camera AND the subject, which the hosted clips do not yet do - Wan-S2V ignored
# the `push_in` it was asked for (background 0.67 is a locked camera). That is the next
# thing to chase, and it is why `camera_move` is still sent.
#
# So the tight framings stay in the rotation for variety and for the cuts, and `medium`
# leads it, which is also what the reference this channel is tuned against does: sampled
# across its runtime it is almost entirely medium and wide shots with hands in frame.
#
# CAUTION: a medium shot that falls through to the LOCAL renderer is the case the original
# comment warned about, and `production.performance.min_face_confidence` is currently 0.0,
# which means local will warp on the 0.35 guess rather than decline. Raise that floor if
# the hosted chain is off, or medium shots will have their jaw band placed by a prior that
# was measured on landscape stills and is stale for the current portrait format.
#
# No repeats in this tuple, and that is load-bearing. It used to read
# ("close_up", "close_up", "over_shoulder"), which is fine when the index is a character's
# appearance count but wrong when it walks the PARTS of one long speech: two adjacent parts
# both resolved to `close_up`, so they shared an image_key and merged back into a single
# unbroken picture. A forty-second monologue that should have been four setups came out as
# a twenty-six-second hold on one still.
_SPEAKER_FRAMINGS = ("medium", "close_up", "over_shoulder", "extreme_close_up")
_NARRATION_FRAMINGS = ("wide", "full", "insert")
_SPEAKER_MOVES = ("push_in", "static", "handheld_drift", "pull_out")
_NARRATION_MOVES = ("pan_right", "push_in", "pan_left", "pull_out")

# Emotion collapses to three buckets before it reaches an image prompt. Ten emotions times
# two framings times three characters is thirty generations per scene; three buckets is
# what keeps a scene inside a handful of images while still letting a furious close-up look
# different from a delighted one.
_WARM = {"happy", "excited", "wry"}
_TENSE = {"sad", "scared", "angry", "surprised"}

# How long one setup may hold before the scene needs a different PICTURE rather than a
# different camera move across the same one.
#
# The distinction is the whole point. `_split_long` below already re-cut a long speech, but
# its sub-shots share the parent's image, so a sixteen-second narration was one still with
# three pans over it. Measured on a finished episode, that left 52% of the runtime resting
# on a photograph - scene 1 was 22% animated, scene 4 was 19% - which is the "static, voice
# from the background" complaint restated as a number.
PLAN_SHOT_S = 5.0
# Sub-shots per speech. Four, because the worst case this exists for is a long monologue -
# one character explaining something for forty seconds - and three setups still leaves
# thirteen seconds a picture. `max_images_per_scene` is the real ceiling; this is the shape.
MAX_PARTS = 4
# Words per second, for estimating a line's length BEFORE it has been synthesised.
#
# The split has to be decided in `plan_shots`, which runs at art time with no audio at all,
# because that is the stage that buys the images - deciding it later would ask the animate
# stage for pictures nobody paid for. Measured across the lines of a finished Hindi episode:
# most sit at 2.3-2.5 w/s. It only has to be close; `time_shots` uses the real durations.
WORDS_PER_SECOND = 2.3


def emotion_bucket(emotion: str | None) -> str:
    e = (emotion or "neutral").strip().lower()
    if e in _WARM:
        return "warm"
    if e in _TENSE:
        return "tense"
    return "neutral"


@dataclass(frozen=True)
class Shot:
    """One continuous run of picture. `image_key` is what it is a picture OF."""

    index: int
    speaker: str | None                 # character_id; None = narration
    lines: tuple[int, ...]              # dialogue.idx values this shot covers
    framing: str
    emotion: str
    camera_move: str
    image_key: str
    start_s: float = 0.0
    end_s: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    # Which slice of its speech this shot is, when one speech is covered by several setups.
    # `time_shots` divides the speech's measured span between the parts.
    part: int = 0
    parts: int = 1

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def speaking(self) -> bool:
        return self.speaker is not None

    @property
    def animates_face(self) -> bool:
        """Whether this shot's subject is close enough for mouth animation to read."""
        return self.speaking and self.framing in FACE_FRAMINGS


def _image_key(speaker: str | None, framing: str, bucket: str) -> str:
    return f"{speaker or 'narration'}|{framing}|{bucket}"


def plan_shots(scene: dict, *, max_images: int = 4) -> list[Shot]:
    """The shot structure of a scene, from its dialogue alone. No audio needed.

    `max_images` caps how many DISTINCT images the scene may generate. Past the cap a new
    shot reuses the nearest earlier setup for the same speaker, which is why a talkative
    scene cuts a dozen times on four images rather than costing a dozen generations.
    """
    lines = list(scene.get("dialogue") or [])
    cast = list((scene.get("staging") or {}).keys())
    scene_idx = int(scene.get("idx") or 0)
    scene_shot = (scene.get("shot") or "medium").strip().lower()

    if not lines:
        # No spoken line at all: the scene is its own establishing shot.
        key = _image_key(None, scene_shot, emotion_bucket(scene.get("emotion")))
        return [Shot(0, None, (), scene_shot, scene.get("emotion") or "neutral",
                     scene.get("camera_move") or "push_in", key)]

    groups: list[tuple[str | None, list[dict]]] = []
    for line in lines:
        cid = line.get("character_id")
        if groups and groups[-1][0] == cid:
            groups[-1][1].append(line)
        else:
            groups.append((cid, [line]))

    seen: dict[str, str] = {}                   # image_key -> framing actually generated
    occurrences: dict[str | None, int] = {}
    shots: list[Shot] = []
    for i, (speaker, glines) in enumerate(groups):
        occ = occurrences.get(speaker, 0)
        occurrences[speaker] = occ + 1
        emotion = glines[0].get("emotion") or scene.get("emotion") or "neutral"
        parts = _parts_for(glines)

        for k in range(parts):
            if speaker is None:
                # Varied by SCENE index, not just by position within the scene. Almost every
                # scene opens on one narration line, so keying off `occ` alone opened every
                # scene of the episode on the same wide establishing shot. `k` then walks a
                # long narration through wide -> full -> insert, so the picture keeps
                # changing instead of one still being panned across for twenty seconds.
                framing = _NARRATION_FRAMINGS[
                    (scene_idx + occ + k) % len(_NARRATION_FRAMINGS)]
                move = _NARRATION_MOVES[(i + k) % len(_NARRATION_MOVES)]
            else:
                framing = _SPEAKER_FRAMINGS[(occ + k) % len(_SPEAKER_FRAMINGS)]
                # An over-the-shoulder needs a shoulder to be over. Substitute the other
                # tight framing rather than `close_up`, which the neighbouring part is
                # probably already using - falling back onto it is how a solo scene ends up
                # holding one picture again.
                if framing == "over_shoulder" and len(cast) < 2:
                    framing = "extreme_close_up"
                move = _SPEAKER_MOVES[(i + k) % len(_SPEAKER_MOVES)]

            key = _image_key(speaker, framing, emotion_bucket(emotion))
            if key not in seen and len(seen) >= max_images:
                # Out of image budget: fall back to a setup already bought. The shot still
                # happens, it just cuts back to a picture the scene already owns.
                key = _fallback_key(seen, speaker)
                framing = seen[key]
            seen.setdefault(key, framing)

            shots.append(Shot(index=len(shots), speaker=speaker,
                              lines=tuple(int(g["idx"]) for g in glines),
                              framing=framing, emotion=emotion, camera_move=move,
                              image_key=key, part=k, parts=parts))
    return shots


def _parts_for(lines: list[dict]) -> int:
    """How many distinct setups a speech is worth, from its word count alone.

    An estimate on purpose: this runs before the audio exists. Getting it slightly wrong
    costs one image too many or leaves `_split_long` to re-cut the same picture, neither of
    which breaks anything - whereas waiting for real durations would mean the art stage
    could not know what to buy.
    """
    words = sum(len(str(line.get("line") or "").split()) for line in lines)
    est_s = words / WORDS_PER_SECOND
    return max(1, min(MAX_PARTS, math.ceil(est_s / PLAN_SHOT_S))) if PLAN_SHOT_S > 0 else 1


def _fallback_key(seen: dict[str, str], speaker: str | None) -> str:
    """Reuse an existing setup rather than paying for another image.

    Prefer one of the same character - cutting to a different framing of the right animal
    is a re-frame; cutting to a different animal is a continuity error.
    """
    prefix = f"{speaker or 'narration'}|"
    for key in reversed(list(seen)):
        if key.startswith(prefix):
            return key
    return next(iter(seen))


def image_keys(shots: list[Shot]) -> list[str]:
    """Distinct setups, in first-appearance order. This is the art stage's shopping list."""
    out: list[str] = []
    for s in shots:
        if s.image_key not in out:
            out.append(s.image_key)
    return out


def time_shots(shots: list[Shot], timing, fps: int = 24, *, min_shot_s: float = 0.75,
               max_shot_s: float = 6.5) -> list[Shot]:
    """Give the planned shots real boundaries, measured from the synthesised audio.

    Cuts land in the SILENCE between two lines rather than on a line's first frame: cutting
    exactly as someone starts talking reads as a late cut, because a viewer needs a beat to
    register the new face before the voice arrives.
    """
    total_frames = max(1, int(round(timing.duration_s * fps)))
    cues = list(timing.cues)
    if not shots:
        return []
    if not cues:
        one = replace(shots[0], index=0, start_s=0.0, end_s=timing.duration_s,
                      start_frame=0, end_frame=total_frames)
        return [one]

    spans: list[tuple[float, float]] = []
    for shot in shots:
        mine = [cues[i] for i in shot.lines if 0 <= i < len(cues)]
        if not mine:
            spans.append((float("nan"), float("nan")))
            continue
        lo = min(c.start_s for c in mine)
        hi = max(c.end_s for c in mine)
        if shot.parts > 1:
            # Several setups cover one speech, and they all carry the same line indices, so
            # without this every part would claim the whole span and land on top of the
            # others. Divide it evenly; the cut falls mid-speech, which is what a cutaway is.
            step = (hi - lo) / shot.parts
            lo, hi = lo + step * shot.part, lo + step * (shot.part + 1)
        spans.append((lo, hi))

    timed: list[Shot] = []
    for i, (shot, span) in enumerate(zip(shots, spans)):
        if span[0] != span[0]:                      # NaN: no audio for these lines
            continue
        prev_end = next((spans[j][1] for j in range(i - 1, -1, -1)
                         if spans[j][1] == spans[j][1]), None)
        next_start = next((spans[j][0] for j in range(i + 1, len(spans))
                           if spans[j][0] == spans[j][0]), None)
        start = 0.0 if prev_end is None else (prev_end + span[0]) / 2.0
        end = timing.duration_s if next_start is None else (span[1] + next_start) / 2.0
        timed.append(replace(shot, start_s=max(0.0, start),
                             end_s=min(timing.duration_s, max(start + 0.1, end))))

    timed = _split_long(timed, max_shot_s)
    timed = _absorb_flashes(timed, min_shot_s)
    return _quantise(timed, fps, total_frames)


def _split_long(shots: list[Shot], max_shot_s: float) -> list[Shot]:
    """A speech that runs past `max_shot_s` is re-cut inside itself.

    The sub-shots share the parent's image - so this costs nothing - and differ only in
    camera move. Holding one framing for twenty seconds is what made the old output feel
    frozen; two moves across the same setup reads as a second camera.
    """
    out: list[Shot] = []
    for shot in shots:
        n = max(1, min(3, math.ceil(shot.duration_s / max_shot_s))) if max_shot_s > 0 else 1
        if n == 1:
            out.append(shot)
            continue
        step = shot.duration_s / n
        for k in range(n):
            move = _SPEAKER_MOVES[(shot.index + k) % len(_SPEAKER_MOVES)] \
                if shot.speaking else _NARRATION_MOVES[(shot.index + k) % len(_NARRATION_MOVES)]
            out.append(replace(shot, start_s=shot.start_s + k * step,
                               end_s=shot.start_s + (k + 1) * step,
                               camera_move=move))
    return out


def _absorb_flashes(shots: list[Shot], min_shot_s: float) -> list[Shot]:
    """Drop a cut that would flash by, but never one that lands on a speaker change.

    A quarter-second cut to the same setup is a mistake. A quarter-second cut to whoever
    just said "Haan!" is the whole point - dropping it would leave a silent face on screen
    while a different voice speaks, which is the failure this module exists to fix.
    """
    out: list[Shot] = []
    for shot in shots:
        if (out and shot.duration_s < min_shot_s
                and out[-1].image_key == shot.image_key):
            out[-1] = replace(out[-1], end_s=shot.end_s,
                              lines=out[-1].lines + shot.lines)
            continue
        out.append(shot)
    return [replace(s, index=i) for i, s in enumerate(out)]


def _quantise(shots: list[Shot], fps: int, total_frames: int) -> list[Shot]:
    """Snap every boundary to a frame and make the parts sum to the whole.

    The clips are concatenated with a stream copy and the audio is laid against the SCENE
    duration, so a half-frame of rounding drift per shot would walk the voices out of sync
    by the end of a long scene.
    """
    out: list[Shot] = []
    cursor = 0
    for i, shot in enumerate(shots):
        end = total_frames if i == len(shots) - 1 else int(round(shot.end_s * fps))
        end = max(cursor + 1, min(total_frames, end))
        out.append(replace(shot, index=i, start_frame=cursor, end_frame=end,
                           start_s=cursor / fps, end_s=end / fps))
        cursor = end
        if cursor >= total_frames:
            break
    if out and out[-1].end_frame != total_frames:
        out[-1] = replace(out[-1], end_frame=total_frames, end_s=total_frames / fps)
    return out
