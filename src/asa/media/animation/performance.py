"""Making a generated still perform: jaw, blink, head and camera, driven by the voice.

This is the answer to the complaint that started it - "the character is static and the
voice comes from the background". The cinematic path had one still per scene and a slow
zoom, so nobody on screen was ever the person talking.

The image model gives us a photoreal head we cannot re-pose, and the machine has no GPU to
run a video model on. What it does have is the amplitude envelope the audio stage already
computes for lip-sync on the puppet path (`media/audio/envelope.py`), one value per frame,
per character. That signal is enough to drive a 2.5D performance:

  jaw     the band from below the eyes to the chin is scaled vertically about the level.
          Speaking shots are generated with the mouth already open mid-word, so the still
          is the MIDPOINT of the motion: a quiet frame compresses the band and the mouth
          closes, a loud frame stretches it and the jaw drops. Animating in both directions
          off a parted mouth is what stops it reading as a hinge.
  blink   a brief vertical squash of the eye band, on a deterministic schedule.
  head    a few pixels of feathered drift, more of it while the character is loud.
  camera  a slow push, pull or pan, plus a breath of handheld motion.

None of this is a substitute for an image-to-video model, and it is not pretending to be:
it is what runs for free on a 2017 laptop. The shot layer above it (`assemble/shotlist.py`)
is shared with the video-provider path, so a shot rendered here today can be rendered by a
video model tomorrow without re-cutting the episode.

Every effect is a pure function of (frame index, envelope, seed). Frames are therefore
independent and the pool below scales across cores, exactly as the puppet compositor does.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ...assemble.cinematic import move_for
from ...core.logging import get_logger
from .face import (EYE_BOTTOM, EYE_HALF_WIDTH, EYE_TOP, MOUTH_BOTTOM, MOUTH_HALF_WIDTH,
                   MOUTH_PEAK, MOUTH_TOP)

log = get_logger("performance")

# The still is generated mid-word, so this is where the jaw sits at rest in the source
# image. Levels above it open the mouth, below it close it. Only roughly true: the speaking
# clause in the prompt lands on some generations and not others, so some stills arrive with
# the mouth already shut and their motion is one-sided in practice.
NEUTRAL_LEVEL = 0.45
# Peak jaw travel, as a fraction of HEAD height per unit of level above or below rest, before
# the head box's confidence scales it down.
#
# Read the name literally: this moves a jaw, it does not shape lips. A vertical displacement
# of a region of a photograph cannot part a mouth or show teeth, whatever the numbers are,
# and calling the result lip-sync was the mistake this comment exists to stop being repeated.
# Real lip-sync comes from a model that was trained on faces - see media/video/replicate.py.
# What this is for is the free fallback: enough motion that a held still does not look
# frozen, confined tightly enough that it cannot wobble the scenery.
JAW_GAIN = 0.09
# Feather width as a fraction of head WIDTH, so a stretched band never shows a vertical
# edge. It follows the band's width rather than the head's height because the band is now a
# muzzle: keying it to height made the ramp wider than the region it was softening.
FEATHER = 0.08

BLINK_EVERY_S = 3.4
BLINK_FRAMES = 3
BLINK_SQUASH = 0.90


@dataclass
class ShotSpec:
    """Everything a worker needs to render one shot. Paths, not images."""

    image_path: str
    frames: int
    fps: int = 24
    size: tuple[int, int] = (1920, 1080)
    camera_move: str = "static"
    face: tuple[float, float, float, float] | None = None
    envelope: list[float] = field(default_factory=list)
    animate_face: bool = True
    blink: bool = True
    seed: int = 0
    final: bool = True
    # How far to trust `face`. Scales the jaw, because the cost of a confident warp on a
    # wrongly located box is the whole picture moving. See animation/face.py.
    face_confidence: float = 0.35
    # Below this, the box is not treated as a face at all. See `ShotRenderer.trusted`.
    min_face_confidence: float = 0.5


class ShotRenderer:
    def __init__(self, spec: ShotSpec):
        self.spec = spec
        self.out_w, self.out_h = spec.size
        dx, dy, dz = move_for(spec.camera_move)
        self.dx, self.dy, self.dz = dx, dy, dz
        # Headroom for the camera to travel inside. The still is upscaled ONCE here rather
        # than resampled from 1024px every frame: moving inside an image that was never
        # bigger than the output is what makes a slow push crawl.
        self.room = abs(dz) + max(abs(dx), abs(dy)) + 0.02
        self.work_w = int(round(self.out_w * (1.0 + self.room)))
        self.work_h = int(round(self.out_h * (1.0 + self.room)))
        with Image.open(spec.image_path) as im:
            self.work = im.convert("RGB").resize((self.work_w, self.work_h), Image.LANCZOS)
        self.resample = Image.BICUBIC if spec.final else Image.BILINEAR
        self._masks: dict = {}
        # A rectangle we do not trust enough to warp is not a face, and scaling the
        # amplitude down is not the same as declining to act on it.
        #
        # Measured, on a finished shot rather than a fixture: asked for a close-up of a fox,
        # FLUX returned a two-thirds figure seated at a desk with its muzzle in the upper
        # right. The `close_up` prior says the head fills the frame, so the muzzle band
        # landed on the character's lap and the desk behind it, and the "speech" the
        # renderer produced was a laptop and a chest sliding up and down in time with the
        # voice - 0.995 correlation with the envelope, on furniture. Confidence scaling did
        # not save it: at 0.35 the artefact is quieter, not absent, and a quieter wrong
        # answer is still a wrong answer.
        #
        # So the threshold is a floor, not a multiplier. Above it - a detector's sidecar,
        # which `face.locate_face` trusts at 1.0 - everything animates as before. Below it,
        # the shot gets its camera move and nothing else, which is the honest rendering of
        # "we do not know where this character's face is".
        #
        # It gates blink and head drift as well as the jaw, deliberately: all three read the
        # same rectangle, so gating only the jaw would leave the eye band squashing a
        # bookshelf on a schedule.
        self.trusted = (spec.face is not None
                        and spec.face_confidence >= spec.min_face_confidence)

    # -------------------------------------------------------------- signals

    def level(self, i: int) -> float:
        env = self.spec.envelope
        if not env:
            return 0.0
        return float(env[i]) if 0 <= i < len(env) else 0.0

    def _blinking(self, i: int) -> float:
        """0 when the eyes are open, 1 at the closed frame of a blink."""
        if not self.spec.blink:
            return 0.0
        period = max(1, int(round(BLINK_EVERY_S * self.spec.fps)))
        phase = (i + self.spec.seed * 7) % period
        if phase >= BLINK_FRAMES:
            return 0.0
        return math.sin(math.pi * (phase + 0.5) / BLINK_FRAMES)

    # -------------------------------------------------------------- geometry

    def _crop_box(self, u: float) -> tuple[float, float, float, float]:
        """The camera's window over the work image at progress `u` in 0..1."""
        if self.dz > 0:                         # push in
            z = 1.0 + abs(self.dz) * u
        elif self.dz < 0:                       # pull out
            z = 1.0 + abs(self.dz) * (1.0 - u)
        else:
            z = 1.0 + self.room * 0.5           # room to pan in both directions
        cw, ch = self.work_w / z, self.work_h / z
        cx = self.work_w / 2 + self.dx * self.work_w * (u - 0.5)
        cy = self.work_h / 2 + self.dy * self.work_h * (u - 0.5)
        # A breath of handheld motion. Even a locked-off shot of a photograph reads as a
        # stalled video without it.
        t = u * self.spec.frames / max(1, self.spec.fps)
        cx += math.sin(t * math.tau * 0.23 + self.spec.seed) * self.work_w * 0.0016
        cy += math.cos(t * math.tau * 0.19 + self.spec.seed) * self.work_h * 0.0014
        cx = min(max(cx, cw / 2), self.work_w - cw / 2)
        cy = min(max(cy, ch / 2), self.work_h - ch / 2)
        return (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)

    def _face_px(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
        """The head box in OUTPUT pixels, having been through the same camera as the frame.

        Animating after the crop rather than before it means the expensive resize happens
        once and the warp touches a few hundred rows, not the whole frame.
        """
        if not self.spec.face:
            return None
        fx, fy, fw, fh = self.spec.face
        sx = self.out_w / (box[2] - box[0])
        sy = self.out_h / (box[3] - box[1])
        x = (fx * self.work_w - box[0]) * sx
        y = (fy * self.work_h - box[1]) * sy
        return (int(round(x)), int(round(y)),
                max(4, int(round(fw * self.work_w * sx))),
                max(4, int(round(fh * self.work_h * sy))))

    # -------------------------------------------------------------- drawing

    def _band_mask(self, size: tuple[int, int], feather: int) -> Image.Image:
        """Opaque everywhere except a horizontal ramp at each side.

        Vertically it must be FULLY opaque. A mask that fades near the top and bottom rows
        - which is what blurring a rectangle gives you - lets the un-warped original show
        through underneath its own displaced copy, so a moving feature appears twice: once
        where it was and once where it moved to. The band's top and bottom edges do not
        need feathering anyway, because the displacement there is zero by construction.
        """
        key = (size, feather)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        w, h = size
        pad = max(1, min(feather, w // 2 - 1)) if w > 4 else 1
        x = np.arange(w, dtype=np.float32)
        ramp = np.clip(np.minimum(x, w - 1 - x) / max(1.0, float(pad)), 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)                # smoothstep
        row = (ramp * 255.0).astype(np.uint8)
        mask = Image.fromarray(np.broadcast_to(row, (h, w)).copy(), mode="L")
        if len(self._masks) > 24:
            self._masks.clear()
        self._masks[key] = mask
        return mask

    def _stretch(self, frame: Image.Image, source: Image.Image, left: int, right: int,
                 top: float, bottom: float, d_top: float, d_bottom: float,
                 feather: int) -> None:
        """Displace a band's content vertically, in place: `d_top` at its top edge,
        `d_bottom` at its bottom, interpolated linearly between the two.

        Everything here is one primitive. A band redrawn from a source slice offset by
        those two amounts IS a linear vertical displacement field, so a jaw is two bands -
        one that carries the chin down, and one below it that absorbs the motion again so
        the chest does not slide with it - and a blink is one band with a negative
        displacement at its lower edge.

        Pixels are read from `source` and written to `frame`, and those must be different
        images. Reading from the frame it is writing into is what left a seam across the
        face at MOUTH_PEAK: the lower band sampled rows the upper band had already carried
        down, so the two disagreed about where the join was and warped it twice.

        The band is clipped by solving for the rows whose SOURCE is still on the image,
        rather than by clamping the source rectangle. Clamping silently rescales the band,
        which breaks the displacement it shares with its neighbour - the same seam by a
        different route.
        """
        left, right = max(0, left), min(source.width, right)
        span = bottom - top
        if span < 4 or right - left < 4 or (abs(d_top) < 0.4 and abs(d_bottom) < 0.4):
            return
        # src(y) = a*y + b, the affine map this band applies.
        a = 1.0 + (d_bottom - d_top) / span
        if a <= 0.01:
            return
        b = d_top - (d_bottom - d_top) * top / span
        lo = max(top, 0.0, -b / a)                       # src(y) >= 0
        hi = min(bottom, float(frame.height), (frame.height - b) / a)
        y0, y1 = int(math.floor(lo)), int(math.ceil(hi))
        h = y1 - y0
        if h < 4:
            return
        src_top = max(0.0, a * y0 + b)
        src_bottom = min(float(frame.height), a * y1 + b)
        if src_bottom - src_top < 2:
            return
        w = right - left
        band = source.resize((w, h), Image.BILINEAR,
                             box=(left, src_top, right, src_bottom))
        frame.paste(band, (left, y0), self._band_mask((w, h), max(2, feather)))

    def _drift_head(self, frame: Image.Image, face: tuple[int, int, int, int],
                    t: float, level: float) -> None:
        """A few pixels of feathered head motion. The copy sits over the original, so the
        edges blend rather than tearing - at this amplitude it reads as motion, not ghosting.
        """
        fx, fy, fw, fh = face
        # A close-up head IS the frame, and sliding a frame-sized ellipse over its own
        # image moves the middle of the picture while the corners stay put - a visible
        # oval seam. At that size the camera breath is the head motion.
        if fw * fh > 0.66 * frame.width * frame.height:
            return
        amp = 1.0 + 3.0 * level
        ox = int(round(math.sin(t * math.tau * 0.8 + self.spec.seed) * amp))
        oy = int(round(math.sin(t * math.tau * 1.3 + self.spec.seed * 2) * amp * 0.7
                       - level * 2.0))
        if ox == 0 and oy == 0:
            return
        pad = int(fw * 0.18)
        left, top = max(0, fx - pad), max(0, fy - pad)
        right = min(frame.width, fx + fw + pad)
        bottom = min(frame.height, fy + fh + pad)
        if right - left < 8 or bottom - top < 8:
            return
        patch = frame.crop((left, top, right, bottom))
        mask = self._ellipse_mask((right - left, bottom - top))
        frame.paste(patch, (left + ox, top + oy), mask)

    def _ellipse_mask(self, size: tuple[int, int]) -> Image.Image:
        key = ("ellipse", size)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        w, h = size
        mask = Image.new("L", (w, h), 0)
        inset = max(2, int(min(w, h) * 0.12))
        ImageDraw.Draw(mask).ellipse((inset, inset, w - inset, h - inset), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(3.0, min(w, h) * 0.06)))
        self._masks[key] = mask
        return mask

    # -------------------------------------------------------------- frames

    def render_frame(self, i: int) -> Image.Image:
        u = i / max(1, self.spec.frames - 1) if self.spec.frames > 1 else 0.0
        box = self._crop_box(u)
        frame = self.work.resize((self.out_w, self.out_h), self.resample, box=box)

        if not self.spec.animate_face or not self.trusted:
            return frame
        face = self._face_px(box)
        if face is None:
            return frame

        fx, fy, fw, fh = face
        if fw < 24 or fh < 24 or fx > self.out_w or fy > self.out_h:
            return frame
        t = i / max(1, self.spec.fps)
        level = self.level(i)
        feather = max(2, int(fw * FEATHER))

        blink = self._blinking(i)
        if not self.spec.envelope and blink <= 0.001:
            self._drift_head(frame, face, t, level)
            return frame

        # One untouched copy per frame: every band below reads from it and writes to the
        # frame, so two bands that share an edge cannot warp each other's pixels.
        source = frame.copy()
        cx = fx + fw // 2
        if blink > 0.001:
            eye_top = fy + int(fh * EYE_TOP)
            eye_bottom = fy + int(fh * EYE_BOTTOM)
            squash = (eye_bottom - eye_top) * (1.0 - BLINK_SQUASH) * blink
            half = int(fw * EYE_HALF_WIDTH)
            self._stretch(frame, source, cx - half, cx + half,
                          eye_top, eye_bottom, 0.0, -squash, max(2, int(fh * 0.04)))

        if self.spec.envelope:
            # Bounded on all four sides and wholly inside the head box. The width is the
            # muzzle's, not the head's: a band spanning the full frame moves the background
            # with the jaw, which is what made a talking close-up read as a flickering
            # photograph rather than a character speaking.
            travel = ((level - NEUTRAL_LEVEL) * JAW_GAIN * fh
                      * max(0.0, min(1.0, self.spec.face_confidence)))
            half = int(fw * MOUTH_HALF_WIDTH)
            left, right = cx - half, cx + half
            top = fy + int(fh * MOUTH_TOP)
            peak = fy + int(fh * MOUTH_PEAK)
            bottom = fy + int(fh * MOUTH_BOTTOM)
            self._stretch(frame, source, left, right, top, peak, 0.0, -travel, feather)
            self._stretch(frame, source, left, right, peak, bottom, -travel, 0.0, feather)
        self._drift_head(frame, face, t, level)
        return frame


# ------------------------------------------------------------------ rendering

_RENDERER: ShotRenderer | None = None


def _init_worker(spec: ShotSpec) -> None:
    global _RENDERER
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _RENDERER = ShotRenderer(spec)


def _render_one(i: int) -> bytes:
    assert _RENDERER is not None
    return _RENDERER.render_frame(i).tobytes()


def render_shot(spec: ShotSpec, out_path: Path, workers: int | None = None,
                crf: int = 20, preset: str = "veryfast", chunksize: int = 8) -> dict:
    """Render one shot to an mp4, raw frames piped straight into ffmpeg.

    Same contract as the puppet renderer: no intermediate PNGs, one encoder invocation, and
    identical encoder settings across shots so the concat afterwards is a stream copy.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, spec.frames)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{spec.size[0]}x{spec.size[1]}", "-r", str(spec.fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-g", str(spec.fps * 2), "-movflags", "+faststart",
        str(out_path),
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        if workers <= 1:
            r = ShotRenderer(spec)
            for i in range(n):
                proc.stdin.write(r.render_frame(i).tobytes())
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(spec,)) as pool:
                for buf in pool.imap(_render_one, range(n), chunksize=chunksize):
                    proc.stdin.write(buf)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}")
    el = time.time() - t0
    return {"frames": n, "workers": workers, "seconds": round(el, 2),
            "ms_per_frame": round(el / n * 1000, 1),
            "path": str(out_path), "bytes": out_path.stat().st_size}
