"""Per-frame puppet compositor.

Characters are transformed directly into screen space rather than drawn into a world-size
plane and cropped - that keeps every resize small and is what makes CPU rendering viable.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from ...characters.rig import Rig
from .camera import Camera, Viewport

RESAMPLE_DRAFT = Image.BILINEAR
RESAMPLE_FINAL = Image.LANCZOS
# Backgrounds are resized every frame and are the dominant cost. BICUBIC is ~2-3x faster
# than LANCZOS and the difference is invisible on a moving camera over flat-vector art.
RESAMPLE_BG = Image.BICUBIC


@dataclass
class BackgroundLayer:
    image: Image.Image
    parallax: float = 1.0


@dataclass
class SpeechSpan:
    start: float
    end: float

    def active(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass
class CharacterInstance:
    """A rig placed in the world, plus its per-shot performance."""
    rig: Rig
    base_dir: Path
    x: float = 0.5                 # world position, normalised
    y: float = 0.92                # where the FEET sit, normalised
    scale: float = 1.0             # 1.0 => rig canvas height == 62% of world height
    facing: str = "right"
    gesture: str = "idle"
    speech: list[SpeechSpan] = field(default_factory=list)
    blink_seed: int = 0
    # Real per-frame RMS from the synthesised audio. When present it drives the mouth and
    # the synthetic cycle below is unused (docs/02 §8.1 Tier A).
    envelope: list[float] = field(default_factory=list)
    envelope_fps: int = 24
    # Scene lighting integration. Without this the puppet reads as pasted on: flat studio
    # colours over a warm golden plate. Applied once per layer at load, then cached.
    grade_rgb: tuple[int, int, int] = (255, 255, 255)
    grade_strength: float = 0.0
    _cache: dict = field(default_factory=dict, repr=False)

    def layer(self, key: str) -> Image.Image:
        if key not in self._cache:
            img = Image.open(self.base_dir / self.rig.layers[key].file).convert("RGBA")
            self._cache[key] = self._grade(img)
        return self._cache[key]

    def _grade(self, img: Image.Image) -> Image.Image:
        """Multiply-blend the layer toward the scene's light colour, alpha untouched."""
        if self.grade_strength <= 0.001:
            return img
        k = min(1.0, max(0.0, self.grade_strength))
        alpha = img.getchannel("A")
        rgb = img.convert("RGB")
        tint = Image.new("RGB", img.size, self.grade_rgb)
        lit = ImageChops.multiply(rgb, tint)
        out = Image.blend(rgb, lit, k)
        out = out.convert("RGBA")
        out.putalpha(alpha)
        return out

    # -------- performance -------------------------------------------------

    def is_speaking(self, t: float) -> bool:
        if self.envelope:
            return self.level(t) >= 0.12
        return any(s.active(t) for s in self.speech)

    def level(self, t: float) -> float:
        if not self.envelope:
            return 0.0
        i = int(round(t * self.envelope_fps))
        return self.envelope[i] if 0 <= i < len(self.envelope) else 0.0

    def viseme(self, t: float) -> str:
        if self.envelope:
            from ..audio.envelope import envelope_to_viseme
            # deterministic: previous shape derives from the previous frame, not mutable state
            prev_i = max(0, int(round(t * self.envelope_fps)) - 1)
            prev_lvl = self.envelope[prev_i] if prev_i < len(self.envelope) else 0.0
            prev = envelope_to_viseme(prev_lvl, "rest")
            return envelope_to_viseme(self.level(t), prev)
        if not any(s.active(t) for s in self.speech):
            return "rest"
        # fallback when there is no audio yet: ~8 shapes/sec, deterministic
        step = int(t * 8.0)
        rng = random.Random(self.blink_seed * 7919 + step)
        return rng.choice(["A", "E", "I", "O", "U", "A", "E", "M"])

    def eye_state(self, t: float) -> str:
        """Blinks on a deterministic Poisson-ish schedule, ~1 per 4s."""
        rng = random.Random(self.blink_seed)
        clock, blinks = 0.0, []
        while clock < 600:
            clock += rng.uniform(2.2, 5.6)
            blinks.append(clock)
        for b in blinks:
            d = t - b
            if 0 <= d < 0.06 or 0.10 <= d < 0.16:
                return "half"
            if 0.06 <= d < 0.10:
                return "closed"
        return "open"

    def body_offset(self, t: float) -> tuple[float, float]:
        g = self.gesture
        if g in ("idle", "talk", "sit"):
            return (0.0, math.sin(t * math.tau * 0.5) * 3.0)
        if g == "walk_cycle":
            return (0.0, abs(math.sin(t * math.tau * 1.6)) * -7.0)
        if g == "run_cycle":
            return (0.0, abs(math.sin(t * math.tau * 2.6)) * -13.0)
        if g == "react_shock":
            return (0.0, -min(18.0, t * 90.0))
        return (0.0, 0.0)

    def head_offset(self, t: float) -> tuple[float, float]:
        if self.gesture == "talk" or self.is_speaking(t):
            return (math.sin(t * math.tau * 1.9) * 2.4, math.sin(t * math.tau * 3.1) * 2.0)
        if self.gesture == "react_sad":
            return (0.0, 6.0)
        return (0.0, math.sin(t * math.tau * 0.37) * 1.4)


class SceneRenderer:
    def __init__(self, world: tuple[int, int], frame: tuple[int, int],
                 background: list[BackgroundLayer], characters: list[CharacterInstance],
                 camera: Camera, duration: float, fps: int = 24, final: bool = True):
        self.world_w, self.world_h = world
        self.frame_w, self.frame_h = frame
        self.background = background
        self.characters = characters
        self.camera = camera
        self.duration = duration
        self.fps = fps
        self.resample = RESAMPLE_FINAL if final else RESAMPLE_DRAFT
        self._bg_cache: dict = {}

    # -------- geometry ----------------------------------------------------

    def _layer_viewport(self, vp: Viewport, parallax: float) -> tuple[float, float, float, float]:
        """Nearer layers track the camera fully; far layers lag, creating depth."""
        cx = self.world_w / 2 + (vp.cx - self.world_w / 2) * parallax
        cy = self.world_h / 2 + (vp.cy - self.world_h / 2) * parallax
        half_w = vp.w / 2
        half_h = vp.h / 2
        cx = min(max(cx, half_w), self.world_w - half_w)
        cy = min(max(cy, half_h), self.world_h - half_h)
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def render_frame(self, i: int) -> Image.Image:
        t = i / self.fps
        tn = min(1.0, t / self.duration) if self.duration else 0.0
        vp = self.camera.at(tn)
        frame = Image.new("RGBA", (self.frame_w, self.frame_h), (0, 0, 0, 255))

        for bl in self.background:
            box = self._layer_viewport(vp, bl.parallax)
            key = (id(bl), tuple(round(v, 1) for v in box))
            img = self._bg_cache.get(key)
            if img is None:
                img = bl.image.resize((self.frame_w, self.frame_h), RESAMPLE_BG, box=box)
                if len(self._bg_cache) > 8:
                    self._bg_cache.clear()
                self._bg_cache[key] = img
            frame.alpha_composite(img)

        px_per_world = self.frame_w / vp.w
        for ch in self.characters:
            self._draw_shadow(frame, ch, t, vp, px_per_world)
        for ch in self.characters:
            self._draw_character(frame, ch, t, vp, px_per_world)
        return frame

    def _draw_shadow(self, frame: Image.Image, ch: CharacterInstance,
                     t: float, vp: Viewport, ppw: float) -> None:
        """Soft contact ellipse. Shrinks as the character lifts off the ground."""
        rig_h = ch.rig.canvas[1]
        world_px_per_rig_px = (self.world_h * 0.62 * ch.scale) / rig_h
        s = world_px_per_rig_px * ppw
        _, bob_y = ch.body_offset(t)
        lift = max(0.0, -bob_y) / 14.0
        rx = int(160 * s * (1.0 - 0.22 * lift))
        ry = int(34 * s * (1.0 - 0.30 * lift))
        if rx < 2 or ry < 1:
            return
        cx = int((ch.x * self.world_w - vp.cx) * ppw + self.frame_w / 2)
        cy = int((ch.y * self.world_h - vp.cy) * ppw + self.frame_h / 2)
        pad = max(rx, ry) // 2 + 4
        sh = Image.new("RGBA", (rx * 2 + pad * 2, ry * 2 + pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse((pad, pad, pad + rx * 2, pad + ry * 2),
                                   fill=(28, 18, 10, int(110 * (1.0 - 0.35 * lift))))
        sh = sh.filter(ImageFilter.GaussianBlur(max(2.0, ry * 0.42)))
        frame.alpha_composite(sh, (cx - rx - pad, cy - ry - pad))

    def _draw_character(self, frame: Image.Image, ch: CharacterInstance,
                        t: float, vp: Viewport, ppw: float) -> None:
        rig = ch.rig
        rig_w, rig_h = rig.canvas
        # rig canvas height maps to 62% of world height at scale 1.0
        world_px_per_rig_px = (self.world_h * 0.62 * ch.scale) / rig_h
        s = world_px_per_rig_px * ppw
        if s <= 0.01:
            return

        feet_world_x = ch.x * self.world_w
        feet_world_y = ch.y * self.world_h
        bob_x, bob_y = ch.body_offset(t)
        hx, hy = ch.head_offset(t)

        # screen position of the rig's origin (canvas top-left)
        origin_x = (feet_world_x - vp.cx) * ppw + self.frame_w / 2 - (rig_w / 2) * s
        origin_y = (feet_world_y - vp.cy) * ppw + self.frame_h / 2 - rig.ground_y * s

        head_keys = {"head", "eyes", "mouth"}
        for key in rig.z_order:
            if key == "eyes":
                layer_key = rig.eyes[ch.eye_state(t)]
            elif key == "mouth":
                layer_key = rig.visemes[ch.viseme(t)]
            else:
                layer_key = key
            if layer_key not in rig.layers:
                continue
            meta = rig.layers[layer_key]
            img = ch.layer(layer_key)
            w = max(1, int(round(meta.size[0] * s)))
            h = max(1, int(round(meta.size[1] * s)))
            img = img.resize((w, h), self.resample)
            if ch.facing == "left":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            ox, oy = meta.offset
            if ch.facing == "left":
                ox = rig_w - ox - meta.size[0]
            dx = bob_x
            dy = bob_y
            if key in head_keys:
                dx += hx
                dy += hy
            px = int(round(origin_x + (ox + dx) * s))
            py = int(round(origin_y + (oy + dy) * s))
            if px + w < 0 or py + h < 0 or px > self.frame_w or py > self.frame_h:
                continue
            frame.alpha_composite(img, (px, py))
