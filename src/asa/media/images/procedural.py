"""Terminal image fallback: a background drawn from primitives, no network, no quota.

This is not a placeholder. When every API is exhausted the choice is between shipping a
composed, on-palette vector-ish plate or failing the render, and a stylised flat plate cut
into parallax bands is perfectly watchable storybook art. The seed comes from the prompt, so
the same location always draws the same plate.
"""
from __future__ import annotations

import colorsys
import hashlib
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .base import GeneratedImage, prompt_key

# Sky colour is a function of TIME, terrain colour is a function of PLACE. Averaging both
# into one hue - which the first version did - produces green skies for "forest at dusk",
# because the forest hue drags the sky with it.
TIME_SKY: dict[str, tuple[float, float, float]] = {
    "night":   (0.63, 0.52, 0.30), "midnight": (0.64, 0.55, 0.24), "moon": (0.63, 0.44, 0.34),
    "dusk":    (0.94, 0.42, 0.66), "evening": (0.95, 0.40, 0.64), "sunset": (0.03, 0.55, 0.86),
    "dawn":    (0.06, 0.38, 0.92), "sunrise": (0.06, 0.42, 0.94),
    "morning": (0.56, 0.32, 0.96), "noon": (0.57, 0.36, 0.98), "afternoon": (0.55, 0.30, 0.96),
    "storm":   (0.62, 0.20, 0.44), "rain": (0.58, 0.22, 0.54), "overcast": (0.60, 0.10, 0.72),
    "fog":     (0.58, 0.08, 0.80), "snow": (0.58, 0.12, 0.92), "winter": (0.58, 0.14, 0.88),
}
DEFAULT_SKY = (0.56, 0.32, 0.94)

PLACE_GROUND: dict[str, tuple[float, float, float]] = {
    "forest": (0.30, 0.50, 0.44), "wood": (0.29, 0.48, 0.46), "pine": (0.36, 0.46, 0.38),
    "jungle": (0.31, 0.58, 0.42), "meadow": (0.23, 0.52, 0.62), "field": (0.20, 0.50, 0.66),
    "grass":  (0.24, 0.52, 0.60), "hill": (0.25, 0.44, 0.58), "valley": (0.26, 0.46, 0.56),
    "mountain": (0.62, 0.20, 0.52), "peak": (0.62, 0.16, 0.58), "cliff": (0.07, 0.28, 0.46),
    "desert": (0.10, 0.46, 0.76), "dune": (0.10, 0.42, 0.80), "canyon": (0.04, 0.48, 0.58),
    "river": (0.52, 0.42, 0.56), "lake": (0.53, 0.40, 0.58), "sea": (0.55, 0.48, 0.54),
    "shore": (0.12, 0.30, 0.80), "beach": (0.12, 0.28, 0.84),
    "village": (0.08, 0.40, 0.58), "town": (0.07, 0.36, 0.56), "market": (0.05, 0.46, 0.64),
    "farm":   (0.16, 0.46, 0.60), "orchard": (0.22, 0.48, 0.58), "garden": (0.26, 0.50, 0.62),
    "snow": (0.58, 0.06, 0.94), "ice": (0.55, 0.14, 0.90),
}
DEFAULT_GROUND = (0.24, 0.44, 0.56)

# Interior walls take their hue from the room's purpose, not from the sky.
ROOM_HUE: dict[str, float] = {
    "bakery": 0.07, "kitchen": 0.08, "workshop": 0.06, "barn": 0.05, "cellar": 0.63,
    "cave": 0.66, "library": 0.05, "attic": 0.07, "shop": 0.09, "inn": 0.06,
}
INTERIORS = ("bakery", "kitchen", "attic", "workshop", "cave", "shop", "room", "indoor",
             "library", "cellar", "barn", "inside", "inn", "hall")


def _hex(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return int(r * 255), int(g * 255), int(b * 255)


def _resolve(prompt: str, table: dict[str, tuple[float, float, float]],
             default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Longest keyword wins, so "moonlit mountain" picks mountain over the substring "moon"
    in the ground table and moon in the sky table - each table only sees its own words."""
    hits = [(len(k), v) for k, v in table.items() if k in prompt]
    if not hits:
        return default
    hits.sort(key=lambda kv: -kv[0])
    top = [v for ln, v in hits if ln == hits[0][0]]
    n = len(top)
    return (sum(h for h, _, _ in top) / n, sum(s for _, s, _ in top) / n,
            sum(v for _, _, v in top) / n)


def _sky(w: int, h: int, hue: float, sat: float, val: float) -> Image.Image:
    img = Image.new("RGB", (w, h))
    dr = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        dr.line([(0, y), (w, y)],
                fill=_hex(hue + 0.04 * t, sat * (0.55 + 0.5 * t), val * (1.0 - 0.30 * t)))
    return img


def _ridge(dr: ImageDraw.ImageDraw, w: int, h: int, base_y: float, amp: float,
           freq: float, colour, rng: random.Random, spiky: bool) -> None:
    pts = [(0, h)]
    phase = rng.uniform(0, math.tau)
    step = max(4, w // 220)
    for x in range(0, w + step, step):
        t = x / w
        y = base_y * h
        y -= math.sin(t * math.tau * freq + phase) * amp * h
        y -= math.sin(t * math.tau * freq * 2.7 + phase * 1.7) * amp * h * 0.35
        if spiky:
            y -= abs(math.sin(t * math.tau * freq * 5.3 + phase)) * amp * h * 0.5
        pts.append((x, y))
    pts.append((w, h))
    dr.polygon(pts, fill=colour)


def _interior(img: Image.Image, w: int, h: int, hue: float, rng: random.Random) -> None:
    dr = ImageDraw.Draw(img)
    wall = _hex(hue + 0.02, 0.34, 0.62)
    floor = _hex(hue + 0.01, 0.42, 0.40)
    dr.rectangle([0, 0, w, int(h * 0.74)], fill=wall)
    dr.rectangle([0, int(h * 0.74), w, h], fill=floor)
    # plank lines give the floor a readable perspective without any 3D
    for i in range(1, 9):
        y = int(h * (0.74 + 0.26 * (i / 9) ** 1.7))
        dr.line([(0, y), (w, y)], fill=_hex(hue, 0.46, 0.32), width=max(1, h // 380))
    # windows: the single strongest cue that a flat rectangle is a room
    for k in range(2):
        x0 = int(w * (0.14 + 0.46 * k))
        y0 = int(h * 0.16)
        ww_, hh_ = int(w * 0.20), int(h * 0.30)
        dr.rectangle([x0, y0, x0 + ww_, y0 + hh_], fill=_hex(hue + 0.5, 0.22, 0.94),
                     outline=_hex(hue, 0.5, 0.30), width=max(2, h // 240))
        dr.line([(x0 + ww_ // 2, y0), (x0 + ww_ // 2, y0 + hh_)],
                fill=_hex(hue, 0.5, 0.30), width=max(2, h // 300))
        dr.line([(x0, y0 + hh_ // 2), (x0 + ww_, y0 + hh_ // 2)],
                fill=_hex(hue, 0.5, 0.30), width=max(2, h // 300))
    # shelving along the back wall
    for i in range(3):
        y = int(h * (0.50 + i * 0.075))
        dr.rectangle([int(w * 0.60), y, int(w * 0.96), y + max(3, h // 150)],
                     fill=_hex(hue, 0.52, 0.34))
        for j in range(6):
            jx = int(w * (0.62 + j * 0.055))
            jh = rng.randint(int(h * 0.02), int(h * 0.045))
            dr.rectangle([jx, y - jh, jx + int(w * 0.035), y],
                         fill=_hex(hue + rng.uniform(-0.08, 0.08), 0.55, 0.78))


def _exterior(img: Image.Image, w: int, h: int, sky: tuple[float, float, float],
              ground: tuple[float, float, float], rng: random.Random, prompt: str) -> None:
    dr = ImageDraw.Draw(img)
    hue, sat, val = sky
    ghue, gsat, gval = ground
    spiky = any(k in prompt for k in ("mountain", "peak", "cliff", "pine", "forest"))
    if any(k in prompt for k in ("night", "moon", "star")):
        for _ in range(90):
            sx, sy = rng.randrange(w), rng.randrange(int(h * 0.55))
            r = rng.choice([1, 1, 1, 2])
            dr.ellipse([sx, sy, sx + r, sy + r], fill=(255, 255, 240))
        mx, my = int(w * 0.78), int(h * 0.18)
        mr = int(h * 0.07)
        dr.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(248, 245, 226))
    else:
        sx, sy = int(w * 0.76), int(h * 0.20)
        sr = int(h * 0.085)
        dr.ellipse([sx - sr, sy - sr, sx + sr, sy + sr],
                   fill=_hex(0.13, 0.22, 1.0))          # the sun is warm white, always
    # three receding ridges, darkening forward: this IS the depth cue
    # Ridges take the GROUND hue but the sky's value, so terrain at night is dark green
    # rather than daylight green pasted onto a night sky.
    tint = min(1.0, val + 0.18)
    for i, (by, amp, freq, dv) in enumerate((
            (0.62, 0.055, 1.6, 0.92), (0.76, 0.045, 2.4, 0.70), (0.90, 0.030, 3.4, 0.50))):
        _ridge(dr, w, h, by, amp, freq,
               _hex(ghue + 0.015 * i, gsat * (0.85 + 0.12 * i), gval * dv * tint),
               rng, spiky and i < 2)


class ProceduralImages:
    name = "procedural"

    @property
    def available(self) -> bool:
        return True

    def generate(self, prompt: str, out_path: Path, size: tuple[int, int],
                 negative: str = "", seed: int | None = None) -> GeneratedImage:
        w, h = size
        key = prompt_key(prompt, size, negative)
        if seed is None:
            seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        p = prompt.lower()
        sky = _resolve(p, TIME_SKY, DEFAULT_SKY)
        ground = _resolve(p, PLACE_GROUND, DEFAULT_GROUND)

        if any(k in p for k in INTERIORS):
            room = next((v for k, v in ROOM_HUE.items() if k in p), 0.07)
            img = Image.new("RGB", (w, h), _hex(room, 0.3, 0.6))
            _interior(img, w, h, room, rng)
            hue, sat, val = room, 0.34, 0.62
        else:
            img = _sky(w, h, *sky)
            _exterior(img, w, h, sky, ground, rng, p)
            hue, sat, val = sky

        # A gentle blur keeps the bands from reading as hard vector edges, and a vignette
        # focuses the eye where characters stand.
        img = img.filter(ImageFilter.GaussianBlur(max(0.6, h / 900)))
        vig = Image.new("L", (w, h), 0)
        ImageDraw.Draw(vig).ellipse(
            [-w * 0.22, -h * 0.30, w * 1.22, h * 1.34], fill=255)
        vig = vig.filter(ImageFilter.GaussianBlur(h / 12))
        img = Image.composite(img, Image.blend(img, Image.new("RGB", (w, h), (0, 0, 0)), 0.30),
                              vig)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return GeneratedImage(path=out_path, provider=self.name,
                              model_id="asa.procedural.v1", prompt_sha=key, seed=seed,
                              meta={"mood": [round(hue, 3), round(sat, 3), round(val, 3)]})
