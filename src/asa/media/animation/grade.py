"""Derive a scene's light colour from its background plate.

Sampling the plate means the character is lit by whatever the image model produced, so the
two stay consistent even when a plate is regenerated with a different palette.
"""
from __future__ import annotations

from PIL import Image


def light_from_plate(plate: Image.Image, strength: float = 0.35,
                     sample_box: tuple[float, float, float, float] = (0.15, 0.35, 0.85, 0.95),
                     ) -> tuple[tuple[int, int, int], float]:
    """Average the plate's mid/lower region (where characters stand) and lift it toward
    white so the multiply darkens rather than recolours."""
    w, h = plate.size
    x0, y0, x1, y1 = sample_box
    region = plate.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    r, g, b = region.resize((1, 1), Image.BOX).getpixel((0, 0))
    peak = max(r, g, b) or 1
    # normalise to a tint (preserve hue, drop overall darkness) then pull toward neutral
    tint = tuple(int(round(255 * (c / peak) * 0.72 + 255 * 0.28)) for c in (r, g, b))
    return tint, strength
