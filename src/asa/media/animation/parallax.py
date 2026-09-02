"""Turn one flat background plate into a multiplane stack.

A true depth model needs a depth estimator (and a GPU). For flat-vector storybook art a
banded multiplane split is visually sufficient and costs nothing: the same plate is drawn
three times with different vertical masks and scales, then moved at different rates.
"""
from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

from .compositor import BackgroundLayer


def _band_mask(size: tuple[int, int], top: float, feather: float) -> Image.Image:
    """Alpha ramp: transparent above `top`, opaque below, feathered across `feather`."""
    w, h = size
    mask = Image.new("L", (1, h), 0)
    px = mask.load()
    t0 = top * h
    t1 = min(h, (top + feather) * h)
    for y in range(h):
        if y <= t0:
            px[0, y] = 0
        elif y >= t1:
            px[0, y] = 255
        else:
            px[0, y] = int(255 * (y - t0) / max(1e-6, t1 - t0))
    return mask.resize((w, h))


def _fit(plate: Image.Image, world: tuple[int, int], scale: float) -> Image.Image:
    """Cover `world` at `scale`, centre-cropped."""
    ww, wh = world
    tw, th = int(ww * scale), int(wh * scale)
    src_ratio = plate.width / plate.height
    dst_ratio = tw / th
    if src_ratio > dst_ratio:
        nh = th
        nw = int(th * src_ratio)
    else:
        nw = tw
        nh = int(tw / src_ratio)
    img = plate.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    img = img.crop((left, top, left + tw, top + th))
    return img.resize((ww, wh), Image.LANCZOS)


# The single source of truth for the plane rates. parallel.py reads pre-baked planes from
# disk and must attach exactly these, or a cached scene drifts differently from a fresh one.
PARALLAX_FACTORS = (0.30, 0.62, 1.00)
PLATE_NAMES = ("far.png", "mid.png", "near.png")


def multiplane(plate: Image.Image, world: tuple[int, int]) -> list[BackgroundLayer]:
    plate = plate.convert("RGB")

    far = _fit(plate, world, 1.00)
    far = far.filter(ImageFilter.GaussianBlur(2.2))
    far = ImageEnhance.Color(far).enhance(0.82)
    far = ImageEnhance.Brightness(far).enhance(1.06)
    far_l = far.convert("RGBA")

    mid = _fit(plate, world, 1.07).convert("RGBA")
    mid.putalpha(_band_mask(world, 0.30, 0.18))

    near = _fit(plate, world, 1.16)
    near = ImageEnhance.Color(near).enhance(1.06)
    near = near.convert("RGBA")
    near.putalpha(_band_mask(world, 0.62, 0.12))

    return [BackgroundLayer(img, parallax=f) for img, f in
            zip((far_l, mid, near), PARALLAX_FACTORS)]
