"""Thumbnail generation from assets we already own.

No extra image credits are spent here: a thumbnail is the episode's own background plate,
its own puppet at a chosen expression, and type. That also guarantees the thumbnail cannot
promise something the video does not deliver, which is both a policy requirement and the
single fastest way to destroy a channel's click-through rate.

Scoring is heuristic and deliberately conservative. It ranks variants against each other;
it does not pretend to predict CTR.
"""
from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from ..characters.rig import Rig
from ..core.db import jdump, tx
from ..core.logging import get_logger

log = get_logger("thumbnail")

SIZE = (1280, 720)
SAFE = 0.055                       # keep type this far from every edge
MAX_BYTES = 2 * 1024 * 1024        # YouTube's hard limit for custom thumbnails

# (character x, character scale, preferred vertical band for the text)
# Horizontal text position is NOT fixed: it is chosen after the character is composited,
# on whichever side actually has room. A fixed anchor is how you get a title across a
# character's face the one time the puppet is wider than you assumed.
LAYOUTS = {
    "right_hero":  (0.74, 1.04, 0.28),
    "left_hero":   (0.26, 1.04, 0.28),
    "centre_low":  (0.50, 0.80, 0.10),
    "corner_up":   (0.80, 0.96, 0.56),
}
EXPRESSIONS = {
    "surprised": ("open", "O"), "happy": ("open", "E"), "determined": ("half", "rest"),
    "curious": ("open", "rest"), "scared": ("open", "A"), "sad": ("half", "rest"),
}


@dataclass
class ThumbVariant:
    variant: int
    path: Path
    text: str
    expression: str
    layout: str
    score: float
    scores: dict


def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _fit_text(draw, text: str, box_w: int, box_h: int, start: int = 132) -> tuple:
    """Shrink until the wrapped block fits. Type that overflows is worse than type that
    is slightly small, and YouTube crops the right edge on some surfaces."""
    words = text.split()
    for size in range(start, 38, -4):
        f = _font(size)
        lines, cur = [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if draw.textlength(cand, font=f) <= box_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lh = int(size * 1.12)
        if len(lines) <= 3 and lh * len(lines) <= box_h and all(
                draw.textlength(ln, font=f) <= box_w for ln in lines):
            return f, lines, lh
    f = _font(40)
    return f, words[:3], 46


def _contrast_colour(img: Image.Image, box: tuple[int, int, int, int]) -> tuple:
    """Pick text + outline colours from the mean luminance behind the text box."""
    crop = img.crop(box).convert("RGB").resize((16, 16))
    px = list(crop.getdata())
    lum = sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in px) / len(px)
    if lum > 140:
        return (18, 16, 14), (255, 252, 244)
    return (255, 250, 238), (16, 14, 12)


def _character_cutout(puppet_dir: Path, expression: str) -> Image.Image | None:
    rig_path = puppet_dir / "rig.json"
    if not rig_path.exists():
        return None
    rig = Rig.load(rig_path)
    eye, mouth = EXPRESSIONS.get(expression, ("open", "rest"))
    frame = Image.new("RGBA", tuple(rig.canvas), (0, 0, 0, 0))
    order = ["tail", "arm_far", "leg_far", "body", "leg_near", "head",
             f"eyes_{eye}", f"mouth_{mouth}", "arm_near"]
    for key in order:
        layer = rig.layers.get(key)
        if layer is None:
            continue
        with Image.open(puppet_dir / layer.file) as im:
            frame.alpha_composite(im.convert("RGBA"), tuple(layer.offset))
    return frame.crop(frame.getbbox())


def _rim_light(cut: Image.Image, colour=(255, 244, 214)) -> Image.Image:
    """A light halo behind the character. This is what separates the subject from a busy
    plate at 210x118 px, which is the size most viewers actually see."""
    alpha = cut.getchannel("A")
    glow = Image.new("RGBA", cut.size, colour + (0,))
    glow.putalpha(alpha.filter(ImageFilter.GaussianBlur(cut.width * 0.035)))
    out = Image.new("RGBA", cut.size, (0, 0, 0, 0))
    out.alpha_composite(glow)
    out.alpha_composite(cut)
    return out


def render_variant(plate: Path, puppet_dir: Path, text: str, expression: str,
                   layout: str, out_path: Path) -> ThumbVariant:
    w, h = SIZE
    with Image.open(plate) as im:
        bg = im.convert("RGB").resize(
            (int(w * 1.12), int(h * 1.12)), Image.LANCZOS)
    bg = bg.crop((0, 0, w, h))
    bg = ImageEnhance.Color(bg).enhance(1.14)
    bg = ImageEnhance.Contrast(bg).enhance(1.10)
    bg = bg.filter(ImageFilter.GaussianBlur(1.6))          # push the plate back
    canvas = bg.convert("RGBA")

    cx, cscale, band = LAYOUTS.get(layout, LAYOUTS["right_hero"])
    cut = _character_cutout(puppet_dir, expression)
    occupied = (0, 0)
    if cut is not None:
        target_h = int(h * 0.86 * cscale)
        ratio = target_h / cut.height
        cut = cut.resize((max(1, int(cut.width * ratio)), target_h), Image.LANCZOS)
        cut = _rim_light(cut)
        left = int(cx * w - cut.width / 2)
        canvas.alpha_composite(cut, (left, h - cut.height))
        occupied = (left, left + cut.width)

    draw = ImageDraw.Draw(canvas)
    margin = int(w * SAFE)
    gap = int(w * 0.03)
    free_left = max(0, occupied[0] - margin - gap)
    free_right = max(0, w - margin - occupied[1] - gap)
    if max(free_left, free_right) < w * 0.30:
        # The character fills the frame. Put the type in the vertical strip above its
        # head, which is the only place left that is not its face.
        box_w = w - 2 * margin
        x0 = margin
        y_limit = h - (cut.height if cut is not None else 0)
        box_h = max(int(h * 0.16), y_limit - margin)
    elif free_left >= free_right:
        box_w, x0, box_h = free_left, margin, int(h * 0.46)
    else:
        box_w, x0, box_h = free_right, occupied[1] + gap, int(h * 0.46)

    font, lines, lh = _fit_text(draw, text.upper(), box_w, box_h)
    y0 = int(band * h)
    y0 = max(margin, min(y0, h - lh * len(lines) - margin))
    fill, stroke = _contrast_colour(canvas, (x0, y0, x0 + box_w, y0 + lh * len(lines)))
    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * lh), line, font=font, fill=fill,
                  stroke_width=max(4, font.size // 14), stroke_fill=stroke)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = canvas.convert("RGB")
    rgb.save(out_path, quality=92)
    _shrink_to_limit(out_path, rgb)

    scores = _score(rgb, cut, lines, text)
    return ThumbVariant(0, out_path, text, expression, layout,
                        round(sum(scores.values()) / len(scores), 4), scores)


def _shrink_to_limit(path: Path, img: Image.Image) -> None:
    q = 92
    while path.stat().st_size > MAX_BYTES and q > 55:
        q -= 8
        img.save(path, quality=q)


def _score(img: Image.Image, cut, lines: list[str], text: str) -> dict:
    small = img.resize((214, 120), Image.LANCZOS)          # judge it at feed size
    px = list(small.convert("RGB").getdata())
    lums = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in px]
    mean = sum(lums) / len(lums)
    spread = (sum((x - mean) ** 2 for x in lums) / len(lums)) ** 0.5

    sats = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[1] for r, g, b in px[::7]]
    saturation = sum(sats) / len(sats)

    words = len(text.split())
    return {
        # Contrast at feed size is the single best mechanical predictor of legibility.
        "contrast": round(min(1.0, spread / 62.0), 3),
        "saturation": round(min(1.0, saturation / 0.55), 3),
        # Three words is the sweet spot; both directions are penalised.
        "text_economy": round(max(0.0, 1.0 - abs(words - 3) * 0.17), 3),
        "line_count": round({1: 1.0, 2: 0.94, 3: 0.72}.get(len(lines), 0.4), 3),
        "subject_size": round(min(1.0, (cut.height / 720) / 0.72), 3) if cut else 0.2,
    }


def generate_set(db: Path, job_id: int, plate: Path, puppet_dir: Path,
                 texts: list[str], out_dir: Path, variants: int = 6) -> list[ThumbVariant]:
    combos = [(t, e, l) for t in texts
              for e, l in (("surprised", "right_hero"), ("determined", "left_hero"),
                           ("happy", "corner_up"), ("curious", "centre_low"))]
    made: list[ThumbVariant] = []
    for i, (text, expr, layout) in enumerate(combos[:variants], start=1):
        v = render_variant(plate, puppet_dir, text, expr, layout,
                           out_dir / f"thumb_{i:02d}.jpg")
        v.variant = i
        made.append(v)
    made.sort(key=lambda v: -v.score)
    with tx(db) as con:
        con.executemany("""
            INSERT INTO thumbnails (job_id, variant, path, text_used, expression, layout,
                                    score, scores, chosen)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id, variant) DO UPDATE SET
                path=excluded.path, score=excluded.score, scores=excluded.scores,
                chosen=excluded.chosen
        """, [(job_id, v.variant, str(v.path), v.text, v.expression, v.layout, v.score,
               jdump(v.scores), 1 if v is made[0] else 0) for v in made])
    log.info("thumbnails_built", job=job_id, n=len(made),
             best=made[0].score if made else None)
    return made
