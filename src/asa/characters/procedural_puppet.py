"""Draw a layered animal puppet with Pillow.

This exists so the compositor can be built and tested with zero API cost and perfectly
separated layers. Phase 4 replaces the artwork with a generated-and-cut turnaround sheet -
but it emits the SAME rig.json, so nothing downstream changes.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .rig import Rig, Layer
from .species import SpeciesProfile, profile as species_profile

SS = 2          # supersample factor for anti-aliasing
# How far a beak opens per viseme. Beaks cannot round or purse, so O/U map to a mid
# opening rather than a shape - the amplitude, not the vowel, is what reads on a bird.
BEAK_OPEN = {"rest": 0.0, "M": 0.0, "A": 1.0, "E": 0.62, "I": 0.42, "O": 0.78, "U": 0.5}
CANVAS = 1024


def _new(size: int = CANVAS):
    return Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))


def _d(img):
    return ImageDraw.Draw(img)


def _S(*vals):
    """Scale coordinates into supersampled space."""
    return tuple(int(round(v * SS)) for v in vals)


def _ellipse(dr, cx, cy, rx, ry, fill, outline=None, w=0):
    dr.ellipse(_S(cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline,
               width=int(w * SS) if w else 0)


def _poly(dr, pts, fill, outline=None, w=0):
    dr.polygon([(_S(x)[0], _S(y)[0]) for x, y in pts], fill=fill, outline=outline,
               width=int(w * SS) if w else 0)


def _down(img):
    return img.resize((CANVAS, CANVAS), Image.LANCZOS)


def _crop(img: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    bbox = img.getbbox()
    if bbox is None:
        return img, (0, 0)
    return img.crop(bbox), (bbox[0], bbox[1])


class AnimalPuppet:
    """An animal drawn from primitives, parameterised by palette + species profile.

    Layer keys, anchors and viseme names are identical for every species, so the
    compositor is species-blind and a cast swap costs nothing downstream.
    """

    def __init__(self, character_id: str, palette: dict[str, str],
                 species: str | SpeciesProfile = "fox"):
        self.cid = character_id
        self.p = dict(BASE_PALETTE, **(palette or {}))
        self.sp = species if isinstance(species, SpeciesProfile) else species_profile(species)
        # rig geometry, in 1024x1024 canvas space
        # A long-necked species raises the head instead of stretching it: the rig's other
        # anchors are all derived from head_c, so lifting it moves the eyes, mouth and
        # blink automata together with no special-casing anywhere downstream.
        self.neck_len = int(self.sp.neck_len)
        self.head_c = (512, 330 - self.neck_len)
        self.head_r = self.sp.head_r
        self.neck = (512, 452)
        self.shoulder_l = (418, 512)
        self.shoulder_r = (606, 512)
        self.hip = (512, 760)
        self.ground_y = 980
        eye_dx = int(self.head_r[0] * 0.31)
        self.eye_l = (512 - eye_dx, 316 - self.neck_len)
        self.eye_r = (512 + eye_dx, 316 - self.neck_len)
        self.mouth_c = (512, self.head_c[1] + self.sp.muzzle_dy + 18)
        self.outline = self.p.get("outline", "#3A2415")

    # ---------------------------------------------------------------- parts

    def tail(self):
        """Swept from a bezier spine so it reads as fur, not a polygon.

        `tail_len` scales the spine away from its root, so a bear stub and a monkey curl
        are the same code with different control points. A profile with tail "none"
        returns an empty layer, which crops to nothing and costs no draw time.
        """
        img = _new(); dr = _d(img)
        if self.sp.tail == "none" or self.sp.tail_len <= 0.01:
            return img
        L = self.sp.tail_len
        root = (438, 742)
        if self.sp.tail == "curl":
            ctrl = [(360, 640), (232, 664), (268, 792)]
        elif self.sp.tail == "stub":
            ctrl = [(400, 728), (368, 748), (356, 792)]
        else:
            ctrl = [(350, 700), (250, 760), (238, 872)]
        p0 = root
        p1, p2, p3 = [(root[0] + (cx - root[0]) * L, root[1] + (cy - root[1]) * L)
                      for cx, cy in ctrl]
        spine = []
        for i in range(41):
            t = i / 40
            u = 1 - t
            x = u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0]
            y = u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]
            spine.append((t, x, y))
        base, swell = (12, 16) if self.sp.tail in ("thin", "tuftend") else (30, 44)
        base *= self.sp.tail_thick
        swell *= self.sp.tail_thick
        tipped = self.sp.tail == "bushy"
        for t, x, y in spine:                       # outline pass
            r = base + swell * math.sin(math.pi * min(1.0, t * 0.92 + 0.08))
            _ellipse(dr, x, y, r + 6, r + 6, self.outline)
        for t, x, y in spine:                       # fur pass
            r = base + swell * math.sin(math.pi * min(1.0, t * 0.92 + 0.08))
            col = self.p["chest"] if (tipped and t > 0.74) else self.p["fur"]
            _ellipse(dr, x, y, r, r, col)
        if self.sp.tail == "tuftend":               # lion / bull / giraffe switch
            tx, ty = spine[-1][1], spine[-1][2]
            _ellipse(dr, tx, ty + 16, 34, 40, self.outline)
            _ellipse(dr, tx, ty + 16, 28, 34, self.p["mane"])
        return img

    def leg(self, near: bool):
        img = _new(); dr = _d(img)
        x = 562 if near else 462
        shade = self.p["fur"] if near else self.p["fur_shadow"]
        _poly(dr, [(x - 44, 762), (x + 44, 762), (x + 40, 926), (x - 40, 926)],
              shade, self.outline, 6)
        _ellipse(dr, x, 942, 56, 30, self.p["paw"], self.outline, 6)
        return img

    def arm(self, near: bool):
        img = _new(); dr = _d(img)
        sx, sy = self.shoulder_r if near else self.shoulder_l
        shade = self.p["hoodie"] if near else self.p["hoodie_shadow"]
        out = 22 if near else -22
        top, bot = 34, 22
        _ellipse(dr, sx, sy - 4, top, top - 4, shade, self.outline, 6)   # shoulder cap
        _poly(dr, [(sx - top, sy - 6), (sx + top, sy - 6),
                   (sx + bot + out, sy + 172), (sx - bot + out, sy + 172)],
              shade, self.outline, 6)
        _ellipse(dr, sx + out, sy + 186, 32, 30, self.p["paw"], self.outline, 6)
        return img

    def body(self):
        img = _new(); dr = _d(img)
        brx, bry = self.sp.body_r
        if self.neck_len:
            nw = brx * 0.46
            dr.rounded_rectangle(
                _S(512 - nw, 470 - self.neck_len, 512 + nw, 540),
                radius=int(nw * SS), fill=self.p["fur"], outline=self.outline,
                width=int(7 * SS))
        _ellipse(dr, 512, 638, brx, bry, self.p["hoodie"], self.outline, 7)
        # hood bunched at the neck
        _ellipse(dr, 512, 500, int(brx * 0.76), 46, self.p["hoodie_shadow"],
                 self.outline, 6)
        # pocket
        dr.rounded_rectangle(_S(452, 690, 572, 748), radius=int(16 * SS),
                             outline=self.outline, width=int(5 * SS))
        # shorts
        _poly(dr, [(392, 726), (632, 726), (620, 812), (404, 812)],
              self.p["shorts"], self.outline, 6)
        return img

    def _ears(self, dr, cx: int, cy: int) -> None:
        """Ears are drawn before the skull so the skull hides their roots."""
        sp = self.sp
        if sp.ear == "none":
            return
        w, tilt = 66 * sp.ear_w, sp.ear_tilt
        by = cy - int(self.head_r[1] * 0.52)
        # Clamp the ear so its tip cannot leave the canvas. A cropped rabbit ear is not a
        # stylistic choice - it is a layer whose bbox silently hits the canvas edge and
        # then scales wrong in every shot.
        h = min(158 * sp.ear_h, by - 24)
        for side in (-1, 1):
            bx = cx + side * int(self.head_r[0] * 0.86)
            tipx = bx + side * tilt
            if sp.ear in ("triangle", "tuft"):
                outer = [(bx - side * w * 0.42, by), (tipx + side * w * 0.28, by - h),
                         (bx + side * w * 0.72, by + h * 0.16)]
                inner = [(bx - side * w * 0.18, by - h * 0.10),
                         (tipx + side * w * 0.18, by - h * 0.80),
                         (bx + side * w * 0.46, by + h * 0.04)]
                _poly(dr, outer, self.p["fur"], self.outline, 7)
                _poly(dr, inner, self.p["ear_inner"])
            elif sp.ear == "fan":
                # A big flat ear that hangs beside the head rather than standing above it -
                # the single feature that makes an elephant read as an elephant.
                rx, ry = w * 0.92, h * 0.62
                ecx = bx + side * tilt
                ecy = by + ry * 0.22
                _ellipse(dr, ecx, ecy, rx, ry, self.p["fur"], self.outline, 7)
                _ellipse(dr, ecx + side * rx * 0.12, ecy, rx * 0.62, ry * 0.70,
                         self.p["ear_inner"])
            elif sp.ear == "tiny":
                r = max(14.0, w * 0.34)
                _ellipse(dr, bx + side * tilt * 0.4, by - r * 0.5, r, r,
                         self.p["fur"], self.outline, 6)
                _ellipse(dr, bx + side * tilt * 0.4, by - r * 0.5, r * 0.5, r * 0.5,
                         self.p["ear_inner"])
            else:                                   # round / long share an ellipse
                ry = h * (0.5 if sp.ear == "round" else 0.56)
                rx = w * (0.62 if sp.ear == "round" else 0.42)
                ecx, ecy = bx + side * tilt * 0.5, by - ry * 0.72
                _ellipse(dr, ecx, ecy, rx, ry, self.p["fur"], self.outline, 7)
                _ellipse(dr, ecx, ecy, rx * 0.56, ry * 0.66, self.p["ear_inner"])
        self._horns(dr, cx, cy)

    def _horns(self, dr, cx: int, cy: int) -> None:
        """Four horn shapes, because a goat, a bull, a rhino and a deer are four different
        silhouettes and using one shape for all of them loses the species."""
        kind = self.sp.horns
        if kind == "none":
            return
        rx, ry = self.head_r
        col = self.p["horn"]
        if kind == "goat":
            for side in (-1, 1):
                hx = cx + side * int(rx * 0.44)
                hy = cy - int(ry * 0.96)
                _poly(dr, [(hx - side * 20, hy + 16), (hx + side * 54, hy - 104),
                           (hx + side * 18, hy - 6)], col, self.outline, 6)
        elif kind == "bull":
            # Swept from a circle sweep, not a thin polygon: at thumbnail size a 6px
            # outline with nothing inside it reads as an antenna, which is what the first
            # version looked like.
            for side in (-1, 1):
                pts = []
                for k in range(15):
                    t = k / 14
                    px = cx + side * (rx * 0.62 + 108 * t)
                    py = cy - ry * 0.30 - 96 * (t ** 1.7) + 26 * math.sin(t * 2.4)
                    pts.append((px, py, 30 - 22 * t))
                for px, py, r in pts:
                    _ellipse(dr, px, py, r + 5, r + 5, self.outline)
                for px, py, r in pts:
                    _ellipse(dr, px, py, r, r, col)
        elif kind == "ossicone":
            for side in (-1, 1):
                hx = cx + side * int(rx * 0.40)
                hy = cy - int(ry * 0.90)
                dr.line([_S(hx, hy + 12), _S(hx + side * 10, hy - 62)],
                        fill=self.outline, width=int(20 * SS))
                dr.line([_S(hx, hy + 12), _S(hx + side * 10, hy - 62)],
                        fill=self.p["fur"], width=int(13 * SS))
                _ellipse(dr, hx + side * 10, hy - 66, 20, 18, col, self.outline, 5)
        elif kind == "antler":
            for side in (-1, 1):
                hx = cx + side * int(rx * 0.46)
                hy = cy - int(ry * 0.92)
                dr.line([_S(hx, hy + 10), _S(hx + side * 34, hy - 96)],
                        fill=col, width=int(11 * SS))
                dr.line([_S(hx + side * 12, hy - 26), _S(hx + side * 62, hy - 58)],
                        fill=col, width=int(8 * SS))
                dr.line([_S(hx + side * 24, hy - 62), _S(hx + side * 8, hy - 128)],
                        fill=col, width=int(8 * SS))

    def head(self):
        img = _new(); dr = _d(img)
        cx, cy = self.head_c
        rx, ry = self.head_r
        mrx, mry = self.sp.muzzle
        mdy = self.sp.muzzle_dy
        if self.sp.mane:                            # mane sits behind everything
            for pass_r, pass_outline in ((46, self.outline), (40, None)):
                for k in range(18):
                    a = 2 * math.pi * k / 18
                    _ellipse(dr, cx + math.cos(a) * rx * 0.98,
                             cy + math.sin(a) * ry * 0.98, pass_r, pass_r,
                             self.p["mane"], pass_outline, 5 if pass_outline else 0)
        if self.sp.spines:                          # spine crown
            for k in range(11):
                t = k / 10
                a = math.pi * (1.06 + 0.88 * t)     # sweep across the top of the skull
                px, py = cx + math.cos(a) * rx * 0.94, cy + math.sin(a) * ry * 0.94
                _poly(dr, [(px - 22, py + 12), (px + math.cos(a) * 74,
                                                py + math.sin(a) * 74), (px + 22, py + 12)],
                      self.p["spine"], self.outline, 4)
        self._ears(dr, cx, cy)
        # skull
        _ellipse(dr, cx, cy, rx, ry, self.p["fur"], self.outline, 7)
        if self.sp.stripes:
            for k in range(-2, 3):
                sx = cx + k * rx * 0.34
                _poly(dr, [(sx - 9, cy - ry * 0.94), (sx + 9, cy - ry * 0.94),
                           (sx + 15, cy - ry * 0.34), (sx - 3, cy - ry * 0.34)],
                      self.p["stripe"])
        if self.sp.spots:
            for k, (ox, oy, r) in enumerate((
                    (-0.52, -0.30, 0.15), (0.46, -0.36, 0.13), (-0.30, 0.24, 0.12),
                    (0.54, 0.14, 0.14), (0.06, -0.62, 0.11), (-0.62, 0.10, 0.10))):
                _ellipse(dr, cx + ox * rx, cy + oy * ry, r * rx, r * ry, self.p["spot"])
        if self.sp.mask:                            # bandit mask
            _ellipse(dr, cx, cy - 8, rx * 0.92, ry * 0.34, self.p["mask"])
        if self.sp.facial_disc:                     # the flat owl face
            for side in (-1, 1):
                _ellipse(dr, cx + side * rx * 0.34, cy - 6, rx * 0.44, ry * 0.44,
                         self.p["chest"], self.outline, 4)
        # muzzle
        if self.sp.trunk:
            # Drawn as a tapering stack of circles so the outline stays smooth. It hangs
            # BELOW the mouth anchor, which is why the elephant's visemes sit at its base
            # rather than on the skull.
            top = cy + mdy - 20
            reach = min(330.0, (self.ground_y - 200) - top)
            spine = []
            for k in range(30):
                t = k / 29
                w_ = mrx * (0.86 - 0.58 * t)
                x_ = cx + math.sin(t * 2.6) * mrx * 0.30
                spine.append((x_, top + t * reach, w_))
            for x_, y_, w_ in spine:
                _ellipse(dr, x_, y_, w_ + 6, w_ + 6, self.outline)
            for x_, y_, w_ in spine:
                _ellipse(dr, x_, y_, w_, w_, self.p["fur"])
        _ellipse(dr, cx, cy + mdy, mrx, mry, self.p["chest"], self.outline, 6)
        if self.sp.horns == "rhino":
            # On the snout, so it must come after the muzzle or the muzzle buries it.
            base = cy + mdy - mry * 0.30
            _poly(dr, [(cx - 36, base), (cx + 36, base), (cx + 12, base - 126),
                       (cx - 8, base - 130)], self.p["horn"], self.outline, 7)
            _poly(dr, [(cx - 24, base - 112), (cx + 22, base - 112),
                       (cx + 2, base - 172)], self.p["horn"], self.outline, 6)
        if self.sp.tusks:
            for side in (-1, 1):
                bx = cx + side * mrx * 0.66
                by_ = cy + mdy + mry * 0.30
                _poly(dr, [(bx - side * 13, by_), (bx + side * 13, by_),
                           (bx + side * 30, by_ + 96), (bx + side * 8, by_ + 104)],
                      self.p["horn"], self.outline, 5)
        # brow line gives the face some character
        dr.arc(_S(cx - rx * 0.64, cy - ry * 0.57, cx + rx * 0.64, cy + ry * 0.29),
               start=200, end=340, fill=self.outline, width=int(5 * SS))
        # nose
        nrx, nry = self.sp.nose_r
        # A beaked species has no separate nose: the beak IS the mouth, and it is drawn in
        # the mouth layer so it can open. Drawing it here too would leave a fixed beak
        # buried under every open viseme.
        if self.sp.mouth_style != "beak":
            _ellipse(dr, cx, cy + mdy - 32, nrx, nry, self.p["nose"])
        if self.sp.whiskers:
            for side in (-1, 1):
                for k, dy in enumerate((-10, 4, 18)):
                    x0 = cx + side * mrx * 0.72
                    dr.line([_S(x0, cy + mdy + dy),
                             _S(x0 + side * (78 - k * 8), cy + mdy + dy - 14 + k * 12)],
                            fill=self.outline, width=int(3 * SS))
        return img

    def eyes(self, state: str):
        img = _new(); dr = _d(img)
        for (ex, ey) in (self.eye_l, self.eye_r):
            if state == "closed":
                dr.arc(_S(ex - 26, ey - 18, ex + 26, ey + 22), start=200, end=340,
                       fill=self.outline, width=int(7 * SS))
                continue
            h = 30 if state == "open" else 16
            _ellipse(dr, ex, ey, 28, h, "#FFFFFF", self.outline, 5)
            _ellipse(dr, ex + 3, ey + (0 if state == "open" else 4),
                     15, min(15, h - 4), self.p["eyes"])
            _ellipse(dr, ex + 3, ey + (0 if state == "open" else 4), 7, min(8, h - 6), "#101010")
            _ellipse(dr, ex - 6, ey - 9, 5, 5, "#FFFFFF")
        return img

    def _beak(self, dr, cx: int, cy: int, openness: float) -> None:
        """A hinged beak. `openness` 0..1 rotates the lower mandible down about the hinge."""
        nrx, nry = self.sp.nose_r
        top = cy - nry * 1.1
        drop = openness * nry * 2.0
        _poly(dr, [(cx - nrx, top), (cx + nrx, top), (cx, top + nry * 1.15)],
              self.p["horn"], self.outline, 5)
        if openness > 0.04:
            _poly(dr, [(cx - nrx * 0.86, top + nry * 0.30 + drop * 0.25),
                       (cx + nrx * 0.86, top + nry * 0.30 + drop * 0.25),
                       (cx, top + nry * 1.30 + drop)],
                  self.p["horn"], self.outline, 5)
            _poly(dr, [(cx - nrx * 0.70, top + nry * 0.42),
                       (cx + nrx * 0.70, top + nry * 0.42),
                       (cx, top + nry * 0.52 + drop * 0.9)], self.p["mouth"])

    def mouth(self, viseme: str):
        """Seven visemes. Shapes are exaggerated - they read at 24fps, subtle does not."""
        img = _new(); dr = _d(img)
        cx, cy = self.mouth_c
        if self.sp.mouth_style == "beak":
            self._beak(dr, cx, cy, BEAK_OPEN[viseme])
            return img
        spec = {
            "rest": (0, 0), "M": (0, 0),
            "A": (46, 40), "E": (44, 20), "I": (34, 14), "O": (30, 34), "U": (22, 26),
        }[viseme]
        if spec == (0, 0):
            dr.arc(_S(cx - 34, cy - 22, cx + 34, cy + 18), start=20, end=160,
                   fill=self.outline, width=int(6 * SS))
        else:
            rx, ry = spec
            _ellipse(dr, cx, cy, rx, ry, self.p["mouth"], self.outline, 6)
            if ry > 22:
                _ellipse(dr, cx, cy + ry * 0.35, rx * 0.55, ry * 0.34, self.p["tongue"])
        return img

    # ---------------------------------------------------------------- build

    def build(self, out_dir: Path) -> Rig:
        out_dir.mkdir(parents=True, exist_ok=True)
        parts: dict[str, Image.Image] = {
            "tail": self.tail(),
            "leg_far": self.leg(near=False),
            "leg_near": self.leg(near=True),
            "arm_far": self.arm(near=False),
            "body": self.body(),
            "head": self.head(),
            "arm_near": self.arm(near=True),
        }
        for s in ("open", "half", "closed"):
            parts[f"eyes_{s}"] = self.eyes(s)
        for v in ("rest", "A", "E", "I", "O", "U", "M"):
            parts[f"mouth_{v}"] = self.mouth(v)

        layers: dict[str, Layer] = {}
        for key, img in parts.items():
            small = _down(img)
            cropped, off = _crop(small)
            fname = f"{key}.png"
            cropped.save(out_dir / fname)
            layers[key] = Layer(file=fname, offset=off, size=cropped.size)

        rig = Rig(
            character_id=self.cid,
            canvas=(CANVAS, CANVAS),
            ground_y=self.ground_y,
            anchors={
                "neck": self.neck, "shoulder_l": self.shoulder_l, "shoulder_r": self.shoulder_r,
                "hip": self.hip, "head": self.head_c,
                "eye_l": self.eye_l, "eye_r": self.eye_r, "mouth": self.mouth_c,
            },
            z_order=["tail", "arm_far", "leg_far", "body", "leg_near",
                     "head", "eyes", "mouth", "arm_near"],
            layers=layers,
            visemes={v: f"mouth_{v}" for v in ("rest", "A", "E", "I", "O", "U", "M")},
            eyes={s: f"eyes_{s}" for s in ("open", "half", "closed")},
            style_hash="sha256:" + hashlib.sha256(
                (self.cid + repr(sorted(self.p.items()))).encode()).hexdigest()[:16],
        )
        rig.save(out_dir / "rig.json")
        return rig


BASE_PALETTE = {
    "fur": "#E07A35", "fur_shadow": "#C4682C", "chest": "#FFF6E8",
    "ear_inner": "#8C4A2A", "eyes": "#4C9A54", "nose": "#2B2B2B",
    "paw": "#F3E3CF", "hoodie": "#3B6EA5", "hoodie_shadow": "#2F5A88",
    "shorts": "#6B4A2F", "mouth": "#4A2320", "tongue": "#D46A72",
    "mask": "#3A3F45", "horn": "#E8DCC0", "outline": "#3A2415",
    "mane": "#B4611F", "spine": "#7A5230", "stripe": "#2E241C", "spot": "#8A5A28",
}

MILO_PALETTE = {
    "fur": "#E07A35", "fur_shadow": "#C4682C", "chest": "#FFF6E8",
    "ear_inner": "#8C4A2A", "eyes": "#4C9A54", "nose": "#2B2B2B",
    "paw": "#F3E3CF", "hoodie": "#3B6EA5", "hoodie_shadow": "#2F5A88",
    "shorts": "#6B4A2F", "mouth": "#4A2320", "tongue": "#D46A72",
}


# Back-compat: the compositor tests and phase1/2 slices import FoxPuppet.
FoxPuppet = AnimalPuppet
