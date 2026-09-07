"""Where the face is in a generated still, and which pixels are the jaw.

Animating a mouth means knowing which pixels are the mouth, and the image arrives from a
text-to-image model with no metadata at all. The three usual answers do not apply here: a
face landmarker is trained on human faces and these are muzzles; a detector is another
dependency and another model file on a machine chosen for having neither; and asking the
image model for coordinates is asking it to describe a picture it has already forgotten.

What this module uses instead is the fact that **we wrote the prompt**. A shot generated
from "close-up portrait, head and shoulders fill the frame, face centred in frame, looking
towards camera" puts the head in a predictable place, so the framing word is itself a
location. The numbers in PRIORS are not guesses: they were measured off real FLUX.1-schnell
output for each of these composition strings, which is why they must be re-measured if
`_SHOT_COMPOSITION` in media/images/scene_image.py is reworded.

An earlier version refined the prior by hunting for the most detailed region of the
picture. It was removed. On a jungle or a barred corridor the foliage and the bars carry
more local contrast than a face does, so the estimate walked steadily away from the head -
worse than the fixed prior it was "correcting", and worse in the way that matters, because
a wrong box animates a chest or a wall while the face sits still.

KNOWN LIMIT, measured rather than suspected: the framing word is only a location while the
image model obeys it, and FLUX.1-schnell does not reliably obey it. In one finished episode
a shot requested as a close-up of an owl came back as a wide two-shot of a rooftop, and the
prior then described sky. Every box from PRIORS therefore carries a low `confidence`, and
callers are expected to scale the effect by it instead of trusting the geometry. This is
why the local renderer produces jaw motion and not lip-sync, and why real lip-sync lives
behind a hosted model (media/video/replicate.py) that was trained to find a face itself.

A real detector can still be dropped in: write `{"face": [x, y, w, h]}` into the image's
`.face.json` sidecar and it is used verbatim, at full confidence. YuNet was evaluated for
this and rejected - on twelve generated stills it missed five, and on one it preferred a
human in the background to the animal filling the frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...core.logging import get_logger

log = get_logger("face")

# Normalised (x, y, w, h) of the head, (x, y) being the top-left corner. Measured from
# generated shots: in every composition the head starts at the very top of frame and the
# mouth sits at about 0.80 of the box height, which is what MOUTH_* below are anchored to.
PRIORS: dict[str, tuple[float, float, float, float]] = {
    # A speaking close-up asks for the head to fill the frame edge to edge, and owl, fox,
    # lion and elephant all came back composed that way - so the head box IS the frame.
    # Both compositions ask for a head that fills the frame, so both boxes are the frame.
    # `extreme_close_up` crops tighter still (eyes and muzzle only), which puts its mouth
    # lower in the box than the prior says - it is approximate, like every entry here, and
    # the confidence below is what keeps that from mattering much.
    "extreme_close_up": (0.10, 0.00, 0.80, 1.00),
    "close_up":         (0.10, 0.00, 0.80, 1.00),
    # The two looser framings are measured separately: their composition strings do not
    # promise a full-frame head, so their heads sit high and small.
    "medium":           (0.30, 0.00, 0.40, 0.75),
    "over_shoulder":    (0.38, 0.00, 0.36, 0.69),
}
# Framings with no animatable face: the subject is too small to read a jaw on, or there are
# two of them and animating the wrong one is worse than animating neither.
NO_FACE = ("wide", "full", "insert", "two_shot")

# The MUZZLE, as fractions of the head box - a region, not a half-face band.
#
# These were re-cut after measuring a finished episode. The previous numbers (top 0.44,
# bottom 1.02, spanning the full head width plus 8% either side) put the top of the warp
# through the EYES and its bottom off the frame entirely, across the whole picture width.
# The frame-to-frame difference map of a "talking" close-up was consequently an edge map of
# the entire image: glasses, ears, hoodie, laptop and the bookshelf behind the character all
# moved together. That is what reads as flickering rather than speech.
#
# So the band is now bounded on all four sides and sits inside the head. It fades to nothing
# at MOUTH_TOP (below the eyes on every sample), travels most at MOUTH_PEAK, and is absorbed
# again by MOUTH_BOTTOM (above the bottom of the head box, so the chest never moves).
MOUTH_TOP = 0.58
MOUTH_PEAK = 0.76
MOUTH_BOTTOM = 0.94
# Half-width of the muzzle as a fraction of head width, measured from the head's centre.
# A muzzle is roughly two fifths of the width of the head it is on; the old code used 1.16x
# the head width, which is why the background moved.
MOUTH_HALF_WIDTH = 0.21
# The eyes, for the blink. Measured on the same samples.
EYE_TOP = 0.22
EYE_BOTTOM = 0.36
EYE_HALF_WIDTH = 0.42


@dataclass(frozen=True)
class FaceBox:
    """Normalised head box, (x, y) = top-left."""

    x: float
    y: float
    w: float
    h: float
    source: str = "prior"
    # How much this box should be trusted, 0..1. A prior is a guess about a picture nobody
    # has looked at: the image model is asked for a close-up and sometimes answers with a
    # wide two-shot, in which case the box describes scenery. Callers scale the effect by
    # this rather than choosing between animating confidently and not animating at all.
    confidence: float = 0.35

    def pixels(self, size: tuple[int, int]) -> tuple[int, int, int, int]:
        w, h = size
        return (int(round(self.x * w)), int(round(self.y * h)),
                max(2, int(round(self.w * w))), max(2, int(round(self.h * h))))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


def locate_face(image_path: Path, framing: str) -> FaceBox | None:
    """The head box for a generated shot, or None when the framing has no usable face."""
    framing = (framing or "medium").strip().lower()
    sidecar = Path(str(image_path) + ".face.json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            x, y, w, h = data["face"]
            return FaceBox(float(x), float(y), float(w), float(h),
                           data.get("source", "sidecar"),
                           float(data.get("confidence", 1.0)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            log.warning("face_sidecar_unreadable", path=str(sidecar))
    if framing in NO_FACE or framing not in PRIORS:
        return None
    return FaceBox(*PRIORS[framing])
