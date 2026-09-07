"""Whether a generated still actually contains a character. Measured; NOT a face locator.

Read this before trying pose estimation for the mouth again, because the obvious version of
that idea was tried here and does not work.

WHAT WAS TRIED. `face.PRIORS` places the head from the framing word we asked for, at
confidence 0.35 - deliberately below `min_face_confidence`, so nothing is warped on a guess.
That is correct, and it is also why the local path animates nothing today: no detector writes
a sidecar, so every shot falls to a prior, every prior is below the floor, and every shot
renders as a camera move over a still. The plan was to raise that with a detector.

A FACE model is the wrong tool and that was already known - YuNet missed five stills in
twelve and preferred a human in the background to the animal filling the frame. So POSE
looked right instead: these characters are "animal head and a human body, wearing real
clothes" (scene_image.py), and while the head is a muzzle, the body is human and clothed.
MediaPipe's pose landmarker was measured on 30 of this project own generated stills.

WHAT WORKED. Finding the character. A pose came back on 22 of 30, and after the anatomy
check below, 20 were usable - against YuNet 7 in 12. Position is right too: the figure it
finds is the figure in the picture.

WHAT DID NOT. The geometry, which is the only part the muzzle band needs. The landmark
SCALE is wrong on these images, and wrong in a way no anchor fixes:

  - Nose-to-shoulder distance came back as 0.01-0.07 of frame height on stills where the
    head fills a third of the frame. A head derived from it is 0.02-0.16 tall.
  - Ear separation collapses in profile, so a width taken from it lands the box on one ear.
  - On a close-up the model rescales the whole skeleton to fit its idea of a human figure,
    putting "nose" roughly where a human nose would be given the shoulders - which is well
    below an animal muzzle. Anchoring the band there put it on chests and scarves across
    every sample.

Anchoring on the prior box instead does not rescue it: the `close_up` prior IS the whole
frame (0.10, 0.00, 0.80, 1.00), so there is no freedom left to position anything, and the
band lands at a fixed 0.58-0.94 of frame however wrong that is.

So pose gives presence and rough position, not a head box. Placing a muzzle on these
characters needs ground-truth measurement of the offset against the sample, or a model
trained on animal heads. Until one of those exists this module deliberately does NOT write a
`.face.json` sidecar - putting a warp band on a chest is exactly the defect
`min_face_confidence` was added to stop, and a confident wrong box is worse than no box.

WHAT IT IS FOR. `has_character` - a cheap, reliable "is there a figure in this frame at
all", 0.05s per still on CPU. Useful for QC (a speaking scene whose still has no character
in it is a bad still) and as the presence half of any later rig, which needs these same
landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...core.logging import get_logger

log = get_logger("detect")

# Landmark indices in MediaPipe 33-point pose.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# Below this the "torso" is too short to be one, and the skeleton is noise that happens to be
# ordered correctly. Normalised image heights.
MIN_TORSO = 0.04

DEFAULT_MODEL = Path("data/cache/models/pose_landmarker_lite.task")

_LANDMARKER = None


@dataclass(frozen=True)
class Presence:
    """A character was found, and roughly where its head is. NOT a head box - see module."""

    head_x: float
    head_y: float
    torso: float


def _landmarker(model_path: Path):
    """The pose model, loaded once per process. None when it is not available."""
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    try:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError:
        log.info("pose_unavailable", detail="mediapipe not installed")
        return None
    if not model_path.exists():
        log.warning("pose_model_missing", path=str(model_path),
                    detail="fetch pose_landmarker_lite.task from storage.googleapis.com")
        return None
    opts = vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3)
    _LANDMARKER = vision.PoseLandmarker.create_from_options(opts)
    return _LANDMARKER


def _plausible(L) -> Presence | None:
    """A figure, or None when the skeleton is not shaped like one.

    `visibility` is not a confidence and must not be used as one. On a shelf of pottery the
    model returned a skeleton with every landmark at visibility 1.00 whose ankles sat ABOVE
    its hips. What separates a figure from a hallucination is the ordering a body actually
    has - and on the sample this check rejected exactly the two hallucinations and nothing
    else.
    """
    nose = L[NOSE]
    sh_y = (L[L_SHOULDER].y + L[R_SHOULDER].y) / 2.0
    hip_y = (L[L_HIP].y + L[R_HIP].y) / 2.0
    if not (nose.y < sh_y < hip_y):
        return None
    torso = hip_y - sh_y
    if torso < MIN_TORSO:
        return None
    return Presence(float(nose.x), float(nose.y), float(torso))


def has_character(image_path: Path, model_path: Path | None = None) -> Presence | None:
    """Is there a character in this still? ~0.05s per image on CPU, no network, no cost."""
    lm = _landmarker(Path(model_path or DEFAULT_MODEL))
    if lm is None:
        return None
    try:
        import mediapipe as mp
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
    except (OSError, ValueError, RuntimeError) as e:
        log.warning("pose_failed", path=str(image_path), error=str(e)[:120])
        return None
    if not res.pose_landmarks:
        return None
    found = _plausible(res.pose_landmarks[0])
    if found is None:
        log.info("pose_rejected", path=Path(image_path).name,
                 reason="skeleton is not head-over-shoulders-over-hips")
    return found
