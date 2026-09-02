"""Camera moves over a world-space plate. Closed vocabulary - see docs/02 §8.2."""
from __future__ import annotations

import math
from dataclasses import dataclass

CAMERA_MOVES = ["static", "push_in", "pull_out", "pan_left", "pan_right",
                "tilt_up", "tilt_down", "handheld_drift"]
SHOTS = {"wide": 1.00, "full": 0.86, "medium": 0.68, "close_up": 0.46,
         "extreme_close_up": 0.30, "two_shot": 0.80, "over_shoulder": 0.60, "insert": 0.40}


def ease(t: float, kind: str = "in_out") -> float:
    t = max(0.0, min(1.0, t))
    if kind == "linear":
        return t
    if kind == "in":
        return t * t
    if kind == "out":
        return 1 - (1 - t) ** 2
    return 0.5 * (1 - math.cos(math.pi * t))     # in_out


@dataclass(frozen=True)
class Viewport:
    """A crop rectangle in world space, later resized to the output frame."""
    cx: float
    cy: float
    w: float
    h: float

    def box(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)


class Camera:
    """Interpolates a viewport across a shot."""

    def __init__(self, world: tuple[int, int], frame: tuple[int, int],
                 move: str = "static", shot_from: str = "wide", shot_to: str | None = None,
                 focus: tuple[float, float] = (0.5, 0.5), easing: str = "in_out"):
        self.world_w, self.world_h = world
        self.aspect = frame[0] / frame[1]
        self.move = move if move in CAMERA_MOVES else "static"
        self.z0 = SHOTS.get(shot_from, 1.0)
        self.z1 = SHOTS.get(shot_to or shot_from, self.z0)
        self.focus = focus
        self.easing = easing

    def _viewport_for(self, zoom: float, fx: float, fy: float) -> Viewport:
        w = self.world_w * zoom
        h = w / self.aspect
        if h > self.world_h:
            h = self.world_h
            w = h * self.aspect
        cx = fx * self.world_w
        cy = fy * self.world_h
        cx = min(max(cx, w / 2), self.world_w - w / 2)
        cy = min(max(cy, h / 2), self.world_h - h / 2)
        return Viewport(cx, cy, w, h)

    def at(self, t: float) -> Viewport:
        """t in [0,1] across the shot."""
        e = ease(t, self.easing)
        fx, fy = self.focus
        zoom = self.z0 + (self.z1 - self.z0) * e
        drift_x = drift_y = 0.0
        if self.move == "push_in":
            zoom = self.z0 + (min(self.z0, self.z1) * 0.72 - self.z0) * e if self.z1 == self.z0 else zoom
        elif self.move == "pull_out":
            zoom = self.z0 + (min(1.0, self.z0 * 1.38) - self.z0) * e if self.z1 == self.z0 else zoom
        elif self.move == "pan_left":
            drift_x = -0.16 * e
        elif self.move == "pan_right":
            drift_x = 0.16 * e
        elif self.move == "tilt_up":
            drift_y = -0.12 * e
        elif self.move == "tilt_down":
            drift_y = 0.12 * e
        elif self.move == "handheld_drift":
            drift_x = 0.006 * math.sin(t * math.tau * 1.7)
            drift_y = 0.005 * math.sin(t * math.tau * 1.3 + 1.1)
        return self._viewport_for(zoom, fx + drift_x, fy + drift_y)
