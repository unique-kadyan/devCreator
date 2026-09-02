"""Closed vocabularies the LLM selects from. Nothing here is free text."""
from __future__ import annotations

CAMERA_MOVES = ["static", "push_in", "pull_out", "pan_left", "pan_right",
                "tilt_up", "tilt_down", "handheld_drift"]
SHOT_TYPES = ["wide", "full", "medium", "close_up", "extreme_close_up",
              "two_shot", "over_shoulder", "insert"]
GESTURES = ["idle", "talk", "point", "wave", "shrug", "jump",
            "run_cycle", "walk_cycle", "sit", "react_shock", "react_sad", "laugh"]
TRANSITIONS = ["cut", "dissolve", "fade_black", "whip_pan", "iris"]
EMOTIONS = ["neutral", "curious", "happy", "sad", "scared", "angry",
            "surprised", "determined", "wry", "excited"]
EASING = ["linear", "in", "out", "in_out"]
