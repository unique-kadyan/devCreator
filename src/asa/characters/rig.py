"""The rig contract. Every puppet - procedural or generated-and-cut - conforms to this."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

VISEMES = ["rest", "A", "E", "I", "O", "U", "M"]
EYE_STATES = ["open", "half", "closed"]


class Layer(BaseModel):
    """One PNG, cropped to its alpha bounding box, plus where it sits on the rig canvas."""

    file: str
    offset: tuple[int, int]          # top-left of this PNG within the rig canvas
    size: tuple[int, int]


class Rig(BaseModel):
    character_id: str
    canvas: tuple[int, int] = (1024, 1024)
    ground_y: int                     # where the feet meet the floor, in canvas space
    anchors: dict[str, tuple[int, int]] = Field(default_factory=dict)
    z_order: list[str] = Field(default_factory=list)
    layers: dict[str, Layer] = Field(default_factory=dict)
    visemes: dict[str, str] = Field(default_factory=dict)   # viseme -> layer key
    eyes: dict[str, str] = Field(default_factory=dict)      # state  -> layer key
    style_hash: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Rig":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(), indent=2))

    def layer_path(self, base: Path, key: str) -> Path:
        return base / self.layers[key].file
