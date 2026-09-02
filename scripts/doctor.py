#!/usr/bin/env python
"""Thin wrapper so `python scripts/doctor.py` and `asa doctor` are the same thing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asa.core.cli import main
raise SystemExit(main(["doctor"] + sys.argv[1:]))
