#!/usr/bin/env python3
"""Separate entry point; the existing benchmark amtbench package is untouched."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from specialist_baselines.runtime.cli import entry

if __name__ == '__main__':
    raise SystemExit(entry())
