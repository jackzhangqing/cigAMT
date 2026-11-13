#!/usr/bin/env python3
"""Standalone clean-task specialist supplement; no legacy package is imported."""
from specialist_baselines.runtime.cli import entry

if __name__ == '__main__':
    raise SystemExit(entry())
