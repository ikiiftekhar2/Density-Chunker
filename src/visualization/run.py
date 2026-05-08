#!/usr/bin/env python3
"""CLI entry point: generate all analysis figures.

Usage:
    python -m visualization.run
    # or:
    python run.py
"""

import sys
from pathlib import Path

# Ensure the package root is on sys.path
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from visualization.figures import generate_all

if __name__ == "__main__":
    generate_all()
