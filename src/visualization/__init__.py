"""Visualization module for Density Chunker analysis.

Generates publication-quality figures for the chunking evaluation paper.

Usage:
    from visualization import generate_all, FIGURES_DIR
    generate_all()  # uses default results path

    # Or from CLI:
    python -m visualization.run
"""

from .style import FIGURES_DIR, color_for, label_for, marker_for
from .figures import generate_all, load_all

__all__ = ["generate_all", "load_all", "FIGURES_DIR", "color_for", "label_for", "marker_for"]
