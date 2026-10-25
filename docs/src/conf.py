"""Sphinx configuration for mpi4py-simulator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

project = "mpi4py-simulator"
author = "Moritz Sallermann"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"{project} {release}"
