"""
This package implements graph networks for materials science.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import clear_cache
from .utils.io import load_model

try:
    __version__ = version("matgl")
except PackageNotFoundError:
    pass  # package not installed
