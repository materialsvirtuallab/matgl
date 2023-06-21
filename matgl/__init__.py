"""MatGL (Materials Graph Library) is a graph deep learning library for materials science."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import clear_cache
from .utils.io import get_available_pretrained_models, load_model

try:
    __version__ = version("matgl")
except PackageNotFoundError:
    pass  # package not installed
