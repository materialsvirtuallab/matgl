"""Global configuration variables for matgl."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pymatgen.core.periodic_table import Element

# Default set of elements supported by universal matgl models. Excludes radioactive and most artificial elements.
DEFAULT_ELEMENTS = tuple(el.symbol for el in Element if el.symbol not in ["Po", "At", "Rn", "Fr", "Ra"] and el.Z < 95)


# Default location of the cache for matgl, e.g., for storing downloaded models.
MATGL_CACHE = Path(os.path.expanduser("~")) / ".cache/matgl"
os.makedirs(MATGL_CACHE, exist_ok=True)

# Download url for pre-trained models.
PRETRAINED_MODELS_BASE_URL = "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/"


def clear_cache(confirm: bool = True):
    """Deletes all files in the matgl.cache. This is used to clean out downloaded models.

    Args:
        confirm: Whether to ask for confirmation. Default is True.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {MATGL_CACHE} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(MATGL_CACHE)
        except FileNotFoundError:
            print(f"matgl cache dir {MATGL_CACHE!r} not found")
