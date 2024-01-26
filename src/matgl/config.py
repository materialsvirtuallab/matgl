"""Global configuration variables for matgl."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Default set of elements supported by universal matgl models.
DEFAULT_ELEMENTS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
)


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
