"""Global configuration variables for matgl."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

"""
Default set of elements supported by universal matgl models.
"""
DEFAULT_ELEMENT_TYPES = (
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

"""
Default location of the cache for matgl, e.g., for storing downloaded models.
"""
MATGL_CACHE = Path(os.path.expanduser("~")) / ".matgl"

"""
Download url for pre-trained models.
"""
PRETRAINED_MODELS_BASE_URL = "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/"

"""
This is an int representing a model version. It is mainly for detecting and warning about the use of old pre-trained
models. This version number is different from the code version number because it depends on whether
backward-incompatible architectural changes are made (which hopefully should be less often than regular code changes).
"""
MODEL_VERSION = 1


def clear_cache():
    """
    Deletes all files in the matgl.cache. This is used to clean out downloaded models.
    """
    r = input(f"Do you really want to delete everything in {MATGL_CACHE} (y|n)? ")
    if r.lower() == "y":
        shutil.rmtree(MATGL_CACHE)
