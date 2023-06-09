"""Global configuration variables for matgl."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch

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

DTYPES = {
    "float32": {"numpy": np.float32, "torch": torch.float32},
    "float16": {"numpy": np.float16, "torch": torch.float16},
    "int32": {"numpy": np.int32, "torch": torch.int32},
    "int16": {"numpy": np.int16, "torch": torch.int16},
}

"""
Default location of the cache for matgl, e.g., for storing downloaded models.
"""
MATGL_CACHE = Path(os.path.expanduser("~")) / ".matgl"

"""
Download url for pre-trained models.
"""
PRETRAINED_MODELS_BASE_URL = "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/"

"""
This is an int representing a serialization scheme version. It is mainly for detecting very old pre-trained models in
cache and invalidating them. This version number is different from the code version number because it depends on
whether backward-incompatible architectural changes are made.
"""
MODEL_VERSION = 1


def clear_cache():
    """
    Deletes all files in the matgl.cache. This is used to clean out downloaded models.
    """
    r = input(f"Do you really want to delete everything in {MATGL_CACHE} (y|n)? ")
    if r.lower() == "y":
        shutil.rmtree(MATGL_CACHE)
