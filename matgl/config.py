"""Global configuration variables for matgl."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch

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


DTYPES = {
    "float32": {"numpy": np.float32, "pt": torch.float32},
    "float16": {"numpy": np.float16, "pt": torch.float16},
    "int32": {"numpy": np.int32, "pt": torch.int32},
    "int16": {"numpy": np.int16, "pt": torch.int16},
}


class DataType:
    """Tensorflow and numpy data types. Used to choose between float16 and float32."""

    np_float = np.float32
    np_int = "int32"
    pt_float = torch.float32
    pt_int = "int32"

    @classmethod
    def set_dtype(cls, data_type: str) -> None:
        """
        Class method to set the data types
        Args:
            data_type (str): '16' or '32'.
        """
        if data_type.endswith("32"):
            float_key = "float32"
            int_key = "int32"
        elif data_type.endswith("16"):
            float_key = "float16"
            int_key = "int16"
        else:
            raise ValueError("Data type not known, choose '16' or '32'")

        cls.np_float = DTYPES[float_key]["numpy"]
        cls.pt_float = DTYPES[float_key]["pt"]
        cls.np_int = DTYPES[int_key]["numpy"]
        cls.pt_int = DTYPES[int_key]["pt"]


def set_global_dtypes(data_type) -> None:
    """Function to set the data types.

    Args:
        data_type (str): '16' or '32'
    Returns:

    """
    DataType.set_dtype(data_type)
