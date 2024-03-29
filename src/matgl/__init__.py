"""MatGL (Materials Graph Library) is a graph deep learning library for materials science."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

from .config import clear_cache
from .utils.io import get_available_pretrained_models, load_model

try:
    __version__ = version("matgl")
except PackageNotFoundError:
    pass  # package not installed


# Default datatypes definitions

float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32


def set_default_dtype(type_: str = "float", size: int = 32):
    """
    Set the default dtype size (16, 32 or 64) for int or float used throughout matgl.

    Args:
        type_: "float" or "int"
        size: 32 or 64.
    """
    if size in (16, 32, 64):
        globals()[f"{type_}_th"] = getattr(torch, f"{type_}{size}")
        globals()[f"{type_}_np"] = getattr(np, f"{type_}{size}")
        torch.set_default_dtype(getattr(torch, f"float{size}"))
    else:
        raise ValueError("Invalid dtype size")
    if type_ == "float" and size == 16 and not torch.cuda.is_available():
        raise Exception(
            "torch.float16 is not supported for M3GNet because addmm_impl_cpu_ is not implemented"
            " for this floating precision, please use size = 32, 64 or using 'cuda' instead !!"
        )
