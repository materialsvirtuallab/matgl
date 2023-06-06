"""Global configuration variables for matgl."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

DEFAULT_MEGNet_EFORM_STD_AND_MEAN = torch.tensor([1.07266128, -1.65459073])
DEFAULT_MEGNet_MFI_BANDGAP_STD_AND_MEAN = torch.tensor([1.85749567, 1.57419002])
DEFAULT_M3GNet_EFS_STD_AND_MEAN = torch.tensor([0.3895997944010980, 0.0])
DEFAULT_M3GNet_EFS_ELEMENTS_REFS = torch.tensor(
    [
        -3.47784146,
        -0.54570321,
        -3.48468882,
        -4.78514743,
        -8.02427714,
        -8.41993537,
        -7.74294567,
        -7.38535712,
        -4.94575169,
        -0.03200176,
        -2.52811010,
        -2.15635732,
        -5.19155831,
        -8.04530034,
        -6.92397975,
        -4.64594672,
        -3.03763270,
        -0.06279714,
        -2.47158168,
        -4.84934248,
        -8.01304186,
        -11.44988214,
        -8.90368996,
        -8.48422385,
        -8.14817511,
        -6.57528915,
        -5.24319289,
        -4.51253748,
        -3.26041298,
        -1.33942309,
        -3.59415256,
        -4.67121237,
        -4.20286103,
        -3.87766930,
        -2.83668019,
        6.39608044,
        -2.38081372,
        -4.36853011,
        -10.28721538,
        -11.68697132,
        -11.85485114,
        -8.79129339,
        -9.47414310,
        -7.61327539,
        -6.44628915,
        -4.99149808,
        -1.87163644,
        -0.65284885,
        -2.75935806,
        -3.70761812,
        -3.34189521,
        -2.80379554,
        -1.94565607,
        10.20523948,
        -2.56766670,
        -4.91659537,
        -8.87867873,
        -9.08253098,
        -7.86708414,
        -8.11366824,
        -6.29536417,
        -8.32632305,
        -13.32304199,
        -17.69382015,
        -7.56847658,
        -8.19099326,
        -8.28439331,
        -7.31815597,
        -8.28897100,
        -3.45119330,
        -7.63744229,
        -12.74151543,
        -13.47056486,
        -9.42626001,
        -11.66913871,
        -9.78914553,
        -7.84288383,
        -5.45528805,
        -2.65382941,
        0.30860656,
        -2.29307233,
        -3.52921737,
        -3.19469432,
        -5.50014363,
        -10.26884564,
        -11.29422408,
        -14.30537449,
        -14.64525690,
        -15.49324206,
    ]
)

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

MATGL_CACHE = Path(os.path.expanduser("~")) / ".matgl"
PRETRAINED_MODELS_BASE_URL = "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/"


class DataType:
    """
    Tensorflow and numpy data types. Used to choose between float16 and float32
    """

    # np_float = tf.keras.mixed_precision.global_policy().compute_dtype
    np_float = "float32"
    np_int = "int32"
    # torch_float = tf.keras.mixed_precision.global_policy().compute_dtype
    torch_float = torch.float32
    torch_int = torch.int32

    @classmethod
    def set_dtype(cls, data_type: str) -> None:
        """
        Class method to set the data types
        Args:
            data_type (str): '16' or '32'
        """
        if data_type.endswith("32"):
            float_key = "float32"
            int_key = "int32"
        elif data_type.endswith("16"):
            float_key = "float16"
            int_key = "int16"
        else:
            raise ValueError("Data type not known, choose '16' or '32'")

        cls.np_float = DTYPES[float_key]["numpy"]  # type: ignore
        cls.torch_float = DTYPES[float_key]["torch"]  # type: ignore
        cls.np_int = DTYPES[int_key]["numpy"]  # type: ignore
        cls.torch_int = DTYPES[int_key]["torch"]  # type: ignore


def set_global_dtypes(data_type) -> None:
    """
    Function to set the data types
    Args:
        data_type (str): '16' or '32'
    Returns:
    """
    DataType.set_dtype(data_type)
