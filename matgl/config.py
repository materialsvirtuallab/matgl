"""Data types"""
from __future__ import annotations

import numpy as np

# import tensorflow as tf
import torch

DTYPES = {
    "float32": {"numpy": np.float32, "torch": torch.float32},
    "float16": {"numpy": np.float16, "torch": torch.float16},
    "int32": {"numpy": np.int32, "torch": torch.int32},
    "int16": {"numpy": np.int16, "torch": torch.int16},
}


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
