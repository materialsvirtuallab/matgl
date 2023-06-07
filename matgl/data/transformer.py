"""
Module implementing various data transformers.
"""
from __future__ import annotations

import numpy as np


class Normalizer:
    """
    Performs a scaling of the data by centering to the mean and dividing by the standard deviation.
    """

    def __init__(self, mean: float, std: float):
        """
        Args:
            mean: Mean of the data
            std: Standard deviation of the data
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Scale the data.

        Args:
            data: Input data

        Returns:
            Scaled data
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Invert the scaling

        Args:
            data: Scaled data

        Returns:
            Unscaled data
        """
        return data * self.std + self.mean

    def __repr__(self):
        return f"Normalizer: Mean={self.mean}, Std: {self.std}"

    @classmethod
    def from_data(cls, data):
        """
        Create Normalizer from data
        Args:
            data: Input data

        Returns:
            Normalizer
        """
        return Normalizer(np.mean(data), np.std(data))
