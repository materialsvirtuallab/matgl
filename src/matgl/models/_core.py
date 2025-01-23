from __future__ import annotations

import abc
from abc import ABCMeta

import torch
from torch import nn

from matgl.utils.io import IOMixIn


class MatGLModel(nn.Module, IOMixIn, metaclass=ABCMeta):
    @abc.abstractmethod
    def predict_structure(self, structure, *args, **kwargs) -> torch.Tensor:
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            *args: Any additional positional arguments.
            **kwargs: Any additional keyword arguments.

        Returns:
            output (torch.tensor): output property
        """
