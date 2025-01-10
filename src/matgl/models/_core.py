from __future__ import annotations

import abc
from abc import ABCMeta

from torch import nn

from matgl.utils.io import IOMixIn


class MatGLModel(nn.Module, IOMixIn, metaclass=ABCMeta):
    @abc.abstractmethod
    def predict_structure(self, structure, *args, **kwargs):
        pass
