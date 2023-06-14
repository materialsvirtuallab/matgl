from __future__ import annotations


import pytest
import torch.nn

from matgl.data.transformer import Normalizer
from matgl.models._wrappers import TransformedTargetModel


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return torch.ones(1)


class TestTransformedTargetModel:
    def test_forward(self):
        model = TransformedTargetModel(DummyModel(), Normalizer(1, 2))
        assert float(model.forward()), pytest.approx(3)
