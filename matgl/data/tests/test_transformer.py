from __future__ import annotations

import pytest
import torch

from matgl.data.transformer import Normalizer


class TestNormalizer:
    def test_transform(self):
        data = torch.randn(100) * 10 + 1
        transformer = Normalizer.from_data(data)
        scaled = transformer.transform(data)

        inverse = transformer.inverse_transform(scaled)
        assert inverse == pytest.approx(data, abs=1e-5)
