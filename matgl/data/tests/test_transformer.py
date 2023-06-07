from __future__ import annotations

import unittest

import numpy as np
import pytest

from matgl.data.transformer import Normalizer


class NormalizerTestCase(unittest.TestCase):
    def test_transform(self):
        data = [2, 3, 4, 5, 6, 7]
        transformer = Normalizer.from_data(data)
        scaled = transformer.transform(data)

        assert scaled.mean(), pytest.approx(0)

        inverse = transformer.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(inverse, data)


if __name__ == "__main__":
    unittest.main()
