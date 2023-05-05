from __future__ import annotations

import unittest

import numpy as np
import torch

from matgl.utils.cutoff import cosine_cutoff, polynomial_cutoff


class TestCutoff(unittest.TestCase):
    def test_polynomial(self):
        r = torch.linspace(1.0, 5.0, 11)
        three_body_cutoff = 4.0
        three_cutoff = polynomial_cutoff(r, three_body_cutoff)
        np.testing.assert_almost_equal(
            three_cutoff.tolist(),
            torch.tensor(
                [0.8965, 0.7648, 0.5931, 0.4069, 0.2352, 0.1035, 0.0266, 0.0012, 0.0000, 0.0000, 0.0000]
            ).tolist(),
            decimal=4,
        )

    def test_cosine(self):
        r = torch.linspace(1.0, 5.0, 11)
        three_body_cutoff = 4.0
        three_cutoff = cosine_cutoff(r, three_body_cutoff)
        np.testing.assert_almost_equal(
            three_cutoff.tolist(),
            torch.tensor([0.8536, 0.7270, 0.5782, 0.4218, 0.2730, 0.1464, 0.0545, 0.0062, 0.0000, 0.0000, 0.0000]),
            decimal=4,
        )


if __name__ == "__main__":
    unittest.main()
