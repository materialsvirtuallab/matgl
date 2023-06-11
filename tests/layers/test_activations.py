from __future__ import annotations

import unittest

import numpy as np
import torch

from matgl.layers._activations import SoftExponential, SoftPlus2


class TestActivations(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.act1 = SoftPlus2()
        cls.act2 = SoftExponential()
        cls.act3 = SoftExponential(1.0)

        cls.x = torch.tensor([1.0, 2.0])

    def test_softplus2(self):
        out = self.act1(self.x)
        np.testing.assert_allclose(out.numpy(), np.array([0.62011445, 1.4337809]))

    def test_soft_exponential(self):
        out = self.act2(self.x)
        np.testing.assert_allclose(out.numpy(), np.array([1.0, 2.0]))
        out = self.act3(self.x)
        np.testing.assert_allclose(out.detach().numpy(), np.array([2.7182817, 7.389056]))


if __name__ == "__main__":
    unittest.main()
