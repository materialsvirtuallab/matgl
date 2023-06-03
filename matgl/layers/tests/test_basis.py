from __future__ import annotations

import unittest

import numpy as np
import torch

from matgl.utils.maths import (
    GaussianExpansion,
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
)


class TestGaussianAndSphericalBesselFuction(unittest.TestCase):
    def test_gaussian(self):
        r = torch.linspace(1.0, 5.0, 11)
        rbf_gaussian = GaussianExpansion(initial=0.0, final=5.0, num_centers=10, width=0.5)
        rbf = rbf_gaussian(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 10]

    def test_sphericalbesselfunction(self):
        r = torch.linspace(1.0, 5.0, 11)
        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=False)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 9]

        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=True)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 3]

    def test_sphericalbesselfunction_smooth(self):
        r = torch.linspace(1.0, 5.0, 11)
        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=False)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 9]

        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=True)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 3]

    def test_spherical_harmonic_function(self):
        theta = torch.linspace(-1, 1, 10)
        phi = torch.linspace(0, 2 * np.pi, 10)
        abf_sb = SphericalHarmonicsFunction(max_l=3, use_phi=True)
        abf = abf_sb(theta, phi)
        assert [abf.size(dim=0), abf.size(dim=1)] == [10, 9]


if __name__ == "__main__":
    unittest.main()
