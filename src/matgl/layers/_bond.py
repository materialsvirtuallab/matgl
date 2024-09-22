"""Generate bond features based on spherical bessel functions or gaussian expansion."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from matgl.layers._basis import ExpNormalFunction, GaussianExpansion, SphericalBesselFunction


class BondExpansion(nn.Module):
    """Expand pair distances into a set of spherical bessel or gaussian functions."""

    def __init__(
        self,
        max_l: int = 3,
        max_n: int = 3,
        cutoff: float = 5.0,
        rbf_type: Literal["SphericalBessel", "Gaussian", "ExpNorm"] = "SphericalBessel",
        smooth: bool = False,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
    ) -> None:
        """
        Args:
            max_l (int): order of angular part
            max_n (int): order of radial part
            cutoff (float): cutoff radius
            rbf_type (str): type of radial basis function .i.e. either "SphericalBessel", "ExpNorm" or 'Gaussian'
            smooth (bool): whether apply the smooth version of spherical bessel functions or not
            initial (float): initial point for gaussian expansion
            final (float): final point for gaussian expansion
            num_centers (int): Number of centers for gaussian expansion.
            width (float): width of gaussian function.
        """
        super().__init__()

        self.max_n = max_n
        self.cutoff = cutoff
        self.max_l = max_l
        self.smooth = smooth
        self.num_centers = num_centers
        self.width = width
        self.initial = initial
        self.final = final
        self.rbf_type = rbf_type

        if rbf_type.lower() == "sphericalbessel":
            self.rbf = SphericalBesselFunction(max_l, max_n, cutoff, smooth)  # type: ignore
        elif rbf_type.lower() == "gaussian":
            self.rbf = GaussianExpansion(initial, final, num_centers, width)  # type: ignore
        elif rbf_type.lower() == "expnorm":
            self.rbf = ExpNormalFunction(cutoff, num_centers, True)
        else:
            raise ValueError("Undefined rbf_type, please use SphericalBessel or Gaussian instead.")

    def forward(self, bond_dist: torch.Tensor):
        """Forward.

        Args:
        bond_dist: Bond distance

        Return:
        bond_basis: Radial basis functions
        """
        bond_basis = self.rbf(bond_dist)
        return bond_basis
