"""
Generate bond features based on spherical bessel functions or gaussian expansion
"""
from __future__ import annotations

import torch
import torch.nn as nn

from matgl.utils.maths import GaussianExpansion, SphericalBesselFunction


class BondExpansion(nn.Module):
    """
    Expand pair distances into a set of spherical bessel or gaussian functions.
    """

    def __init__(
        self,
        max_l: int = 3,
        max_n: int = 3,
        cutoff: float = 5.0,
        rbf_type: str = "SphericalBessel",
        smooth: bool = False,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
        device="cpu",
    ) -> None:
        """
        Parameters:
        ----------
        max_l (int): order of angular part
        max_n (int): order of radial part
        cutoff (float): cutoff radius
        rbf_type (str): type of radial basis function .i.e. either "SphericalBessel" or 'Gaussian'
        smooth (bool): whether apply the smooth version of spherical bessel functions or not
        initial (float): initial point for gaussian expansion
        final (float): final point for gaussian expansion
        width (float): width of gaussian function
        device (torch.device): cpu, cuda, etc...
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

        if rbf_type == "SphericalBessel":
            self.rbf = SphericalBesselFunction(max_l, max_n, cutoff, smooth, device)  # type: ignore
        elif rbf_type == "Gaussian":
            self.rbf = GaussianExpansion(initial, final, num_centers, width)  # type: ignore
        else:
            raise Exception("undefined rbf_type, please use SphericalBessel or Gaussian instead.")

    def forward(self, bond_dist: torch.tensor):
        """
        Forward

        Args:
        bond_dist: Bond distance

        Return:
        bond_basis: Radial basis functions
        """
        bond_basis = self.rbf(bond_dist)
        return bond_basis
