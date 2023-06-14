from __future__ import annotations

from functools import lru_cache
from math import sqrt, pi

import sympy
import torch
from torch import nn

from matgl.utils.maths import _get_lambda_func, SPHERICAL_BESSEL_ROOTS


class GaussianExpansion(nn.Module):
    r"""Gaussian Radial Expansion.
    The bond distance is expanded to a vector of shape [m],
    where m is the number of Gaussian basis centers.
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """Args:
        initial : float
                Location of initial Gaussian basis center.
        final : float
                Location of final Gaussian basis center
        number : int
                Number of Gaussian Basis functions
        width : float
                Width of Gaussian Basis functions.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(initial, final, num_centers), requires_grad=False)  # type: ignore
        if width is None:
            self.width = 1.0 / torch.diff(self.centers).mean()
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.centers = nn.Parameter(self.centers, requires_grad=False)

    def forward(self, bond_dists):
        """Expand distances.

        Args:
            bond_dists :
                Bond (edge) distances between two atoms (nodes)

        Returns:
            A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return torch.exp(-self.width * (diff**2))


class SphericalBesselFunction:
    """Calculate the spherical Bessel function based on sympy + pytorch implementations."""

    def __init__(self, max_l: int, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False):
        """Args:
        max_l: int, max order (excluding l)
        max_n: int, max number of roots used in each l
        cutoff: float, cutoff radius
        smooth: Whether to smooth the function.
        """
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = cutoff
        self.smooth = smooth
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list:
        """Spherical basis functions based on Rayleigh formula. This function
        generates
        symbolic formula.

        Returns: list of symbolic functions

        """
        x = sympy.symbols("x")
        funcs = [sympy.expand_func(sympy.functions.special.bessel.jn(i, x)) for i in range(self.max_l + 1)]
        return [sympy.lambdify(x, func, torch) for func in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> list:
        return _get_lambda_func(max_n=self.max_n, cutoff=self.cutoff)

    def __call__(self, r):
        """Args:
            r: torch.tensor, distance tensor, 1D.


        Returns: [n, max_n * max_l] spherical Bessel function results

        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r):
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r):
        roots = SPHERICAL_BESSEL_ROOTS[: self.max_l, : self.max_n]

        results = []
        factor = torch.tensor(sqrt(2.0 / self.cutoff**3))
        for i in range(self.max_l):
            root = roots[i]
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r[:, None] * root[None, :] / self.cutoff) * factor / torch.abs(func_add1(root[None, :]))
            )
        return torch.cat(results, axis=1)

    @staticmethod
    def rbf_j0(r, cutoff: float = 5.0, max_n: int = 3):
        """Spherical Bessel function of order 0, ensuring the function value
        vanishes at cutoff.

        Args:
            r: torch.tensor pytorch tensors
            cutoff: float, the cutoff radius
            max_n: int max number of basis
        Returns: basis function expansion using first spherical Bessel function
        """
        n = (torch.arange(1, max_n + 1)).type(dtype=torch.float32)[None, :]
        r = r[:, None]
        return sqrt(2.0 / cutoff) * torch.sin(n * pi / cutoff * r) / r
