"""
Various math function implementations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from math import pi
from typing import List, Optional, Union

import numpy as np
import sympy

# import tensorflow as tf
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn

from math import sqrt
from megnet.config import DataType

CWD = os.path.dirname(os.path.abspath(__file__))

"""
Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index） root of
order l Spherical Bessel function is the `[l, n]` entry.
"""
SPHERICAL_BESSEL_ROOTS = np.load(os.path.join(CWD, "sb_roots.npy"))


@lru_cache(maxsize=128)
def spherical_bessel_roots(max_l: int, max_n: int):
    """
    Calculate the spherical Bessel roots. The n-th root of the l-th
    spherical bessel function is the `[l, n]` entry of the return matrix.
    The calculation is based on the fact that the n-root for l-th
    spherical Bessel function `j_l`, i.e., `z_{j, n}` is in the range
    `[z_{j-1,n}, z_{j-1, n+1}]`. On the other hand we know precisely the
    roots for j0, i.e., sinc(x)

    Args:
        max_l: max order of spherical bessel function
        max_n: max number of roots
    Returns: root matrix of size [max_l, max_n]
    """
    temp_zeros = np.arange(1, max_l + max_n + 1) * pi  # j0
    roots = [temp_zeros[:max_n]]
    for i in range(1, max_l):
        roots_temp = []
        for j in range(max_n + max_l - i):
            low = temp_zeros[j]
            high = temp_zeros[j + 1]
            root = brentq(lambda x, v: spherical_jn(v, x), low, high, (i,))
            roots_temp.append(root)
        temp_zeros = np.array(roots_temp)
        roots.append(temp_zeros[:max_n])
    return np.array(roots)


class SphericalBesselFunction:
    """
    Calculate the spherical Bessel function based on sympy + pytorch implementations
    """

    def __init__(
        self, max_l: int, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False
    ):
        """
        Args:
            max_l: int, max order (excluding l)
            max_n: int, max number of roots used in each l
            cutoff: float, cutoff radius
        """
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = cutoff
        self.smooth = smooth
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()

        self.zeros = torch.from_numpy(SPHERICAL_BESSEL_ROOTS).type(DataType.torch_float)

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> List:
        """
        Spherical basis functions based on Rayleigh formula. This function
        generates
        symbolic formula.

        Returns: list of symbolic functions

        """
        x = sympy.symbols("x")
        funcs = [
            sympy.expand_func(sympy.functions.special.bessel.jn(i, x))
            for i in range(self.max_l + 1)
        ]
        return [sympy.lambdify(x, func, torch) for func in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> List:
        return _get_lambda_func(max_n=self.max_n, cutoff=self.cutoff)

    def __call__(self, r: torch.tensor) -> torch.tensor:
        """
        Args:
            r: torch.tensor, distance tensor, 1D


        Returns: [n, max_n * max_l] spherical Bessel function results

        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r: torch.tensor) -> torch.tensor:
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r: torch.tensor) -> torch.tensor:
        roots = self.zeros[: self.max_l, : self.max_n]

        results = []
        factor = torch.tensor(sqrt(2.0 / self.cutoff**3))
        for i in range(self.max_l):
            root = roots[i]
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r[:, None] * root[None, :] / self.cutoff)
                * factor
                / torch.abs(func_add1(root[None, :]))
            )
        return torch.cat(results, axis=1)

    @staticmethod
    def rbf_j0(r: torch.tensor, cutoff: float = 5.0, max_n: int = 3) -> torch.tensor:
        """
        Spherical Bessel function of order 0, ensuring the function value
        vanishes at cutoff

        Args:
            r: torch.tensor pytorch tensors
            cutoff: float, the cutoff radius
            max_n: int max number of basis
        Returns: basis function expansion using first spherical Bessel function
        """
        n = (torch.arange(1, max_n + 1)).type(dtype=DataType.torch_float)[None, :]
        r = r[:, None]
        return sqrt(2.0 / cutoff) * torch.sin(n * pi / cutoff * r) / r


def _y00(theta, phi):
    r"""
    Spherical Harmonics with `l=m=0`

    ..math::
        Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

    Args:
        theta: torch.tensor, the azimuthal angle
        phi: torch.tensor, the polar angle

    Returns: `Y_0^0` results

    """
    dtype = theta.dtype
    return 0.5 * torch.ones_like(theta) * sqrt(1.0 / pi)


def _conjugate(x):
    return torch.conj(x)


class Gaussian:
    """
    Gaussian expansion function
    """

    def __init__(
        self, centers: Union[torch.tensor, np.ndarray], width: float, **kwargs
    ):
        """
        Args:
            centers (torch.tensor or np.ndarray): Gaussian centers for the
                expansion
            width (float): Gaussian width
            **kwargs:
        """
        self.centers = torch.tensor(centers)
        self.width = width

    def __call__(self, r: torch.tensor) -> torch.tensor:
        """
        Convert the radial distances into Gaussian functions
        Args:
            r (torch.tensor): radial distances
        Returns: Gaussian expanded vectors

        """
        r = torch.tensor(r)
        return torch.exp(-((r[:, None] - self.centers[None, :]) ** 2) / self.width**2)


class SphericalHarmonicsFunction:
    """
    Spherical Harmonics function
    """

    def __init__(self, max_l: int, use_phi: bool = True):
        """
        Args:
            max_l: int, max l (excluding l)
            use_phi: bool, whether to use the polar angle. If not,
                the function will compute `Y_l^0`
        """
        self.max_l = max_l
        self.use_phi = use_phi
        self.funcs = self._calculate_symbolic_funcs()

    def _calculate_symbolic_funcs(self):
        funcs = []
        theta, phi = sympy.symbols("theta phi")
        for lval in range(self.max_l):
            if self.use_phi:
                m_list = range(-lval, lval + 1)
            else:
                m_list = [0]
            for m in m_list:
                func = sympy.functions.special.spherical_harmonics.Znm(
                    lval, m, theta, phi
                ).expand(func=True)
                funcs.append(func)
        # replace all theta with cos(theta)
        costheta = sympy.symbols("costheta")
        funcs = [i.subs({theta: sympy.acos(costheta)}) for i in funcs]
        self.orig_funcs = [sympy.simplify(i).evalf() for i in funcs]
        results = [
            sympy.lambdify([costheta, phi], i, [{"conjugate": _conjugate}, torch])
            for i in self.orig_funcs
        ]
        results[0] = _y00
        return results

    def __call__(self, costheta, phi: Optional[torch.tensor] = None):
        """
        Args:
            theta: torch.tensor, the azimuthal angle
            phi: torch.tensor, the polar angle

        Returns: [n, m] spherical harmonic results, where n is the number
            of angles. The column is arranged following
            `[Y_0^0, Y_1^{-1}, Y_1^{0}, Y_1^1, Y_2^{-2}, ...]`
        """
        costheta = torch.tensor(costheta, dtype=torch.complex64)
        phi = torch.tensor(phi, dtype=torch.complex64)
        results = torch.stack([func(costheta, phi) for func in self.funcs], axis=1)
        results = results.type(dtype=DataType.torch_float)
        return results


def _block_repeat(array, block_size, repeats):
    col_index = torch.arange(array.size()[1])
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start : start + b], [repeats[i]]))
        start += b
    indices = torch.cat(indices, axis=0)
    return torch.index_select(array, 1, indices)


def combine_sbf_shf(
    sbf: torch.tensor, shf: torch.tensor, max_n: int, max_l: int, use_phi: bool
):
    """
    Combine the spherical Bessel function and the spherical Harmonics function
    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.tensor spherical bessel function results
        shf: torch.tensor spherical harmonics function results
        max_n: int, max number of n
        max_l: int, max number of l
        use_phi: whether to use phi
    Returns:
    """
    if sbf.size()[0] == 0:
        return sbf

    if not use_phi:
        repeats_sbf = torch.tensor([1] * max_l * max_n)
        block_size = [1] * max_l
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ...]
        repeats_sbf = torch.tensor(np.repeat(2 * np.arange(max_l) + 1, repeats=max_n))
        # tf.repeat(2 * tf.range(max_l) + 1, repeats=max_n)
        block_size = 2 * np.arange(max_l) + 1
        # 2 * tf.range(max_l) + 1
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf, block_size=block_size, repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])


def _sinc(x):
    return torch.sin(x) / x


def spherical_bessel_smooth(r, cutoff: float = 5.0, max_n: int = 10):
    """
    This is an orthogonal basis with first
    and second derivative at the cutoff
    equals to zero. The function was derived from the order 0 spherical Bessel
    function, and was expanded by the different zero roots

    Ref:
        https://arxiv.org/pdf/1907.02374.pdf

    Args:
        r: torch.tensor distance tensor
        cutoff: float, cutoff radius
        max_n: int, max number of basis, expanded by the zero roots

    Returns: expanded spherical harmonics with derivatives smooth at boundary

    """
    n = torch.arange(max_n).type(dtype=DataType.torch_float)[None, :]
    r = r[:, None]
    fnr = (
        (-1) ** n
        * sqrt(2.0)
        * pi
        / cutoff**1.5
        * (n + 1)
        * (n + 2)
        / torch.sqrt(2 * n**2 + 6 * n + 5)
        * (_sinc(r * (n + 1) * pi / cutoff) + _sinc(r * (n + 2) * pi / cutoff))
    )
    en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
    dn = [torch.tensor(1.0)]
    for i in range(1, max_n):
        dn.append(1 - en[0, i] / dn[-1])
    dn = torch.stack(dn)
    gn = [fnr[:, 0]]
    for i in range(1, max_n):
        gn.append(
            1
            / torch.sqrt(dn[i])
            * (fnr[:, i] + torch.sqrt(en[0, i] / dn[i - 1]) * gn[-1])
        )

    return torch.t(torch.stack(gn))


@lru_cache(maxsize=128)
def _get_lambda_func(max_n, cutoff: float = 5.0):
    r = sympy.symbols("r")
    d0 = 1.0
    en = []
    for i in range(max_n):
        en.append(i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1))

    dn = [d0]
    for i in range(1, max_n):
        dn.append(1 - en[i] / dn[-1])

    fnr = []
    for i in range(max_n):
        fnr.append(
            (-1) ** i
            * sympy.sqrt(2.0)
            * sympy.pi
            / cutoff**1.5
            * (i + 1)
            * (i + 2)
            / sympy.sqrt(1.0 * (i + 1) ** 2 + (i + 2) ** 2)
            * (
                sympy.sin(r * (i + 1) * sympy.pi / cutoff)
                / (r * (i + 1) * sympy.pi / cutoff)
                + sympy.sin(r * (i + 2) * sympy.pi / cutoff)
                / (r * (i + 2) * sympy.pi / cutoff)
            )
        )

    gnr = [fnr[0]]
    for i in range(1, max_n):
        gnr.append(
            1 / sympy.sqrt(dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1])
        )
    return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]
