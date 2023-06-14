from __future__ import annotations

from math import sqrt, pi

import sympy
import torch


class SphericalHarmonicsFunction:
    """Spherical Harmonics function."""

    def __init__(self, max_l: int, use_phi: bool = True):
        """Args:
        max_l: int, max l (excluding l)
        use_phi: bool, whether to use the polar angle. If not,
        the function will compute `Y_l^0`.
        """
        self.max_l = max_l
        self.use_phi = use_phi
        funcs = []
        theta, phi = sympy.symbols("theta phi")
        for lval in range(self.max_l):
            m_list = range(-lval, lval + 1) if self.use_phi else [0]  # type: ignore
            for m in m_list:
                func = sympy.functions.special.spherical_harmonics.Znm(lval, m, theta, phi).expand(func=True)
                funcs.append(func)
        # replace all theta with cos(theta)
        costheta = sympy.symbols("costheta")
        funcs = [i.subs({theta: sympy.acos(costheta)}) for i in funcs]
        self.orig_funcs = [sympy.simplify(i).evalf() for i in funcs]
        self.funcs = [sympy.lambdify([costheta, phi], i, [{"conjugate": _conjugate}, torch]) for i in self.orig_funcs]
        self.funcs[0] = _y00

    def __call__(self, costheta, phi=None):
        """Args:
            costheta: Cosine of the azimuthal angle
            phi: torch.tensor, the polar angle.

        Returns: [n, m] spherical harmonic results, where n is the number
            of angles. The column is arranged following
            `[Y_0^0, Y_1^{-1}, Y_1^{0}, Y_1^1, Y_2^{-2}, ...]`
        """
        # costheta = torch.tensor(costheta, dtype=torch.complex64)
        # phi = torch.tensor(phi, dtype=torch.complex64)
        return torch.stack([func(costheta, phi) for func in self.funcs], axis=1)
        # results = results.type(dtype=DataType.torch_float)
        # return results


def _y00(theta, phi):
    r"""Spherical Harmonics with `l=m=0`.

    ..math::
        Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

    Args:
        theta: torch.tensor, the azimuthal angle
        phi: torch.tensor, the polar angle

    Returns: `Y_0^0` results

    """
    return 0.5 * torch.ones_like(theta) * sqrt(1.0 / pi)


def _conjugate(x):
    return torch.conj(x)
