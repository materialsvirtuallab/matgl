from __future__ import annotations

from math import sqrt, pi

import torch


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
