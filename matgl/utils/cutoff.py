"""Cutoff functions for constructing M3GNet potentials."""

from __future__ import annotations

from math import pi

import torch


def polynomial_cutoff(r: torch.tensor, cutoff: float, exponent: int = 3) -> torch.tensor:
    """Envelope polynomial function that ensures a smooth cutoff.

    Ensures first and second derivative vanish at cuttoff. As described in:
        https://arxiv.org/abs/2003.03123

    Args:
        r (torch.tensor): radius distance tensor
        cutoff (float): cutoff distance.
        exponent (int): minimum exponent of the polynomial. Default is 5.
            The polynomial will include terms of order exponent, exponent + 1, exponent + 2.

    Returns: polynomial cutoff function
    """
    a = -(exponent + 1) * (exponent + 2) / 2
    b = exponent * (exponent + 2)
    c = -exponent * (exponent + 1) / 2
    ratio = r / cutoff
    poly_envelope = 1 + a * ratio**exponent + b * ratio ** (exponent + 1) + c * ratio ** (exponent + 2)

    return torch.where(r <= cutoff, poly_envelope, 0.0)


def cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff function
    Args:
        r (torch.tensor): radius distance tensor
        cutoff (float): cutoff distance.

    Returns: cosine cutoff functions

    """
    return torch.where(r <= cutoff, 0.5 * (torch.cos(pi * r / cutoff) + 1), 0.0)
