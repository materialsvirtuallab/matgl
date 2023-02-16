"""
Cutoff functions for constructing M3GNet potentials.
"""

from __future__ import annotations

from math import pi

import torch


def polynomial_cutoff(r, cutoff: float):
    """
    Polynomial cutoff function
    Args:
        r (torch.tensor): radius distance tensor
        cutoff (float): cutoff distance

    Returns: polynomial cutoff functions

    """
    ratio = r / cutoff
    return torch.where(r <= cutoff, 1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3, 0.0)


def cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Cosine cutoff function
    Args:
        r (torch.tensor): radius distance tensor
        cutoff (float): cutoff distance

    Returns: cosine cutoff functions

    """
    return torch.where(r <= cutoff, 0.5 * (torch.cos(pi * r / cutoff) + 1), 0.0)
