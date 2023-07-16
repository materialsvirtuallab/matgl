"""Implementations of math functions."""

from __future__ import annotations

import os
from functools import lru_cache
from math import pi, sqrt

import numpy as np
import sympy
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn

CWD = os.path.dirname(os.path.abspath(__file__))

# Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index) root of
# order l Spherical Bessel function is the `[l, n]` entry.
SPHERICAL_BESSEL_ROOTS = torch.tensor(np.load(os.path.join(CWD, "sb_roots.npy")))


@lru_cache(maxsize=128)
def spherical_bessel_roots(max_l: int, max_n: int):
    """Calculate the spherical Bessel roots. The n-th root of the l-th
    spherical bessel function is the `[l, n]` entry of the return matrix.
    The calculation is based on the fact that the n-root for l-th
    spherical Bessel function `j_l`, i.e., `z_{j, n}` is in the range
    `[z_{j-1,n}, z_{j-1, n+1}]`. On the other hand we know precisely the
    roots for j0, i.e., sinc(x).

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


def _block_repeat(array, block_size, repeats):
    col_index = torch.arange(array.size()[1])
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start : start + b], [repeats[i]]))
        start += b
    indices = torch.cat(indices, axis=0)
    return torch.index_select(array, 1, indices.to(array.device))


@lru_cache(maxsize=128)
def _get_lambda_func(max_n, cutoff: float = 5.0):
    r = sympy.symbols("r")
    en = [i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1) for i in range(max_n)]

    dn = [1.0]
    for i in range(1, max_n):
        dn_value = 1 - en[i] / dn[-1]
        dn.append(dn_value)

    fnr = [
        (-1) ** i
        * sqrt(2.0)
        * pi
        / cutoff**1.5
        * (i + 1)
        * (i + 2)
        / sympy.sqrt(1.0 * (i + 1) ** 2 + (i + 2) ** 2)
        * (
            sympy.sin(r * (i + 1) * pi / cutoff) / (r * (i + 1) * pi / cutoff)
            + sympy.sin(r * (i + 2) * pi / cutoff) / (r * (i + 2) * pi / cutoff)
        )
        for i in range(max_n)
    ]

    gnr = [fnr[0]]
    for i in range(1, max_n):
        gnr_value = 1 / sympy.sqrt(dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1])
        gnr.append(gnr_value)
    return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]


def get_segment_indices_from_n(ns):
    """Get segment indices from number array. For example if
    ns = [2, 3], then the function will return [0, 0, 1, 1, 1].

    Args:
        ns: torch.Tensor, the number of atoms/bonds array

    Returns:
        object:

    Returns: segment indices tensor
    """
    B = ns
    A = torch.arange(B.size(dim=0)).to(B.device)
    return A.repeat_interleave(B, dim=0)


def get_range_indices_from_n(ns):
    """Give ns = [2, 3], return [0, 1, 0, 1, 2].

    Args:
        ns: torch.Tensor, the number of atoms/bonds array

    Returns: range indices
    """
    max_n = torch.max(ns)
    n = ns.size(dim=0)
    n_range = torch.arange(max_n)
    matrix = n_range.tile(
        [n, 1],
    )
    mask = torch.arange(max_n)[None, :] < ns[:, None]

    #    return matrix[mask]
    return torch.masked_select(matrix, mask)


def repeat_with_n(ns, n):
    """Repeat the first dimension according to n array.

    Args:
        ns (torch.tensor): tensor
        n (torch.tensor): a list of replications

    Returns: repeated tensor

    """
    return torch.repeat_interleave(ns, n, dim=0)


def broadcast_states_to_bonds(g, state_feat):
    """Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate].

    Args:
        g: DGL graph
        state_feat: state_feature

    Returns: broadcasted state attributes
    """
    return state_feat.repeat((g.num_edges(), 1))


def broadcast_states_to_atoms(g, state_feat):
    """Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate].

    Args:
        g: DGL graph
        state_feat: state_feature

    Returns: broadcasted state attributes

    """
    return state_feat.repeat((g.num_nodes(), 1))


def scatter_sum(input_tensor: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int) -> torch.Tensor:
    """Scatter sum operation along the specified dimension. Modified from the
    torch_scatter library (https://github.com/rusty1s/pytorch_scatter).

    Args:
        input_tensor (torch.Tensor): The input tensor to be scattered.
        segment_ids (torch.Tensor): Segment ID for each element in the input tensor.
        num_segments (int): The number of segments.
        dim (int): The dimension along which the scatter sum operation is performed (default: -1).

    Returns:
        resulting tensor
    """
    segment_ids = broadcast(segment_ids, input_tensor, dim)
    size = list(input_tensor.size())
    if segment_ids.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = num_segments
    output = torch.zeros(size, dtype=input_tensor.dtype, device=input_tensor.device)
    return output.scatter_add_(dim, segment_ids, input_tensor)


def unsorted_segment_fraction(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int):
    """Segment fraction
    Args:
        data (torch.tensor): original data
        segment_ids (torch.tensor): segment ids
        num_segments (int): number of segments
    Returns:
        data (torch.tensor): data after fraction.
    """
    segment_sum = scatter_sum(input_tensor=data, segment_ids=segment_ids, dim=0, num_segments=num_segments)
    sums = torch.gather(segment_sum, 0, segment_ids)
    return torch.div(data, sums)


def broadcast(input_tensor: torch.Tensor, target_tensor: torch.Tensor, dim: int):
    """Broadcast input tensor along a given dimension to match the shape of the target tensor.
    Modified from torch_scatter library (https://github.com/rusty1s/pytorch_scatter).

    Args:
        input_tensor: The tensor to broadcast.
        target_tensor: The tensor whose shape to match.
        dim: The dimension along which to broadcast.

    Returns:
        resulting input tensor after broadcasting
    """
    if input_tensor.dim() == 1:
        for _ in range(0, dim):
            input_tensor = input_tensor.unsqueeze(0)
    for _ in range(input_tensor.dim(), target_tensor.dim()):
        input_tensor = input_tensor.unsqueeze(-1)
    target_shape = list(target_tensor.shape)
    target_shape[dim] = input_tensor.shape[dim]
    return input_tensor.expand(target_shape)
