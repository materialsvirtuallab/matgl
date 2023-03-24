"""
Various math function implementations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from math import pi, sqrt

import numpy as np
import sympy
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.special import spherical_jn

from matgl.config import DataType

CWD = os.path.dirname(os.path.abspath(__file__))

"""
Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index) root of
order l Spherical Bessel function is the `[l, n]` entry.
"""
SPHERICAL_BESSEL_ROOTS = np.load(os.path.join(CWD, "sb_roots.npy"))


class GaussianExpansion(nn.Module):
    r"""
    Gaussian Radial Expansion.
    The bond distance is expanded to a vector of shape [m],
    where m is the number of Gaussian basis centers
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """
        Parameters
        ----------
        initial : float
                Location of initial Gaussian basis center.
        final : float
                Location of final Gaussian basis center
        number : int
                Number of Gaussian Basis functions
        width : float
                Width of Gaussian Basis functions
        """
        super().__init__()
        self.centers = np.linspace(initial, final, num_centers)
        self.centers = nn.Parameter(torch.tensor(self.centers).float(), requires_grad=False)  # type: ignore
        if width is None:
            self.width = float(1.0 / np.diff(self.centers).mean())
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(self.centers.clone().detach().float(), requires_grad=False).to(device)

    def forward(self, bond_dists):
        """Expand distances.

        Parameters
        ----------
        bond_dists :
            Bond (edge) distances between two atoms (nodes)

        Returns:
        -------
        A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return torch.exp(-self.width * (diff**2))


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

    def __init__(self, max_l: int, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False, device: str = "cpu"):
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
        self.device = torch.device(device)

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list:
        """
        Spherical basis functions based on Rayleigh formula. This function
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
        """
        Args:
            r: torch.tensor, distance tensor, 1D


        Returns: [n, max_n * max_l] spherical Bessel function results

        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    def _call_smooth_sbf(self, r):
        results = [i(r) for i in self.funcs]
        return torch.t(torch.stack(results))

    def _call_sbf(self, r):
        roots = self.zeros[: self.max_l, : self.max_n]

        results = []
        factor = torch.tensor(sqrt(2.0 / self.cutoff**3)).to(self.device)
        for i in range(self.max_l):
            root = roots[i].to(self.device)
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r[:, None] * root[None, :] / self.cutoff) * factor / torch.abs(func_add1(root[None, :]))
            )
        return torch.cat(results, axis=1)

    @staticmethod
    def rbf_j0(r, cutoff: float = 5.0, max_n: int = 3):
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
    return 0.5 * torch.ones_like(theta) * sqrt(1.0 / pi)


def _conjugate(x):
    return torch.conj(x)


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
    return torch.index_select(array, 1, indices.to(array.device))


def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool, use_smooth: bool):
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
        block_size = 2 * np.arange(max_l) + 1  # type: ignore
        # 2 * tf.range(max_l) + 1
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf.to(sbf.device), 1)
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
    dn = torch.stack(dn)  # type: ignore
    gn = [fnr[:, 0]]
    for i in range(1, max_n):
        gn.append(1 / torch.sqrt(dn[i]) * (fnr[:, i] + torch.sqrt(en[0, i] / dn[i - 1]) * gn[-1]))

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
                sympy.sin(r * (i + 1) * sympy.pi / cutoff) / (r * (i + 1) * sympy.pi / cutoff)
                + sympy.sin(r * (i + 2) * sympy.pi / cutoff) / (r * (i + 2) * sympy.pi / cutoff)
            )
        )

    gnr = [fnr[0]]
    for i in range(1, max_n):
        gnr.append(1 / sympy.sqrt(dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1]))
    return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]


def get_segment_indices_from_n(ns):
    """
    Get segment indices from number array. For example if
    ns = [2, 3], then the function will return [0, 0, 1, 1, 1]

    Args:
        ns: torch.tensor, the number of atoms/bonds array

    Returns:
        object:

    Returns: segment indices tensor
    """
    B = ns
    A = torch.arange(B.size(dim=0)).to(B.device)
    return A.repeat_interleave(B, dim=0)


def get_range_indices_from_n(ns):
    """
    Give ns = [2, 3], return [0, 1, 0, 1, 2]

    Args:
        ns: torch.tensor, the number of atoms/bonds array

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


# def unsorted_segment_softmax(data, segment_ids, num_segments, weights=None):
#    """
#    Unsorted segment softmax with optional weights
#    Args:
#        data (tf.Tensor): original data
#        segment_ids (tf.Tensor): tensor segment ids
#        num_segments (int): number of segments
#    Returns: tf.Tensor
#    """
#    if weights is None:
#        weights = torch.ones(1)
#    segment_max = scatter(data, segment_ids, dim=0, reduce="max")
#    maxes = torch.gather(segment_max, 0, segment_ids)
#    data -= maxes
#    exp = torch.exp(data) * torch.squeeze(weights)
#    softmax = torch.div(exp, torch.gather(scatter(exp, segment_ids, dim=0, reduce="sum"), 0, segment_ids))
#    return softmax


def repeat_with_n(ns, n):
    """
    Repeat the first dimension according to n array.

    Args:
        ns (torch.tensor): tensor
        n (torch.tensor): a list of replications

    Returns: repeated tensor

    """
    return torch.repeat_interleave(ns, n, dim=0)


def broadcast_states_to_bonds(g, state_feat):
    """
    Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate]

    Args:
        g: DGL graph
        state_feat: state_feature

    Returns: broadcasted state attributes

    """
    return state_feat.repeat((g.num_edges(), 1))


def broadcast_states_to_atoms(g, state_feat):
    """
    Broadcast state attributes of shape [Ns, Nstate] to
    bond attributes shape [Nb, Nstate]

    Args:
        g: DGL graph
        state_feat: state_feature

    Returns: broadcasted state attributes

    """
    return state_feat.repeat((g.num_nodes(), 1))


def scatter_sum(input_tensor: torch.tensor, segment_ids: torch.tensor, num_segments: int, dim: int) -> torch.tensor:
    """
    Scatter sum operation along the specified dimension. Modified from the
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


def unsorted_segment_fraction(data: torch.tensor, segment_ids: torch.tensor, num_segments: torch.tensor):
    """
    Segment fraction
    Args:
        data (torch.tensor): original data
        segment_ids (torch.tensor): segment ids
        num_segments (torch.tensor): number of segments
    Returns:
        data (torch.tensor): data after fraction
    """
    segment_sum = scatter_sum(input_tensor=data, segment_ids=segment_ids, dim=0, num_segments=num_segments)
    sums = torch.gather(segment_sum, 0, segment_ids)
    data = torch.div(data, sums)
    return data


def broadcast(input_tensor: torch.tensor, target_tensor: torch.tensor, dim: int):
    """
    Broadcast input tensor along a given dimension to match the shape of the target tensor.
    Modified from torch_scatter library (https://github.com/rusty1s/pytorch_scatter).
    Args:
        input_tensor: The tensor to broadcast.
        target_tensor: The tensor whose shape to match.
        dim: The dimension along which to broadcast.
    Returns:
        resulting inout tensor after broadcasting
    """
    if input_tensor.dim() == 1:
        for _ in range(0, dim):
            input_tensor = input_tensor.unsqueeze(0)
    for _ in range(input_tensor.dim(), target_tensor.dim()):
        input_tensor = input_tensor.unsqueeze(-1)
    target_shape = list(target_tensor.shape)
    target_shape[dim] = input_tensor.shape[dim]
    input_tensor = input_tensor.expand(target_shape)
    return input_tensor
