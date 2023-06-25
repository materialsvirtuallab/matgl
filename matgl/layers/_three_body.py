"""Three-Body interaction implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from matgl.utils.maths import _block_repeat, get_segment_indices_from_n, scatter_sum

if TYPE_CHECKING:
    import dgl


class ThreeBodyInteractions(nn.Module):
    """Include 3D interactions to the bond update."""

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """Init ThreeBodyInteractions.

        Args:
            update_network_atom: MLP for node features in Eq.2
            update_network_bond: Gated-MLP for edge features in Eq.3
            **kwargs: Kwargs pass-through to nn.Module.__init__().
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        graph: dgl.DGLGraph,
        line_graph: dgl.DGLGraph,
        three_basis: torch.Tensor,
        three_cutoff: float,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """
        Forward function for ThreeBodyInteractions.

        Args:
            graph: dgl graph
            line_graph: line graph.
            three_basis: three body basis expansion
            three_cutoff: cutoff radius
            node_feat: node features
            edge_feat: edge features.
        """
        end_atom_index = torch.gather(graph.edges()[1], 0, line_graph.edges()[1].to(torch.int64))
        atoms = self.update_network_atom(node_feat)
        end_atom_index = torch.unsqueeze(end_atom_index, 1)
        atoms = torch.squeeze(atoms[end_atom_index])
        basis = three_basis * atoms
        three_cutoff = torch.unsqueeze(three_cutoff, dim=1)  # type: ignore
        weights = torch.reshape(
            three_cutoff[torch.stack(list(line_graph.edges()), dim=1).to(torch.int64)], (-1, 2)  # type: ignore
        )
        weights = torch.prod(weights, axis=-1)  # type: ignore
        basis = basis * weights[:, None]
        new_bonds = scatter_sum(
            basis.to(torch.float32),
            segment_ids=get_segment_indices_from_n(line_graph.ndata["n_triple_ij"]),
            num_segments=graph.num_edges(),
            dim=0,
        )
        if not new_bonds.data.shape[0]:
            return edge_feat
        edge_feat_updated = edge_feat + self.update_network_bond(new_bonds)
        return edge_feat_updated


def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool):
    """Combine the spherical Bessel function and the spherical Harmonics function.

    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.Tensor spherical bessel function results
        shf: torch.Tensor spherical harmonics function results
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
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf, block_size=block_size, repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])
