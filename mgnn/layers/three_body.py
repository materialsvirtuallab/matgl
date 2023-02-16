"""
Three-Body interaction implementations.
"""

from typing import List

import torch
import torch.nn as nn
from torch_scatter import scatter

from mgnn.utils.maths import (
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    combine_sbf_shf,
    get_segment_indices_from_n,
)


class SphericalBesselWithHarmonics(nn.Module):
    """
    Expansion of basis using Spherical Bessel and Harmonics
    """

    def __init__(self, max_n, max_l, cutoff, use_smooth, use_phi):
        """
        :param max_n: Degree of radial basis functions
        :param max_l: Degree of angular basis functions
        :param cutoff: Cutoff sphere
        :param use_smooth: Whether using smooth version of SBFs or not
        :param use_phi: using phi as angular basis functions
        """
        super().__init__()

        assert max_n <= 64
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi
        self.use_smooth = use_smooth

        # retrieve formulas
        self.shf = SphericalHarmonicsFunction(self.max_l, self.use_phi)
        if self.use_smooth:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n * self.max_l, self.cutoff, self.use_smooth)
        else:
            self.sbf = SphericalBesselFunction(self.max_l, self.max_n, self.cutoff, self.use_smooth)

    def forward(self, graph, line_graph):
        sbf = self.sbf(line_graph.edata["triple_bond_lengths"])
        shf = self.shf(line_graph.edata["cos_theta"], line_graph.edata["phi"])
        return combine_sbf_shf(
            sbf, shf, max_n=self.max_n, max_l=self.max_l, use_phi=self.use_phi, use_smooth=self.use_smooth
        )


class ThreeBodyInteractions(nn.Module):
    """
    Include 3D interactions to the bond update
    """

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """
        Args:
            update_network_atom: MLP for node features in Eq.2
            update_network_bond: Gated-MLP for edge features in Eq.3
            **kwargs:
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self, graph, line_graph, three_basis: torch.tensor, three_cutoff: float, node_feat, edge_feat, **kwargs
    ) -> torch.tensor:
        """
        Args:
            graph: dgl graph
            three_basis: three body basis expansion
            three_cutoff: cutoff radius
            node_feat: node features
            edge_feat: edge features
            **kwargs:
        Returns:
        """
        end_atom_index = torch.gather(graph.edges()[1], 0, line_graph.edges()[1].to(torch.int64))
        atoms = self.update_network_atom(node_feat)
        end_atom_index = torch.unsqueeze(end_atom_index, 1)
        atoms = torch.squeeze(atoms[end_atom_index])
        basis = three_basis * atoms
        torch.tensor(graph.num_edges())
        three_cutoff = torch.unsqueeze(three_cutoff, dim=1)
        weights = torch.reshape(three_cutoff[torch.stack(list(line_graph.edges()), dim=1).to(torch.int64)], (-1, 2))
        weights = torch.prod(weights, axis=-1)
        weights = basis * weights[:, None]
        new_bonds = scatter(
            basis.to(torch.float32),
            get_segment_indices_from_n(line_graph.ndata["n_triple_ij"]),
            dim=0,
            reduce="sum",
            dim_size=graph.num_edges(),
        )
        edge_feat_updated = edge_feat + self.update_network_bond(new_bonds)
        return edge_feat_updated
