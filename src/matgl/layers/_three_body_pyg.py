"""Three-Body interaction implementations for PyG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

import matgl
from matgl.utils.maths import get_segment_indices_from_n, scatter_sum

if TYPE_CHECKING:
    from torch_geometric.data import Data


class ThreeBodyInteractions(nn.Module):
    """Include 3D interactions to the bond update for PyG."""

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """
        Initialize ThreeBodyInteractions.

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
        graph: Data,
        line_graph: Data,
        three_basis: torch.Tensor,
        three_cutoff: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """
        Forward function for ThreeBodyInteractions.

        Args:
            graph: PyG graph
            line_graph: PyG line graph
            three_basis: three body basis expansion
            three_cutoff: cutoff radius
            node_feat: node features
            edge_feat: edge features
        """
        # Get the indices of the end atoms for each bond in the line graph
        # In line graph, nodes represent edges in original graph
        # We need to get the destination node of each edge in the original graph
        line_node_indices = line_graph.edge_index[1]  # Destination nodes in line graph
        original_edge_indices = line_graph.edge_ids[line_node_indices]  # Map to original edge indices
        end_atom_indices = graph.edge_index[1][original_edge_indices].to(matgl.int_th)

        # Update node features using the atom update network
        updated_atoms = self.update_network_atom(node_feat)

        # Gather updated atom features for the end atoms
        end_atom_features = updated_atoms[end_atom_indices]

        # Compute the basis term
        basis = three_basis * end_atom_features

        # Reshape and compute weights based on the three-cutoff tensor
        three_cutoff = three_cutoff.unsqueeze(1)
        line_edge_src = line_graph.edge_index[0]
        line_edge_dst = line_graph.edge_index[1]
        edge_indices = torch.stack([line_edge_src, line_edge_dst], dim=0)
        weights = three_cutoff[edge_indices].view(-1, 2)
        weights = weights.prod(dim=-1)

        # Compute the weighted basis
        basis = basis * weights[:, None]

        # Aggregate the new bonds using scatter_sum
        # We need to map line graph edges back to original graph edges
        segment_ids = get_segment_indices_from_n(line_graph.n_triple_ij)
        new_bonds = scatter_sum(
            basis.to(matgl.float_th),
            segment_ids=segment_ids,
            num_segments=graph.num_edges,
            dim=0,
        )

        # If no new bonds are generated, return the original edge features
        if new_bonds.shape[0] == 0:
            return edge_feat

        # Update edge features using the bond update network
        updated_edge_feat = edge_feat + self.update_network_bond(new_bonds)

        return updated_edge_feat

