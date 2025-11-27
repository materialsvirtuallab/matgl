"""Computing various graph based operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data


def compute_pair_vector_and_distance(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate bond vectors and distances using PyTorch Geometric Data object.

    Args:
        graph: PyTorch Geometric Data object containing pos, edge_index, and pbc_offshift.

    Returns:
        Tuple containing:
        - bond_vec (torch.Tensor): Vector from source node to destination node.
        - bond_dist (torch.Tensor): Bond distance between two atoms.
    """
    # Get source and destination node indices from edge_index
    src_idx, dst_idx = graph.edge_index

    # Get positions of source and destination nodes
    src_pos = graph.pos[src_idx]
    dst_pos = graph.pos[dst_idx]

    # Apply periodic boundary condition offsets
    if hasattr(graph, "pbc_offshift") and graph.pbc_offshift is not None:
        dst_pos = dst_pos + graph.pbc_offshift

    # Compute bond vectors and distances
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def separate_node_edge_keys(graph: Data) -> tuple[list[str], list[str], list[str]]:
    """Separates keys in a PyTorch Geometric Data object into node attributes, edge attributes, and other attributes.

    Args:
        graph: PyTorch Geometric Data object.

    Returns:
        tuple: (node_keys, edge_keys, other_keys) where each is a list of attribute names.
    """
    node_keys = []
    edge_keys = []
    other_keys = []

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    for key in graph:
        value = graph[key]
        if key == "edge_index":
            other_keys.append(key)
            continue
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            first_dim = value.size(0)
            if first_dim == num_nodes:
                node_keys.append(key)
            elif first_dim == num_edges:
                edge_keys.append(key)
            else:
                other_keys.append(key)
        else:
            other_keys.append(key)

    return node_keys, edge_keys, other_keys
