"""Computing various graph based operations."""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def compute_pair_vector_and_distance_pyg(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
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


def compute_theta_and_phi_pyg(graph: Data, line_graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate bond angle Theta and Phi using PyG graphs.

    Args:
        graph: PyG graph
        line_graph: PyG line graph

    Returns:
        cos_theta: torch.Tensor
        phi: torch.Tensor
        triple_bond_lengths: torch.Tensor (bond distances of destination bonds in line graph)
    """
    # Get bond vectors from the line graph nodes (which represent bonds)
    # In line graph, nodes are edges from original graph
    vec1 = line_graph.bond_vec[line_graph.edge_index[0]]  # Source bond vector
    vec2 = line_graph.bond_vec[line_graph.edge_index[1]]  # Destination bond vector

    # Compute cos(theta)
    cos_theta = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1) + 1e-8)
    cos_theta = cos_theta.clamp(min=-1 + 1e-7, max=1 - 1e-7)

    # For phi, we need the cross product and additional computation
    # Simplified version - can be extended if needed
    cross = torch.cross(vec1, vec2)
    phi = torch.atan2(torch.norm(cross, dim=1), torch.sum(vec1 * vec2, dim=1))

    # triple_bond_lengths is the bond distance of the destination bond
    triple_bond_lengths = line_graph.bond_dist[line_graph.edge_index[1]]

    return cos_theta, phi, triple_bond_lengths


def create_line_graph_pyg(
    graph: Data,
    threebody_cutoff: float,
) -> Data:
    """
    Calculate the three body indices from pair atom indices for PyG.

    Args:
        graph: PyG graph
        threebody_cutoff (float): cutoff for three-body interactions

    Returns:
        l_g: PyG Data object containing three body information from graph
    """
    import numpy as np

    edge_index = graph.edge_index
    bond_dist = graph.bond_dist

    # Filter edges within threebody cutoff
    valid_mask = bond_dist <= threebody_cutoff
    valid_edges = edge_index[:, valid_mask]
    valid_bond_dist = bond_dist[valid_mask]

    if valid_edges.size(1) == 0:
        # Return empty line graph
        empty_graph = Data(edge_index=torch.empty((2, 0), dtype=torch.long, device=graph.edge_index.device))
        empty_graph.n_triple_ij = torch.empty(0, dtype=torch.long, device=graph.edge_index.device)
        return empty_graph

    # Compute n_triple_ij: number of three-body angles for each bond
    # Count bonds per atom
    src_nodes = valid_edges[0].cpu().numpy()
    n_atoms = graph.num_nodes
    n_bond_per_atom = np.bincount(src_nodes, minlength=n_atoms)
    n_triple_ij = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)
    n_triple_ij = torch.tensor(n_triple_ij, dtype=torch.long, device=graph.edge_index.device)

    # Create line graph: nodes are edges in original graph
    # Two edges are connected if they share a common node
    num_line_nodes = valid_edges.size(1)
    line_edge_list = []

    # For each pair of edges, check if they share a node
    for i in range(num_line_nodes):
        for j in range(i + 1, num_line_nodes):
            edge_i = valid_edges[:, i]
            edge_j = valid_edges[:, j]
            # Check if edges share a node (form a triangle)
            if edge_i[0] == edge_j[0] or edge_i[0] == edge_j[1] or edge_i[1] == edge_j[0] or edge_i[1] == edge_j[1]:
                # They share a node, add edge in line graph
                line_edge_list.append([i, j])
                line_edge_list.append([j, i])  # Undirected

    if len(line_edge_list) == 0:
        line_edge_index = torch.empty((2, 0), dtype=torch.long, device=graph.edge_index.device)
    else:
        line_edge_index = torch.tensor(line_edge_list, dtype=torch.long, device=graph.edge_index.device).t()

    # Create line graph data
    line_graph = Data(edge_index=line_edge_index, num_nodes=num_line_nodes)

    # Copy bond information to line graph nodes
    if hasattr(graph, "bond_vec"):
        line_graph.bond_vec = graph.bond_vec[valid_mask]
    if hasattr(graph, "bond_dist"):
        line_graph.bond_dist = valid_bond_dist
    if hasattr(graph, "pbc_offshift"):
        line_graph.pbc_offshift = graph.pbc_offshift[valid_mask]

    # Store mapping from line graph nodes to original edge indices
    line_graph.edge_ids = torch.arange(valid_edges.size(1), device=graph.edge_index.device)
    line_graph.n_triple_ij = n_triple_ij[:num_line_nodes]

    return line_graph
