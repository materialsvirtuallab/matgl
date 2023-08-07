"""Computing various graph based operations."""
from __future__ import annotations

from typing import Any, Callable

import dgl
import numpy as np
import torch


def compute_3body(g: dgl.DGLGraph):
    """Calculate the three body indices from pair atom indices.

    Args:
        g: DGL graph

    Returns:
        l_g: DGL graph containing three body information from graph
        triple_bond_indices (np.ndarray): bond indices that form three-body
        n_triple_ij (np.ndarray): number of three-body angles for each bond
        n_triple_i (np.ndarray): number of three-body angles each atom
        n_triple_s (np.ndarray): number of three-body angles for each structure
    """
    n_atoms = [g.num_nodes()]
    n_atoms_total = np.sum(g.num_nodes())
    first_col = g.edges()[0].reshape(-1, 1)
    all_indices = torch.arange(n_atoms_total).reshape(1, -1)
    n_bond_per_atom = torch.count_nonzero(first_col == all_indices, dim=0)
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
    n_triple = torch.sum(n_triple_i)
    n_triple_ij = (n_bond_per_atom - 1).repeat_interleave(n_bond_per_atom)
    triple_bond_indices = torch.empty((n_triple, 2), dtype=torch.int64)  # type: ignore

    start = 0
    cs = 0
    for n in n_bond_per_atom:
        if n > 0:
            """
            triple_bond_indices is generated from all pair permutations of atom indices. The
            numpy version below does this with much greater efficiency. The equivalent slow
            code is:

            ```
            for j, k in itertools.permutations(range(n), 2):
                triple_bond_indices[index] = [start + j, start + k]
            ```
            """
            r = torch.arange(n)
            x, y = torch.meshgrid(r, r)
            c = torch.stack([y.ravel(), x.ravel()], dim=1)
            final = c[c[:, 0] != c[:, 1]]
            triple_bond_indices[start : start + (n * (n - 1)), :] = final + cs
            start += n * (n - 1)
            cs += n

    n_triple_s = []
    i = 0
    for n in n_atoms:
        j = i + n
        n_triple_s.append(torch.sum(n_triple_i[i:j]))
        i = j

    src_id, dst_id = (triple_bond_indices[:, 0], triple_bond_indices[:, 1])
    l_g = dgl.graph((src_id, dst_id))
    three_body_id = torch.unique(triple_bond_indices)
    max_three_body_id = max(torch.cat([three_body_id + 1, torch.tensor([0])]))
    l_g.ndata["bond_dist"] = g.edata["bond_dist"][:max_three_body_id]
    l_g.ndata["bond_vec"] = g.edata["bond_vec"][:max_three_body_id]
    l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][:max_three_body_id]
    l_g.ndata["n_triple_ij"] = n_triple_ij[:max_three_body_id]
    n_triple_s = torch.tensor(n_triple_s, dtype=torch.int64)  # type: ignore
    return l_g, triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs.

    Args:
    g: DGL graph

    Returns:
    bond_vec (torch.tensor): bond distance between two atoms
    bond_dist (torch.tensor): vector from src node to dst node
    """
    bond_vec = torch.zeros(g.num_edges(), 3)
    bond_vec[:, :] = (
        g.ndata["pos"][g.edges()[1][:].long(), :]
        + torch.squeeze(torch.matmul(g.edata["pbc_offset"].unsqueeze(1), torch.squeeze(g.edata["lattice"])))
        - g.ndata["pos"][g.edges()[0][:].long(), :]
    )

    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def compute_theta_and_phi(edges: dgl.udf.EdgeBatch):
    """Calculate bond angle Theta and Phi using dgl graphs.

    Args:
    edges: DGL graph edges

    Returns:
    cos_theta: torch.Tensor
    phi: torch.Tensor
    triple_bond_lengths (torch.tensor):
    """
    angles = compute_theta(edges, cosine=True)
    angles["phi"] = torch.zeros_like(angles["cos_theta"])
    return angles


def compute_theta(edges: dgl.udf.EdgeBatch, cosine: bool = False) -> dict[str, torch.Tensor]:
    """User defined dgl function to calculate bond angles from edges in a graph.

    Args:
        edges: DGL graph edges
        cosine: Whether to return the cosine of the angle or the angle itself

    Returns:
        dict[str, torch.Tensor]: Dictionary containing bond angles and distances
    """
    vec1 = edges.src["bond_vec"]
    vec2 = edges.dst["bond_vec"]
    key = "cos_theta" if cosine else "theta"
    val = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
    if not cosine:
        val = torch.acos(val)
    return {key: val, "triple_bond_lengths": edges.dst["bond_dist"]}


def create_line_graph(g: dgl.DGLGraph, threebody_cutoff: float):
    """
    Calculate the three body indices from pair atom indices.

    Args:
        g: DGL graph
        threebody_cutoff (float): cutoff for three-body interactions

    Returns:
        l_g: DGL graph containing three body information from graph
    """
    graph_with_three_body = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff)
    l_g, triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s = compute_3body(graph_with_three_body)
    return l_g


def has_aliased_edges(graph: dgl.DGLGraph) -> bool:
    """Check if graph has aliased edges.

    edges are aliased if they share the same node tuple but have different images.

    Args:
        graph: DGL graph

    Returns:
        bool: whether graph has aliased edges
    """
    graph_adjacency = torch.stack(graph.edges(), dim=1)
    num_unique_edges = len(torch.unique(graph_adjacency, dim=0))
    return num_unique_edges < graph.number_of_edges()


def prune_edges_by_features(
    graph: dgl.DGLGraph,
    feat_name: str,
    condition: Callable[[torch.Tensor, Any, ...], torch.Tensor],
    keep_ndata: bool = False,
    keep_edata: bool = True,
    *args,
    **kwargs,
) -> dgl.DGLGraph:
    """Removes edges graph that do satisfy given condition based on a specified feature value.

    Returns a new graph with edges removed.

    Args:
        graph: DGL graph
        feat_name: edge field name
        condition: condition function. Must be a function where the first is the value
            of the edge field data and returns a Tensor of boolean values.
        keep_ndata: whether to keep node features
        keep_edata: whether to keep edge features
        *args: additional arguments to pass to condition function
        **kwargs: additional keyword arguments to pass to condition function

    Returns: dgl.Graph with removed edges.
    """
    if feat_name not in graph.edata.keys():
        raise ValueError(f"Edge field {feat_name} not an edge feature in given graph.")

    valid_edges = torch.logical_not(condition(graph.edata[feat_name], *args, **kwargs))
    src, dst = graph.edges()
    src, dst = src[valid_edges], dst[valid_edges]
    e_ids = valid_edges.nonzero().squeeze()
    new_g = dgl.graph((src, dst))
    new_g.edata["edge_ids"] = e_ids  # keep track of original edge ids

    if keep_ndata:
        for key, value in graph.ndata.items():
            new_g.ndata[key] = value
    if keep_edata:
        for key, value in graph.edata.items():
            new_g.edata[key] = value[valid_edges]

    return new_g
