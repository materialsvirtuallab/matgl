"""Computing various graph based operations."""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import line_graph


def compute_pair_vector_and_distance(data: Data):
    src, dst = data.edge_index
    dst_pos = data.pos[dst] + data.pbc_offshift
    src_pos = data.pos[src]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)
    return bond_vec, bond_dist


def compute_theta_and_phi(theta_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # expects output of compute_theta as a dict
    theta = compute_theta(theta_dict, cosine=True, directed=False)
    theta["phi"] = torch.zeros_like(theta["cos_theta"])
    return theta


def compute_theta(
    inputs: dict[str, torch.Tensor], cosine: bool = False, directed: bool = True, eps: float = 1e-7
) -> dict[str, torch.Tensor]:
    vec1 = inputs["src_bond_vec"] * inputs["src_bond_sign"] if directed else inputs["src_bond_vec"]
    vec2 = inputs["dst_bond_vec"]
    key = "cos_theta" if cosine else "theta"
    val = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
    val = val.clamp_(min=-1 + eps, max=1 - eps)
    if not cosine:
        val = torch.acos(val)
    return {key: val, "triple_bond_lengths": inputs["dst_bond_dist"]}


def create_line_graph(
    data: Data,
    threebody_cutoff: float,
    directed: bool = False,
    error_handling: bool = False,
    numerical_noise: float = 1e-6,
) -> Data:
    # prune edges based on bond_dist
    mask = data.bond_dist <= threebody_cutoff
    edge_index = data.edge_index[:, mask]
    edge_masked = Data(
        x=data.x,
        edge_index=edge_index,
        pos=data.pos,
        pbc_offshift=data.pbc_offshift,
        bond_vec=data.bond_vec[mask],
        bond_dist=data.bond_dist[mask],
    )
    # construct line graph
    lg = line_graph(edge_index, force_directed=directed)
    # copy edge features into node features of line graph
    lg_data = Data(
        edge_index=lg.edge_index,
        pos=edge_masked.bond_vec,
        bond_dist=edge_masked.bond_dist,
        pbc_offshift=edge_masked.pbc_offshift,
    )
    return lg_data
