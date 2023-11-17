from __future__ import annotations

import torch
from matgl.graph.compute import (
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.layers import BondExpansion, EmbeddingBlock, SphericalBesselWithHarmonics
from matgl.layers._core import MLP, GatedMLP
from matgl.layers._three_body import ThreeBodyInteractions
from matgl.utils.cutoff import polynomial_cutoff
from torch import nn


def test_three_body_interactions(graph_MoS):
    s1, g1, state = graph_MoS
    l_g1 = create_line_graph(g1, threebody_cutoff=4.0)
    l_g1.apply_edges(compute_theta_and_phi)
    bond_expansion = BondExpansion(max_l=3, max_n=3, cutoff=5.0, rbf_type="SphericalBessel", smooth=False)
    bond_basis = bond_expansion(g1.edata["bond_dist"])
    g1.edata["rbf"] = bond_basis
    sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
    three_body_basis = sb_and_sh(l_g1)
    three_body_cutoff = polynomial_cutoff(g1.edata["bond_dist"], 4.0)
    max_n = 3
    max_l = 3
    num_node_feats = 16
    num_edge_feats = 16
    state_attr = torch.tensor([0.0, 0.0])
    embedding = EmbeddingBlock(
        degree_rbf=9,
        dim_node_embedding=num_node_feats,
        dim_edge_embedding=num_edge_feats,
        activation=nn.SiLU(),
        ntypes_node=2,
    )

    node_attr = g1.ndata["node_type"]
    edge_attr = g1.edata["rbf"]
    node_feat, edge_feat, state_feat = embedding(node_attr, edge_attr, state_attr)
    degree = max_n * max_l
    three_body_interactions = ThreeBodyInteractions(
        update_network_atom=MLP(dims=[num_node_feats, degree], activation=nn.Sigmoid(), activate_last=True),
        update_network_bond=GatedMLP(in_feats=degree, dims=[num_edge_feats], use_bias=False),
    )
    edge_feat_updated = three_body_interactions(g1, l_g1, three_body_basis, three_body_cutoff, node_feat, edge_feat)
    assert [edge_feat_updated.size(dim=0), edge_feat_updated.size(dim=1)] == [28, 16]
