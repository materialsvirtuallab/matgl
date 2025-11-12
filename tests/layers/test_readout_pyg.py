from __future__ import annotations

import pytest
import torch
from torch import nn

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._readout_pyg import (
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)


class TestReadOut:
    def test_weighted_readout(self, graph_MoS_pyg):
        """Test WeightedReadOut."""
        _, g1, _ = graph_MoS_pyg
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(g1.bond_dist)
        embed = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            activation=nn.SiLU(),
            ntypes_node=2,
        )
        node_attr = g1.node_type
        edge_attr = bond_basis
        node_feat, edge_feat, _ = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        g1.node_feat = node_feat
        g1.edge_feat = edge_feat
        read_out = WeightedReadOut(in_feats=16, dims=[32, 32], num_targets=4)
        atomic_properties = read_out(g1)
        assert [atomic_properties.size(dim=0), atomic_properties.size(dim=1)] == [2, 4]

    def test_weighted_atom_readout(self, graph_MoS_pyg):
        """Test WeightedAtomReadOut."""
        _, g1, _ = graph_MoS_pyg
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(g1.bond_dist)
        embed = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            activation=nn.SiLU(),
            ntypes_node=2,
        )
        node_attr = g1.node_type
        edge_attr = bond_basis
        node_feat, edge_feat, _ = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        g1.node_feat = node_feat
        g1.edge_feat = edge_feat
        read_out = WeightedAtomReadOut(in_feats=16, dims=[32, 32], activation=nn.SiLU())
        graph_properties = read_out(g1)
        assert [graph_properties.size(dim=0), graph_properties.size(dim=1)] == [1, 32]

    def test_reduce_readout(self, graph_MoS_pyg):
        """Test ReduceReadOut."""
        _, g1, _ = graph_MoS_pyg
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(g1.bond_dist)
        embed = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            activation=nn.SiLU(),
            ntypes_node=2,
        )
        node_attr = g1.node_type
        edge_attr = bond_basis
        node_feat, edge_feat, _ = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        g1.node_feat = node_feat
        g1.edge_feat = edge_feat
        read_out = ReduceReadOut(op="mean", field="node_feat")
        output = read_out(g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 16]

    def test_set2set_readout(self, graph_MoS_pyg):
        """Test Set2SetReadOut."""
        _, g1, _ = graph_MoS_pyg
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(g1.bond_dist)
        embed = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=32,
            dim_state_feats=16,
            activation=nn.SiLU(),
            ntypes_node=2,
        )
        node_attr = g1.node_type
        edge_attr = bond_basis
        node_feat, edge_feat, _ = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        g1.node_feat = node_feat
        g1.edge_feat = edge_feat
        read_out = Set2SetReadOut(
            in_feats=16,
            n_iters=3,
            n_layers=3,
            field="node_feat",
        )
        output = read_out(g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 32]
        read_out2 = Set2SetReadOut(
            in_feats=32,
            n_iters=3,
            n_layers=3,
            field="edge_feat",
        )
        output2 = read_out2(g1)
        assert [output2.size(dim=0), output2.size(dim=1)] == [1, 64]

        with pytest.raises(ValueError, match="Field must be node_feat or edge_feat"):
            Set2SetReadOut(1, 2, 3, field="nonsense")
