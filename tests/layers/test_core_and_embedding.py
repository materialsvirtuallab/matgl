from __future__ import annotations

import matgl
import pytest
import torch
from matgl.layers import (
    BondExpansion,
    EmbeddingBlock,
    GatedEquivariantBlock,
    NeighborEmbedding,
    TensorEmbedding,
    build_gated_equivariant_mlp,
)
from matgl.layers._core import MLP, GatedMLP
from torch import nn


@pytest.fixture()
def x():
    return torch.randn(4, 10, requires_grad=True)


class TestCoreAndEmbedding:
    def test_mlp(self, x):
        layer = MLP(dims=[10, 3], activation=nn.SiLU())
        out = layer(x).double()
        assert [out.size()[0], out.size()[1]] == [4, 3]
        assert layer.last_linear.out_features == 3
        assert layer.depth == 1
        assert layer.out_features == 3
        assert layer.in_features == 10

    def test_gated_mlp(self, x):
        torch.manual_seed(42)
        layer = GatedMLP(in_feats=10, dims=[10, 1], activate_last=False)
        out = layer(x)
        assert [out.size()[0], out.size()[1]] == [4, 1]

    def test_gated_equivairant_block(self, x):
        scaler_input = x
        vector_input = torch.randn(4, 3, 10)

        output_scalar, output_vector = GatedEquivariantBlock(
            n_sin=10,
            n_vin=10,
            n_sout=10,
            n_vout=10,
            n_hidden=10,
            activation=nn.SiLU(),
            sactivation=nn.SiLU(),
        )([scaler_input, vector_input])

        assert output_scalar.shape == (4, 10)
        assert output_vector.shape == (4, 3, 10)

    def test_build_gated_equivariant_mlp(selfself, x):
        scaler_input = x
        vector_input = torch.randn(4, 3, 10)
        net = build_gated_equivariant_mlp(  # type: ignore
            n_in=10,
            n_out=1,
            n_hidden=10,
            n_layers=2,
            activation=nn.SiLU(),
            sactivation=nn.SiLU(),
        )
        output_scalar, output_vector = net([scaler_input, vector_input])

        assert output_scalar.shape == (4, 1)
        assert torch.squeeze(output_vector).shape == (4, 3)
        # without n_hidden
        net = build_gated_equivariant_mlp(n_in=10, n_out=1, n_layers=2, activation=nn.SiLU(), sactivation=nn.SiLU())
        output_scalar, output_vector = net([scaler_input, vector_input])

        assert output_scalar.shape == (4, 1)
        assert torch.squeeze(output_vector).shape == (4, 3)
        # with n_gating_hidden
        net = build_gated_equivariant_mlp(
            n_in=10,
            n_out=1,
            n_hidden=10,
            n_layers=2,
            n_gating_hidden=2,
            activation=nn.SiLU(),
            sactivation=nn.SiLU(),
        )
        output_scalar, output_vector = net([scaler_input, vector_input])

        assert output_scalar.shape == (4, 1)
        assert torch.squeeze(output_vector).shape == (4, 3)

        # with sequence n_gating_hidden
        net = build_gated_equivariant_mlp(
            n_in=10,
            n_out=1,
            n_hidden=10,
            n_layers=2,
            n_gating_hidden=[10, 10],
            activation=nn.SiLU(),
            sactivation=nn.SiLU(),
        )
        output_scalar, output_vector = net([scaler_input, vector_input])

        assert output_scalar.shape == (4, 1)
        assert torch.squeeze(output_vector).shape == (4, 3)

    def test_embedding(self, graph_Mo):
        s1, g1, state1 = graph_Mo
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(g1.edata["bond_dist"])
        # include state features
        embed = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            ntypes_node=2,
            include_state=True,
            activation=nn.SiLU(),
        )
        state_attr = torch.tensor([1.0, 2.0])
        node_attr = g1.ndata["node_type"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, state_attr)

        assert [node_feat.size(dim=0), node_feat.size(dim=1)] == [2, 16]
        assert [edge_feat.size(dim=0), edge_feat.size(dim=1)] == [52, 16]
        assert [state_feat.size(dim=0), state_feat.size(dim=1)] == [1, 16]
        # include state embedding
        embed2 = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            include_state=True,
            dim_state_embedding=32,
            ntypes_state=2,
            ntypes_node=2,
            activation=nn.SiLU(),
        )
        node_feat, edge_feat, state_feat = embed2(node_attr, edge_attr, torch.tensor([1]))
        assert [state_feat.size(dim=0), state_feat.size(dim=1)] == [1, 32]
        # include state features
        embed3 = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            include_state=True,
            ntypes_node=2,
            activation=nn.SiLU(),
        )
        node_feat, edge_feat, state_feat = embed3(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        assert [state_feat.size(dim=0), state_feat.size(dim=1)] == [1, 16]
        # without any state feature
        embed4 = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            ntypes_node=2,
            activation=nn.SiLU(),
        )
        node_feat, edge_feat, state_feat = embed4(
            node_attr, edge_attr, torch.tensor([0.0, 0.0])
        )  # this will be default value
        assert state_feat is None

        # No ntypes_node.
        embed5 = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            include_state=True,
            ntypes_node=None,
            activation=nn.SiLU(),
        )
        node_feat, edge_feat, state_feat = embed5(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        assert [state_feat.size(dim=0), state_feat.size(dim=1)] == [1, 16]

    def test_tensor_embedding(self, graph_Mo):
        s1, g1, state1 = graph_Mo
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=True)
        g1.edata["edge_attr"] = bond_expansion(g1.edata["bond_dist"])
        # without state
        tensor_embedding = TensorEmbedding(
            units=64,
            degree_rbf=3,
            activation=nn.SiLU(),
            ntypes_node=1,
            cutoff=5.0,
            dtype=matgl.float_th,
        )

        X, edge_feat, state_feat = tensor_embedding(g1, state1)

        assert [X.shape[0], X.shape[1], X.shape[2], X.shape[3]] == [2, 64, 3, 3]
        assert [edge_feat.shape[0], edge_feat.shape[1]] == [52, 64]
        # with state embedding
        tensor_embedding = TensorEmbedding(
            units=64,
            degree_rbf=3,
            activation=nn.SiLU(),
            ntypes_node=1,
            cutoff=5.0,
            dtype=matgl.float_th,
            ntypes_state=2,
            include_state=True,
            dim_state_embedding=16,
        )

        X, edge_feat, state_feat = tensor_embedding(g1, torch.tensor([1]))

        assert [X.shape[0], X.shape[1], X.shape[2], X.shape[3]] == [2, 64, 3, 3]
        assert [edge_feat.shape[0], edge_feat.shape[1]] == [52, 64]
        assert [state_feat.shape[0], state_feat.shape[1]] == [1, 16]

        # with state MLP
        tensor_embedding = TensorEmbedding(
            units=64,
            degree_rbf=3,
            activation=nn.SiLU(),
            ntypes_node=1,
            cutoff=5.0,
            dtype=matgl.float_th,
            include_state=True,
            dim_state_feats=16,
        )
        X, edge_feat, state_feat = tensor_embedding(g1, torch.tensor(state1))

        assert [X.shape[0], X.shape[1], X.shape[2], X.shape[3]] == [2, 64, 3, 3]
        assert [edge_feat.shape[0], edge_feat.shape[1]] == [52, 64]
        assert [state_feat.shape[0], state_feat.shape[1]] == [1, 16]

    def test_neighbor_embedding(self, graph_Mo):
        s1, g1, state1 = graph_Mo
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=True)
        g1.edata["edge_attr"] = bond_expansion(g1.edata["bond_dist"])

        embedding = NeighborEmbedding(ntypes_node=1, hidden_channels=64, num_rbf=3, cutoff=5.0)

        g1.ndata["node_feat"] = torch.rand(2, 64)

        x = embedding(
            g1.ndata["node_type"],
            g1.ndata["node_feat"],
            g1.edges(),
            g1.edata["bond_dist"],
            g1.edata["edge_attr"],
        )

        assert [x.shape[0], x.shape[1]] == [2, 64]
