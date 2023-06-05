from __future__ import annotations

import unittest

import torch
from pymatgen.core.structure import Lattice, Structure
from torch import nn

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._core import MLP, GatedMLP


class TestCoreAndEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(4.0), ["Mo", "Mo"], [[0.0, 0, 0], [0.5, 0.5, 0.5]])
        Structure(Lattice.cubic(3), ["Mo", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        element_types = get_element_list([cls.s1])
        p2g = Structure2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph(cls.s1)
        cls.g1 = graph
        cls.state1 = state

        bond_vec, bond_dist = compute_pair_vector_and_distance(cls.g1)
        cls.g1.edata["bond_dist"] = bond_dist

        cls.x = torch.randn(4, 10, requires_grad=True)

    def test_mlp(self):
        layer = MLP(dims=[10, 3], activation=nn.SiLU())
        out = layer(self.x).double()
        assert [out.size()[0], out.size()[1]] == [4, 3]

    def test_gated_mlp(self):
        torch.manual_seed(42)
        layer = GatedMLP(in_feats=10, dims=[10, 1], activate_last=False)
        out = layer(self.x)
        assert [out.size()[0], out.size()[1]] == [4, 1]

    def test_embedding(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_dist = self.g1.edata["bond_dist"]
        bond_basis = bond_expansion(bond_dist)
        # include state features
        embed = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_feats=16,
            include_state=True,
            activation=nn.SiLU(),
        )
        state_attr = torch.tensor([1.0, 2.0])
        node_attr = self.g1.ndata["attr"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, state_attr)

        assert [node_feat.size(dim=0), node_feat.size(dim=1)] == [2, 16]
        assert [edge_feat.size(dim=0), edge_feat.size(dim=1)] == [28, 16]
        assert [state_feat.size(dim=0), state_feat.size(dim=1)] == [1, 16]
        # include state embedding
        embed2 = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=16,
            dim_edge_embedding=16,
            include_state=True,
            dim_state_embedding=32,
            ntypes_state=2,
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
            activation=nn.SiLU(),
        )
        node_feat, edge_feat, state_feat = embed3(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        assert [state_feat.size(dim=0), state_feat.size(dim=1)] == [1, 16]
        # without any state feature
        embed4 = EmbeddingBlock(degree_rbf=9, dim_node_embedding=16, dim_edge_embedding=16, activation=nn.SiLU())
        node_feat, edge_feat, state_feat = embed4(
            node_attr, edge_attr, torch.tensor([0.0, 0.0])
        )  # this will be default value
        assert state_feat is None


if __name__ == "__main__":
    unittest.main()
