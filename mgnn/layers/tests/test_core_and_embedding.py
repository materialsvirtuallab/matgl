import unittest

import torch
import torch.nn as nn
from pymatgen.core.structure import Lattice, Structure

from mgnn.graph.compute import compute_pair_vector_and_distance
from mgnn.graph.converters import Pmg2Graph, get_element_list
from mgnn.layers.bond_expansion import BondExpansion
from mgnn.layers.core import MLP, GatedMLP
from mgnn.layers.embedding_block import EmbeddingBlock


class TestCoreAndEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])
        Structure(Lattice.cubic(3), ["Mo", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        element_types = get_element_list([cls.s1])
        p2g = Pmg2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph_from_structure(cls.s1)
        cls.g1 = graph
        cls.state1 = state
        bond_vec, bond_dist = compute_pair_vector_and_distance(cls.g1)
        cls.g1.edata["bond_dist"] = bond_dist

        cls.x = torch.randn(4, 10, requires_grad=True)

    def test_mlp(self):
        layer = MLP(dims=[10, 3], activation=nn.SiLU())
        out = layer(self.x)
        self.assertListEqual([out.size()[0], out.size()[1]], [4, 3])

    def test_gated_mlp(self):
        torch.manual_seed(42)
        layer = GatedMLP(in_feats=10, dims=[10, 1], activate_last=False)
        out = layer(self.x)
        self.assertListEqual([out.size()[0], out.size()[1]], [4, 1])

    def test_embedding(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_dist = self.g1.edata["bond_dist"]
        bond_basis = bond_expansion(bond_dist)
        # include state features
        embed = EmbeddingBlock(
            num_node_feats=16, num_edge_feats=16, num_state_feats=16, include_states=True, activation="swish"
        )
        graph_attr = torch.tensor([1.0, 2.0])
        node_attr = self.g1.ndata["attr"]
        node_attr.requires_grad = True
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, graph_attr)

        self.assertListEqual([node_feat.size(dim=0), node_feat.size(dim=1)], [2, 16])
        self.assertListEqual([edge_feat.size(dim=0), edge_feat.size(dim=1)], [28, 16])
        self.assertListEqual([state_feat.size(dim=0), state_feat.size(dim=1)], [1, 16])
        # include state embedding
        embed2 = EmbeddingBlock(
            num_node_feats=16,
            num_edge_feats=16,
            include_states=True,
            state_embedding_dim=32,
            num_state_types=2,
            activation="swish",
        )
        node_feat, edge_feat, state_feat = embed2(node_attr, edge_attr, torch.tensor([1]))
        self.assertListEqual([state_feat.size(dim=0), state_feat.size(dim=1)], [1, 32])
        # include state features
        embed3 = EmbeddingBlock(
            num_node_feats=16, num_edge_feats=16, num_state_feats=16, include_states=True, activation="swish"
        )
        node_feat, edge_feat, state_feat = embed3(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        self.assertListEqual([state_feat.size(dim=0), state_feat.size(dim=1)], [1, 16])
        # without any state feature
        embed4 = EmbeddingBlock(num_node_feats=16, num_edge_feats=16, activation="swish")
        node_feat, edge_feat, state_feat = embed4(
            node_attr, edge_attr, torch.tensor([0.0, 0.0])
        )  # this will be default value
        self.assertTrue(state_feat is None)


if __name__ == "__main__":
    unittest.main()
