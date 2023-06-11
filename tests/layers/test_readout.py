from __future__ import annotations

import unittest

import torch
from pymatgen.core.structure import Lattice, Structure
from torch import nn

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._readout import (
    ReduceReadOut,
    Set2SetReadOut,
    WeightedReadOut,
    WeightedReadOutPair,
)


class TestReadOut(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "S"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        element_types = get_element_list([cls.s1])
        p2g = Structure2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph(cls.s1)
        cls.g1 = graph
        cls.state1 = state

        bond_vec, bond_dist = compute_pair_vector_and_distance(cls.g1)

        cls.g1.edata["bond_dist"] = bond_dist
        cls.g1.edata["bond_vec"] = bond_vec

    def test_weighted_readout(self):
        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g1)
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(self.g1.edata["bond_dist"])
        embed = EmbeddingBlock(
            degree_rbf=9, dim_node_embedding=16, dim_edge_embedding=16, dim_state_feats=16, activation=nn.SiLU()
        )
        node_attr = self.g1.ndata["attr"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        self.g1.ndata["node_feat"] = node_feat
        self.g1.edata["edge_feat"] = edge_feat
        read_out = WeightedReadOut(in_feats=16, dims=[32, 32], num_targets=4)
        atomic_properties = read_out(self.g1)
        assert [atomic_properties.size(dim=0), atomic_properties.size(dim=1)] == [2, 4]

    def test_readout_pair(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(self.g1.edata["bond_dist"])
        node_attr = self.g1.ndata["attr"]
        edge_attr = bond_basis
        embed = EmbeddingBlock(
            degree_rbf=9, dim_node_embedding=16, dim_edge_embedding=16, dim_state_feats=16, activation=nn.SiLU()
        )
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        read_out = WeightedReadOutPair(in_feats=16, dims=[32, 32], num_targets=1)
        self.g1.ndata["node_feat"] = node_feat
        self.g1.edata["edge_feat"] = edge_feat
        pair_properties = read_out(self.g1)
        assert [pair_properties.size(dim=0), pair_properties.size(dim=1), pair_properties.size(dim=2)] == [2, 2, 1]

    def test_reduce_readout(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(self.g1.edata["bond_dist"])
        embed = EmbeddingBlock(
            degree_rbf=9, dim_node_embedding=16, dim_edge_embedding=16, dim_state_feats=16, activation=nn.SiLU()
        )
        node_attr = self.g1.ndata["attr"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        self.g1.ndata["node_feat"] = node_feat
        self.g1.edata["edge_feat"] = edge_feat
        read_out = ReduceReadOut(op="mean", field="node_feat")
        output = read_out(self.g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 16]
        read_out2 = ReduceReadOut(op="mean", field="edge_feat")
        output2 = read_out2(self.g1)
        assert [output2.size(dim=0), output2.size(dim=1)] == [1, 16]

    def test_set2set_readout(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_basis = bond_expansion(self.g1.edata["bond_dist"])
        embed = EmbeddingBlock(
            degree_rbf=9, dim_node_embedding=16, dim_edge_embedding=32, dim_state_feats=16, activation=nn.SiLU()
        )
        node_attr = self.g1.ndata["attr"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embed(node_attr, edge_attr, torch.tensor([1.0, 2.0]))
        self.g1.ndata["node_feat"] = node_feat
        self.g1.edata["edge_feat"] = edge_feat
        read_out = Set2SetReadOut(
            num_steps=3,
            num_layers=3,
            field="node_feat",
        )
        output = read_out(self.g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 32]
        read_out2 = Set2SetReadOut(
            num_steps=3,
            num_layers=3,
            field="edge_feat",
        )
        output2 = read_out2(self.g1)
        assert [output2.size(dim=0), output2.size(dim=1)] == [1, 64]


if __name__ == "__main__":
    unittest.main()
