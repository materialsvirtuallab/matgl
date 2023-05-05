from __future__ import annotations

import unittest

import torch as th
from pymatgen.core.structure import Lattice, Structure

from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.graph.converters import Pmg2Graph, get_element_list
from matgl.layers._bond import BondExpansion
from matgl.models import MEGNet


class TestMEGNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        cls.element_types = get_element_list([s])
        p2g = Pmg2Graph(element_types=cls.element_types, cutoff=5.0)
        graph, state = p2g.get_graph_from_structure(s)
        cls.g1 = graph
        cls.state1 = state

    def test_megnet(self):
        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_attr_embedding=2,
            nblocks=3,
            include_states=True,
            hidden_layer_sizes_input=(64, 32),
            hidden_layer_sizes_conv=(64, 64, 32),
            activation_type="swish",
            nlayers_set2set=4,
            niters_set2set=3,
            hidden_layer_sizes_output=(32, 16),
            is_classification=True,
        )
        bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=6.0, num_centers=100, width=0.5)
        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g1)
        self.g1.edata["edge_attr"] = bond_expansion(bond_dist)
        self.state1 = th.tensor(self.state1)
        output = model(self.g1, self.g1.edata["edge_attr"], self.g1.ndata["node_type"], self.state1)
        self.assertListEqual([th.numel(output)], [1])


if __name__ == "__main__":
    unittest.main()
