from __future__ import annotations

import os
import unittest

import torch as th
from pymatgen.core.structure import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import BondExpansion
from matgl.models import MEGNet


class MEGNetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        cls.element_types = get_element_list([s])  # type: ignore
        p2g = Structure2Graph(element_types=cls.element_types, cutoff=5.0)  # type: ignore
        graph, state = p2g.get_graph(s)
        cls.g1 = graph  # type: ignore
        cls.state1 = state  # type: ignore

    def test_megnet(self):
        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=2,
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
        assert [th.numel(output)] == [1]

    def test_save_load(self):
        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=2,
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
        model.save(".", metadata={"description": "forme model"})
        MEGNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")


if __name__ == "__main__":
    unittest.main()
