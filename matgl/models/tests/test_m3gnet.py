from __future__ import annotations

import os
import unittest

import numpy as np
import torch
from pymatgen.core.structure import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.models._m3gnet import M3GNet


class TestM3GNet(unittest.TestCase):
    element_types = None
    g1 = None
    state1 = None

    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s.states = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        cls.element_types = get_element_list([s])
        p2g = Structure2Graph(element_types=cls.element_types, cutoff=5.0)
        graph, state = p2g.get_graph(s)
        cls.g1 = graph
        cls.state1 = state

    def test_model(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        output = model(g=self.g1)
        assert torch.numel(output) == 1
        model.save(".")
        M3GNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")

    def test_model_intensive(self):
        model = M3GNet(element_types=self.element_types, is_intensive=True)
        output = model(g=self.g1)
        assert torch.numel(output) == 1

    def test_model_intensive_with_classification(self):
        model = M3GNet(
            element_types=self.element_types,
            is_intensive=True,
            task_type="classification",
        )
        output = model(g=self.g1)
        assert torch.numel(output) == 1

    def test_model_intensive_set2set_classification(self):
        model = M3GNet(
            element_types=self.element_types, is_intensive=True, task_type="classification", readout_type="set2set"
        )
        output = model(g=self.g1)
        assert torch.numel(output) == 1


if __name__ == "__main__":
    unittest.main()
