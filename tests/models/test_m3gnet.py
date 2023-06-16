from __future__ import annotations

import os

import numpy as np
import torch
from pymatgen.core.structure import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.models import M3GNet


class TestM3GNet:
    # s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    # s.states = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    #
    # element_types = get_element_list([s])
    # p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
    # graph, state = p2g.get_graph(s)
    # g1 = graph
    # state1 = state

    def test_model(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
        output = model(g=graph)
        assert torch.numel(output) == 1
        model.save(".")
        M3GNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")

    def test_model_intensive(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=True)
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_classification(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(
            element_types=["Mo", "S"],
            is_intensive=True,
            task_type="classification",
        )
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_set2set_classification(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=True, task_type="classification", readout_type="set2set")
        output = model(g=graph)
        assert torch.numel(output) == 1
