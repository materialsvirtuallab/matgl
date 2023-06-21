from __future__ import annotations

import os

import torch

from matgl.models import M3GNet


class TestM3GNet:
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
