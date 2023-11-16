from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl
from matgl.models import M3GNet


class TestM3GNet:
    def test_model(self, graph_MoS):
        structure, graph, state = graph_MoS
        for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
            model = M3GNet(is_intensive=False, activation_type=act)
            output = model(g=graph)
            assert torch.numel(output) == 1
        model.save(".")
        M3GNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")

    def test_exceptions(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            _ = M3GNet(element_types=None, is_intensive=False, activation_type="whatever")
        with pytest.raises(ValueError, match="Classification task cannot be extensive."):
            _ = M3GNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")

    def test_model_intensive(self, graph_MoS):
        structure, graph, state = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = M3GNet(element_types=["Mo", "S"], is_intensive=True)
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_classification(self, graph_MoS):
        structure, graph, state = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = M3GNet(
            element_types=["Mo", "S"],
            is_intensive=True,
            task_type="classification",
        )
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_set2set_classification(self, graph_MoS):
        structure, graph, state = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = M3GNet(element_types=["Mo", "S"], is_intensive=True, task_type="classification", readout_type="set2set")
        output = model(g=graph)
        assert torch.numel(output) == 1
