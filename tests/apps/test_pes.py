from __future__ import annotations

import torch
from matgl.apps.pes import Potential
from matgl.models._m3gnet import M3GNet


class TestPotential:
    def test_potential_efsh(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_efs(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
        ff = Potential(model=model)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0)] == [1]

    def test_potential_ef(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
        ff = Potential(model=model, calc_stresses=False)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_potential_e(self, graph_MoS):
        structure, graph, state = graph_MoS
        model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
        ff = Potential(model=model, calc_forces=False, calc_stresses=False)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0)] == [1]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]
