from __future__ import annotations

import pytest
import torch

from matgl.apps.pes import Potential
from pymatgen.core import Structure, Lattice
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.models._m3gnet import M3GNet


@pytest.fixture()
def model():
    return M3GNet(element_types=["Mo", "S"], is_intensive=False)


class TestPotential:
    def test_potential_efsh(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_efs(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        ff = Potential(model=model)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0)] == [1]

    def test_potential_ef(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        ff = Potential(model=model, calc_stresses=False)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_potential_e(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        ff = Potential(model=model, calc_forces=False, calc_stresses=False)
        e, f, s, h = ff(graph, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0)] == [1]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_potential_two_body(self, model):
        structure = Structure(Lattice.cubic(10.0), ["Mo", "Mo"], [[0.0, 0, 0], [0.2, 0.0, 0.0]])
        element_types = get_element_list([structure])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, state = p2g.get_graph(structure)
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(graph, torch.tensor(state))
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_isolated_atom(self, model):
        structure = Structure(Lattice.cubic(10.0), ["Mo"], [[0.0, 0, 0]])
        element_types = get_element_list([structure])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, state = p2g.get_graph(structure)
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(graph, torch.tensor(state))
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [1, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [3, 3]
