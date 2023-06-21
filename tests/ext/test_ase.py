from __future__ import annotations

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.apps.pes import Potential
from matgl.ext.ase import Atoms2Graph, M3GNetCalculator
from matgl.models import M3GNet


def test_M3GNetCalculator(MoS):
    adaptor = AseAtomsAdaptor()
    s_ase = adaptor.get_atoms(MoS)  # type: ignore
    model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
    ff = Potential(model=model)
    calc = M3GNetCalculator(potential=ff)
    s_ase.set_calculator(calc)
    assert [s_ase.get_potential_energy().size] == [1]
    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]


def test_get_graph_from_atoms(LiFePO4):
    adaptor = AseAtomsAdaptor()
    structure_ase = adaptor.get_atoms(LiFePO4)
    a2g = Atoms2Graph(element_types=["Li", "Fe", "P", "O"], cutoff=4.0)
    graph, state = a2g.get_graph(structure_ase)
    # check the number of nodes
    assert np.allclose(graph.num_nodes(), len(structure_ase.get_atomic_numbers()))
    # check the atomic feature of atom 0
    assert np.allclose(graph.ndata["attr"][0].numpy(), [1, 0, 0, 0])
    # check the atomic feature of atom 4
    assert np.allclose(graph.ndata["attr"][4].numpy(), [0, 1, 0, 0])
    # check the number of bonds
    assert np.allclose(graph.num_edges(), 704)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])
