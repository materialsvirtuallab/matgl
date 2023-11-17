"""
Define commonly used text fixtures. These are meant to be reused in unittests.
- Fixtures that are formulae (e.g., LiFePO4) returns the appropriate pymatgen Structure or Molecule based on the most
  commonly known structure.
- Fixtures that are prefixed with `graph_` returns a (structure, graph, state) tuple.

Given that the fixtures are unlikely to be modified by the underlying code, the fixtures are set with a scope of
"session". In the event that future tests are written that modifies the fixtures, these can be set to the default scope
of "function".
"""
from __future__ import annotations

import matgl
import pytest
import torch
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
)
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.util.testing import PymatgenTest

matgl.clear_cache(confirm=False)


def get_graph(structure, cutoff):
    """
    Helper class to generate DGL graph from an input Structure or Molecule.

    Returns:
        Structure/Molecule, Graph, State
    """
    element_types = get_element_list([structure])
    if isinstance(structure, Structure):
        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
    else:
        converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
    graph, lattice, state = converter.get_graph(structure)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
    graph.edata["bond_dist"] = bond_dist
    graph.edata["bond_vec"] = bond_vec
    return structure, graph, state


@pytest.fixture(scope="session")
def LiFePO4():
    return PymatgenTest.get_structure("LiFePO4")


@pytest.fixture(scope="session")
def CH4():
    coords = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 1.089000],
        [1.026719, 0.000000, -0.363000],
        [-0.513360, -0.889165, -0.363000],
        [-0.513360, 0.889165, -0.363000],
    ]
    return Molecule(["C", "H", "H", "H", "H"], coords)


@pytest.fixture(scope="session")
def AcAla3NHMe():
    coords = [
        [2.04866000, 3.03970000, -1.90538000],
        [2.57508000, 1.72268000, -0.85631500],
        [2.17184000, 3.30896000, -0.15238200],
        [1.90655000, 2.58401000, -0.92348000],
        [0.03771400, 1.25465000, -1.59488000],
        [0.48396700, 2.10062000, -0.82751300],
        [-0.28972800, 2.66225000, 0.13607100],
        [0.16103700, 3.23064000, 0.83232600],
        [-1.66536000, 2.24227000, 0.38856700],
        [-1.98080000, 2.78577000, 1.28200000],
        [-2.60093000, 2.60713000, -0.76094600],
        [-1.65737000, 0.76935300, 0.84076700],
        [-1.07671000, 0.46484900, 1.87157000],
        [-3.62795000, 2.31590000, -0.52731400],
        [-2.58345000, 3.68786000, -0.91203300],
        [-2.28934000, 2.13627000, -1.69468000],
        [-2.28796000, -0.13783200, 0.06597500],
        [-2.61433000, 0.15084400, -0.84271100],
        [-2.21633000, -1.56248000, 0.35525800],
        [-2.49106000, -1.71026000, 1.40128000],
        [-3.17539000, -2.32986000, -0.54719400],
        [-0.79661200, -2.15180000, 0.24417300],
        [-0.54538500, -3.20721000, 0.79546200],
        [-3.09773000, -3.39555000, -0.33182000],
        [-4.20586000, -2.01059000, -0.37590300],
        [-2.92852000, -2.17659000, -1.60274000],
        [0.08518600, -1.43854000, -0.48961600],
        [-0.20393600, -0.57022000, -0.92587000],
        [1.49370000, -1.76346000, -0.59747700],
        [1.62352000, -2.74612000, -0.13439600],
        [1.93595000, -1.80971000, -2.05523000],
        [2.37375000, -0.79291000, 0.21284200],
        [3.55545000, -0.63911100, -0.05660100],
        [3.00626000, -2.00417000, -2.11517000],
        [1.39238000, -2.59759000, -2.58040000],
        [1.73079000, -0.85338400, -2.54426000],
        [1.77244000, -0.14321700, 1.23186000],
        [0.79538700, -0.29827700, 1.43700000],
        [2.52442000, 0.70798900, 2.12658000],
        [3.17617000, 0.12474300, 2.78380000],
        [3.15630000, 1.39339000, 1.55717000],
        [1.81944000, 1.27508000, 2.73527000],
    ]
    return Molecule(
        [
            "H",
            "H",
            "H",
            "C",
            "O",
            "C",
            "N",
            "H",
            "C",
            "H",
            "C",
            "C",
            "O",
            "H",
            "H",
            "H",
            "N",
            "H",
            "C",
            "H",
            "C",
            "C",
            "O",
            "H",
            "H",
            "H",
            "N",
            "H",
            "C",
            "H",
            "C",
            "C",
            "O",
            "H",
            "H",
            "H",
            "N",
            "H",
            "C",
            "H",
            "H",
            "H",
        ],
        coords,
    )


@pytest.fixture(scope="session")
def graph_AcAla3NHMe(AcAla3NHMe):
    return get_graph(AcAla3NHMe, 5.0)


@pytest.fixture(scope="session")
def CO():
    return Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])


@pytest.fixture(scope="session")
def BaNiO3():
    return PymatgenTest.get_structure("BaNiO3")


@pytest.fixture(scope="session")
def MoS():
    return Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="session")
def Mo():
    return Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="session")
def graph_Mo(Mo):
    return get_graph(Mo, 5.0)


@pytest.fixture(scope="session")
def graph_CH4(CH4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(CH4, 2.0)


@pytest.fixture(scope="session")
def graph_LiFePO4(LiFePO4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(LiFePO4, 4.0)


@pytest.fixture(scope="session")
def graph_MoS(MoS):
    return get_graph(MoS, 5.0)


@pytest.fixture(scope="session")
def graph_CO(CO):
    return get_graph(CO, 5.0)


@pytest.fixture(scope="session")
def graph_MoSH():
    s = Structure(Lattice.cubic(3.17), ["Mo", "S", "H"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
    return get_graph(s, 4.0)
