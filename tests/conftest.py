"""
Define commonly used text fixtures.

"""
import pytest

from pymatgen.util.testing import PymatgenTest
from pymatgen.core import Structure, Lattice, Molecule
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
)


def get_graph(structure, cutoff):
    element_types = get_element_list([structure])
    if isinstance(structure, Structure):
        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
    else:
        converter = Molecule2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
    graph, state = converter.get_graph(structure)
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
def CO():
    return Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])


@pytest.fixture(scope="session")
def BaNiO3():
    return PymatgenTest.get_structure("BaNiO3")


@pytest.fixture(scope="session")
def MoS():
    return Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="session")
def graph_Mo():
    """
    Returns:
        Structure, Graph, State
    """
    s = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])
    return get_graph(s, 5.0)


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
