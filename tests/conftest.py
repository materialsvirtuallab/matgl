"""
Define commonly used text fixtures.

"""
import pytest

from pymatgen.util.testing import PymatgenTest
from pymatgen.core import Structure, Lattice, Molecule
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list


def get_graph(s, cutoff):
    element_types = get_element_list([s])
    if isinstance(s, Structure):
        p2g = Structure2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
        return [s] + list(p2g.get_graph(s))
    else:
        m2g = Molecule2Graph(element_types=element_types, cutoff=cutoff)  # type: ignore
        return [s] + list(m2g.get_graph(s))


@pytest.fixture
def LiFePO4():
    return PymatgenTest.get_structure("LiFePO4")


@pytest.fixture
def CH4():
    coords = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 1.089000],
        [1.026719, 0.000000, -0.363000],
        [-0.513360, -0.889165, -0.363000],
        [-0.513360, 0.889165, -0.363000],
    ]
    return Molecule(["C", "H", "H", "H", "H"], coords)


@pytest.fixture
def BaNiO3():
    return PymatgenTest.get_structure("BaNiO3")


@pytest.fixture
def graph_Mo():
    """
    Returns:
        Structure, Graph, State
    """
    s = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])
    return get_graph(s, 5.0)


@pytest.fixture
def graph_CH4(CH4):
    """
    Returns:
        Molecule, Graph, State
    """

    return get_graph(CH4, 2.0)


@pytest.fixture
def graph_LiFePO4(LiFePO4):
    """
    Returns:
        Molecule, Graph, State
    """
    return get_graph(LiFePO4, 4.0)
