import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Molecule, Structure

from megnet.graph.converters import Pmg2Graph, get_element_list
from megnet.models.m3gnet import M3GNet


class TestM3GNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s.states = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])

        cls.element_types = get_element_list([s])
        p2g = Pmg2Graph(element_types=cls.element_types, cutoff=5.0)
        graph, state = p2g.get_graph_from_structure(s)
        cls.g1 = graph
        cls.state1 = state

    def test_model(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False, element_refs=np.array([-0.5, -1.0]))
        total_energy = model(self.g1, self.state1)
        self.assertListEqual([total_energy.size(dim=0)], [1])


if __name__ == "__main__":
    unittest.main()
