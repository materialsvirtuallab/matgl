import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Molecule, Structure

from megnet.graph.converters import Pmg2Graph, get_element_list
from megnet.models.m3gnet import M3GNet
from megnet.models.potential import Potential


class TestPotential(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0025, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s.states = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])

        cls.element_types = get_element_list([s])
        p2g = Pmg2Graph(element_types=cls.element_types, cutoff=5.0)
        graph, state = p2g.get_graph_from_structure(s)
        cls.g1 = graph
        cls.state1 = state

    def test_potential_efsh(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(self.g1, self.state1)
        self.assertListEqual([e.size(dim=0)], [1])
        self.assertListEqual([f.size(dim=0), f.size(dim=1)], [2, 3])
        self.assertListEqual([s.size(dim=0), s.size(dim=1)], [3, 3])
        self.assertListEqual([h.size(dim=0), h.size(dim=1)], [6, 6])

    def test_potential_efs(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        ff = Potential(model=model)
        e, f, s, h = ff(self.g1, self.state1)
        self.assertListEqual([e.size(dim=0)], [1])
        self.assertListEqual([f.size(dim=0), f.size(dim=1)], [2, 3])
        self.assertListEqual([s.size(dim=0), s.size(dim=1)], [3, 3])
        self.assertListEqual([h.size(dim=0)], [1])

    def test_potential_ef(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        ff = Potential(model=model, calc_stresses=False)
        e, f, s, h = ff(self.g1, self.state1)
        self.assertListEqual([e.size(dim=0)], [1])
        self.assertListEqual([f.size(dim=0), f.size(dim=1)], [2, 3])
        self.assertListEqual([s.size(dim=0)], [1])
        self.assertListEqual([h.size(dim=0)], [1])

    def test_potential_e(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        ff = Potential(model=model, calc_forces=False, calc_stresses=False)
        e, f, s, h = ff(self.g1, self.state1)
        self.assertListEqual([e.size(dim=0)], [1])
        self.assertListEqual([f.size(dim=0)], [1])
        self.assertListEqual([s.size(dim=0)], [1])
        self.assertListEqual([h.size(dim=0)], [1])


if __name__ == "__main__":
    unittest.main()
