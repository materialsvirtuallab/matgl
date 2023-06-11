from __future__ import annotations

import unittest

import dgl
import numpy as np
import torch
from pymatgen.core.structure import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.layers._atom_ref import AtomRef


class TestAtomRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "S", "H"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])
        cls.element_types = get_element_list([cls.s1])
        p2g = Structure2Graph(element_types=cls.element_types, cutoff=4.0)
        graph, state = p2g.get_graph(cls.s1)
        cls.g1 = graph
        cls.state1 = state

    def test_atom_ref(self):
        element_ref = AtomRef(np.array([0.5, 1.0, 2.0]))

        atom_ref = element_ref(self.g1)
        assert atom_ref == 3.5

    def test_atom_ref_fit(self):
        element_ref = AtomRef(np.array([0.5, 1.0, 2.0]))
        properties = torch.tensor([2.0, 2.0])
        bg = dgl.batch([self.g1, self.g1])
        element_ref.fit([self.g1, self.g1], self.element_types, properties)
        atom_ref = element_ref(bg)
        assert list(np.round(atom_ref.numpy())) == [2.0, 2.0]

    def test_atom_ref_with_states(self):
        element_ref = AtomRef(np.array([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
        state_label = torch.tensor([1])
        atom_ref = element_ref(self.g1, state_label)
        assert atom_ref == 10


if __name__ == "__main__":
    unittest.main()
