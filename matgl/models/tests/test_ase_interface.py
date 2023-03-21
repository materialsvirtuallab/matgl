import unittest

import numpy as np
from pymatgen.core.structure import Lattice, Molecule, Structure

from matgl.graph.converters import Pmg2Graph, get_element_list
from matgl.models.m3gnet import M3GNet
from matgl.models.potential import Potential
from matgl.models.ase_interface import M3GNetCalculator, Relaxer, MolecularDynamics
from pymatgen.io.ase import AseAtomsAdaptor


class TestAseInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        cls.element_types = get_element_list([s])
        adaptor = AseAtomsAdaptor()
        cls.s_ase = adaptor.get_atoms(s)

    def test_M3GNetCalculator(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        ff = Potential(model=model)
        calc = M3GNetCalculator(potential=ff)
        self.s_ase.set_calculator(calc)
        self.assertListEqual([self.s_ase.get_potential_energy().size], [1])
        self.assertListEqual(list(self.s_ase.get_forces().shape), [2, 3])
        self.assertListEqual(list(self.s_ase.get_stress().shape), [6])


if __name__ == "__main__":
    unittest.main()
