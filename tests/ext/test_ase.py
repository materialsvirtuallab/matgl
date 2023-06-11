from __future__ import annotations

import unittest

from pymatgen.core.structure import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.apps.pes import Potential
from matgl.ext.ase import M3GNetCalculator
from matgl.ext.pymatgen import get_element_list
from matgl.models import M3GNet


class TestAseInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        cls.element_types = get_element_list([s])  # type: ignore
        adaptor = AseAtomsAdaptor()
        cls.s_ase = adaptor.get_atoms(s)  # type: ignore

    def test_M3GNetCalculator(self):
        model = M3GNet(element_types=self.element_types, is_intensive=False)
        ff = Potential(model=model)
        calc = M3GNetCalculator(potential=ff)
        self.s_ase.set_calculator(calc)
        assert [self.s_ase.get_potential_energy().size] == [1]
        assert list(self.s_ase.get_forces().shape) == [2, 3]
        assert list(self.s_ase.get_stress().shape) == [6]


if __name__ == "__main__":
    unittest.main()
