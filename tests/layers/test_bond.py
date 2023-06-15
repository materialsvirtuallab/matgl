from __future__ import annotations

from pymatgen.core import Lattice, Molecule, Structure

from matgl.ext.pymatgen import Structure2Graph, Molecule2Graph, get_element_list
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import BondExpansion


class TestBondExpansion:
    @classmethod
    def setup_class(cls):
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.01, 0.0, 0.0], [0.5, 0.5, 0.5]])
        mol = Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])

        element_types = get_element_list([s])
        p2g = Structure2Graph(element_types=element_types, cutoff=4.0)  # type: ignore
        graph, state = p2g.get_graph(s)
        cls.g1 = graph
        cls.state1 = state

        m2g = Molecule2Graph(["C", "O"])
        graph, state = m2g.get_graph(mol)
        cls.g2 = graph
        cls.state2 = state

    def test_gaussian(self):
        bond_expansion = BondExpansion(rbf_type="Gaussian", num_centers=10, initial=0.0, final=4.0, width=0.5)
        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g1)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (28, 10)

        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g2)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (2, 10)

    def test_spherical_bessel_with_smooth(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=True)
        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g1)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (28, 3)

        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g2)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (2, 3)

    def test_spherical_bessel(self):
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g1)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (28, 9)

        bond_vec, bond_dist = compute_pair_vector_and_distance(self.g2)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (2, 9)
