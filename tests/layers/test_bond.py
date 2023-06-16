from __future__ import annotations

from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers import BondExpansion


class TestBondExpansion:
    def test_gaussian(self, graph_MoS, graph_CO):
        _, g1, _ = graph_MoS
        _, g2, _ = graph_CO
        bond_expansion = BondExpansion(rbf_type="Gaussian", num_centers=10, initial=0.0, final=4.0, width=0.5)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g1)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (28, 10)

        bond_vec, bond_dist = compute_pair_vector_and_distance(g2)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (2, 10)

    def test_spherical_bessel_with_smooth(self, graph_MoS, graph_CO):
        _, g1, _ = graph_MoS
        _, g2, _ = graph_CO
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=True)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g1)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (28, 3)

        bond_vec, bond_dist = compute_pair_vector_and_distance(g2)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (2, 3)

    def test_spherical_bessel(self, graph_MoS, graph_CO):
        _, g1, _ = graph_MoS
        _, g2, _ = graph_CO
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=False)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g1)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (28, 9)

        bond_vec, bond_dist = compute_pair_vector_and_distance(g2)
        bond_basis = bond_expansion(bond_dist)
        assert bond_basis.shape == (2, 9)
