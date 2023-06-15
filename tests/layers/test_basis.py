from __future__ import annotations

import numpy as np
import torch

from pymatgen.core import Lattice, Structure

from matgl.layers._basis import (
    GaussianExpansion,
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    spherical_bessel_smooth,
    SphericalBesselWithHarmonics,
)
from matgl.layers._three_body import combine_sbf_shf
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)


class TestGaussianAndSphericalBesselFuction:
    def test_gaussian(self):
        r = torch.linspace(1.0, 5.0, 11)
        rbf_gaussian = GaussianExpansion(initial=0.0, final=5.0, num_centers=10, width=0.5)
        rbf = rbf_gaussian(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 10]

        rbf_gaussian = GaussianExpansion()
        r = torch.tensor([1.0])
        rbf = rbf_gaussian(r)
        # check the shape of a vector
        assert np.allclose([rbf.size(dim=0), rbf.size(dim=1)], [1, 20])
        # check the first value of expanded distance
        assert np.allclose(rbf[0][0], np.exp(-0.5 * np.power(1.0 - 0.0, 2.0)))
        # check the last value of expanded distance
        assert np.allclose(rbf[0][-1], np.exp(-0.5 * np.power(1.0 - 4.0, 2.0)))

    def test_sphericalbesselfunction(self):
        r = torch.linspace(1.0, 5.0, 11)
        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=False)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 9]

        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=True)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 3]

    def test_sphericalbesselfunction_smooth(self):
        r = torch.linspace(1.0, 5.0, 11)
        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=False)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 9]

        rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=True)
        rbf = rbf_sb(r)
        assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 3]

    def test_spherical_harmonic_function(self):
        theta = torch.linspace(-1, 1, 10)
        phi = torch.linspace(0, 2 * np.pi, 10)
        abf_sb = SphericalHarmonicsFunction(max_l=3, use_phi=True)
        abf = abf_sb(theta, phi)
        assert [abf.size(dim=0), abf.size(dim=1)] == [10, 9]


class TestSphericalBesselHarmonicsFunction:
    def test_spherical_bessel_harmonics_function(self):
        r = torch.empty(10).normal_()
        sbf = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=False)
        res = sbf(r)
        res2 = sbf.rbf_j0(r, cutoff=5.0, max_n=3)
        assert np.allclose(res[:, :3].numpy().ravel(), res2.numpy().ravel(), atol=1e-07)

        assert res.numpy().shape == (10, 9)

        sbf2 = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=True)

        res2 = sbf2(r)
        assert tuple(res2.shape) == (10, 3)

        shf = SphericalHarmonicsFunction(max_l=3, use_phi=True)
        res_shf = shf(costheta=torch.linspace(-1, 1, 10), phi=torch.linspace(0, 2 * np.pi, 10))

        assert res_shf.numpy().shape == (10, 9)
        combined = combine_sbf_shf(res, res_shf, max_n=3, max_l=3, use_phi=True)

        assert combined.shape == (10, 27)

        res_shf2 = SphericalHarmonicsFunction(max_l=3, use_phi=False)(
            costheta=torch.linspace(-1, 1, 10), phi=torch.linspace(0, 2 * np.pi, 10)
        )
        combined = combine_sbf_shf(res, res_shf2, max_n=3, max_l=3, use_phi=False)

        assert combined.shape == (10, 9)
        rdf = spherical_bessel_smooth(r, cutoff=5.0, max_n=3)
        assert rdf.numpy().shape == (10, 3)

    def test_spherical_bessel_with_harmonics(self):
        s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        element_types = get_element_list([s])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, state = p2g.get_graph(s)
        g1 = graph
        state1 = state
        bond_vec, bond_dist = compute_pair_vector_and_distance(g1)
        g1.edata["bond_dist"] = bond_dist
        g1.edata["bond_vec"] = bond_vec
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        l_g1 = create_line_graph(g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 9]

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=2, cutoff=5.0, use_smooth=False, use_phi=True)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 12]

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=True, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 9]

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=True, use_phi=True)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 27]
