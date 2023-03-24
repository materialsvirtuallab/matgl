from __future__ import annotations

import unittest

import dgl
import numpy as np
import torch

from matgl.utils.maths import (
    SPHERICAL_BESSEL_ROOTS,
    GaussianExpansion,
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    broadcast_states_to_atoms,
    broadcast_states_to_bonds,
    combine_sbf_shf,
    get_range_indices_from_n,
    get_segment_indices_from_n,
    repeat_with_n,
    spherical_bessel_roots,
    spherical_bessel_smooth,
    unsorted_segment_fraction,
)


class TestMath(unittest.TestCase):
    def test_spherical_bessel_roots(self):
        roots = spherical_bessel_roots(max_l=1, max_n=5)
        roots2 = SPHERICAL_BESSEL_ROOTS
        self.assertTrue(np.allclose(roots2[0, :5], roots.ravel()))

    def test_spherical_bessel_harmonics_function(self):
        r = torch.empty(10).normal_()
        sbf = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=False)
        res = sbf(r)
        res2 = sbf.rbf_j0(r, cutoff=5.0, max_n=3)
        self.assertTrue(np.allclose(res[:, :3].numpy().ravel(), res2.numpy().ravel()))

        self.assertTrue(res.numpy().shape == (10, 9))

        sbf2 = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=True)

        res2 = sbf2(r)
        self.assertTupleEqual(tuple(res2.shape), (10, 3))

        shf = SphericalHarmonicsFunction(max_l=3, use_phi=True)
        res_shf = shf(costheta=np.linspace(-1, 1, 10), phi=np.linspace(0, 2 * np.pi, 10))

        self.assertTrue(res_shf.numpy().shape == (10, 9))
        combined = combine_sbf_shf(res, res_shf, max_n=3, max_l=3, use_phi=True, use_smooth=False)

        self.assertTrue(combined.shape == (10, 27))

        res_shf2 = SphericalHarmonicsFunction(max_l=3, use_phi=False)(
            costheta=np.linspace(-1, 1, 10), phi=np.linspace(0, 2 * np.pi, 10)
        )
        combined = combine_sbf_shf(res, res_shf2, max_n=3, max_l=3, use_phi=False, use_smooth=False)

        self.assertTrue(combined.shape == (10, 9))
        rdf = spherical_bessel_smooth(r, cutoff=5.0, max_n=3)
        self.assertTrue(rdf.numpy().shape == (10, 3))

    def test_torch_operations(self):
        ns = torch.tensor([2, 3])
        self.assertListEqual([0, 0, 1, 1, 1], get_segment_indices_from_n(ns).tolist())
        ns = torch.tensor([2, 3])
        self.assertListEqual([0, 1, 0, 1, 2], get_range_indices_from_n(ns).tolist())
        self.assertListEqual(
            repeat_with_n(torch.tensor([[0, 0], [1, 1], [2, 2]]), torch.tensor([1, 2, 3])).tolist(),
            [[0, 0], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2]],
        )

    def test_segments(self):
        x = torch.tensor([1.0, 1.0, 2.0, 3.0])
        res = unsorted_segment_fraction(x, torch.tensor([0, 0, 1, 1]), 2)
        self.assertTrue(np.allclose(res.tolist(), [0.5, 0.5, 0.4, 0.6]))

    #        res = unsorted_segment_softmax(x, torch.tensor([0, 0, 1, 1]), 2)
    #        np.testing.assert_array_almost_equal(res, [0.5, 0.5, 0.26894143, 0.7310586])

    def test_broadcast(self):
        src_ids = torch.tensor([2, 3, 4])
        dst_ids = torch.tensor([1, 2, 3])
        g = dgl.graph((src_ids, dst_ids))
        state_attr = torch.tensor([0.0, 0.0])
        broadcasted_state_feat = broadcast_states_to_bonds(g, state_attr)
        self.assertListEqual([broadcasted_state_feat.size(dim=0), broadcasted_state_feat.size(dim=1)], [3, 2])
        broadcasted_state_feat = broadcast_states_to_atoms(g, state_attr)
        self.assertListEqual([broadcasted_state_feat.size(dim=0), broadcasted_state_feat.size(dim=1)], [5, 2])

    def test_gaussian_expansion(self):
        bond_dist = torch.tensor([1.0])
        dist_converter = GaussianExpansion()
        expanded_dist = dist_converter(bond_dist)
        # check the shape of a vector
        self.assertTrue(np.allclose([expanded_dist.size(dim=0), expanded_dist.size(dim=1)], [1, 20]))
        # check the first value of expanded distance
        self.assertTrue(np.allclose(expanded_dist[0][0], np.exp(-0.5 * np.power(1.0 - 0.0, 2.0))))
        # check the last value of expanded distance
        self.assertTrue(np.allclose(expanded_dist[0][-1], np.exp(-0.5 * np.power(1.0 - 4.0, 2.0))))


if __name__ == "__main__":
    unittest.main()
