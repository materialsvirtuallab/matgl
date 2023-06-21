from __future__ import annotations

import dgl
import numpy as np
import torch

from matgl.utils.maths import (
    SPHERICAL_BESSEL_ROOTS,
    broadcast_states_to_atoms,
    broadcast_states_to_bonds,
    get_range_indices_from_n,
    get_segment_indices_from_n,
    repeat_with_n,
    spherical_bessel_roots,
    unsorted_segment_fraction,
)


def test_spherical_bessel_roots():
    roots = spherical_bessel_roots(max_l=1, max_n=5)
    roots2 = SPHERICAL_BESSEL_ROOTS
    assert np.allclose(roots2[0, :5], roots.ravel())


def test_torch_operations():
    ns = torch.tensor([2, 3])
    assert [0, 0, 1, 1, 1] == get_segment_indices_from_n(ns).tolist()
    ns = torch.tensor([2, 3])
    assert [0, 1, 0, 1, 2] == get_range_indices_from_n(ns).tolist()
    assert repeat_with_n(torch.tensor([[0, 0], [1, 1], [2, 2]]), torch.tensor([1, 2, 3])).tolist() == [
        [0, 0],
        [1, 1],
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
    ]


def test_segments():
    x = torch.tensor([1.0, 1.0, 2.0, 3.0])
    res = unsorted_segment_fraction(x, torch.tensor([0, 0, 1, 1]), 2)
    assert np.allclose(res.tolist(), [0.5, 0.5, 0.4, 0.6])


def test_broadcast():
    src_ids = torch.tensor([2, 3, 4])
    dst_ids = torch.tensor([1, 2, 3])
    g = dgl.graph((src_ids, dst_ids))
    state_attr = torch.tensor([0.0, 0.0])
    broadcasted_state_feat = broadcast_states_to_bonds(g, state_attr)
    assert [broadcasted_state_feat.size(dim=0), broadcasted_state_feat.size(dim=1)] == [3, 2]
    broadcasted_state_feat = broadcast_states_to_atoms(g, state_attr)
    assert [broadcasted_state_feat.size(dim=0), broadcasted_state_feat.size(dim=1)] == [5, 2]
