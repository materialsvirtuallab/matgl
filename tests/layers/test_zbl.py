"""United tests for ZBL repulsion"""
from __future__ import annotations

import dgl
import matgl
import pytest
import torch
from matgl.layers import NuclearRepulsion


@pytest.fixture()
def example_data():
    element_types = "H"
    g = dgl.graph(([0, 1], [1, 0]))
    g.ndata["node_type"] = torch.tensor([0, 0], dtype=matgl.int_th)
    g.edata["bond_dist"] = torch.tensor([1.0, 1.0], dtype=matgl.float_th)
    return element_types, g


def test_nuclear_repulsion(example_data):
    element_types, graph = example_data

    r_cut = 3.0
    nuclear_repulsion = NuclearRepulsion(r_cut=r_cut, trainable=True)

    energy = nuclear_repulsion(element_types, graph)

    assert energy.shape == torch.Size([])
