from __future__ import annotations

import os
import pytest

import torch

from matgl.models import CHGNet
from matgl.ext.pymatgen import Structure2Graph


@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("learn_basis", [True, False])
@pytest.mark.parametrize("atom2bond_dim", [None, (16,)])
@pytest.mark.parametrize("angle_dim", [None, (16,)])
@pytest.mark.parametrize("activation", ["swish", "softplus2"])
def test_model(graph_MoS, activation, angle_dim, atom2bond_dim, learn_basis, dropout):
    structure, graph, state = graph_MoS
    model = CHGNet(
        element_types=["Mo", "S"], activation_type=activation,
        atom2bond_dim=atom2bond_dim,
        learn_basis=learn_basis,
        angle_layer_hidden_dims=angle_dim,
        conv_dropout=dropout,
    )
    global_out, site_wise_out = model(graph=graph)
    assert torch.numel(global_out) == 1
    assert torch.numel(site_wise_out) == graph.num_nodes()
    model.save(".")
    CHGNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")


@pytest.mark.parametrize("structure", ["LiFePO4", "BaNiO3", "MoS"])
def test_prediction_validity(structure, request):
    structure = request.getfixturevalue(structure)
    supercell1 = structure.copy()
    supercell1.make_supercell([2, 4, 3])
    supercell2 = structure.copy()
    supercell2.make_supercell(2)

    model = CHGNet()
    converter = Structure2Graph(element_types=model.element_types, cutoff=5.0)
    g, _ = converter.get_graph(structure)
    g1, _ = converter.get_graph(supercell1)
    g2, _ = converter.get_graph(supercell2)

    out, swout = model(g)
    out1, swout1 = model(g1)
    out2, swout2 = model(g2)

    assert torch.allclose(out / g.num_nodes(), out1 / g1.num_nodes())
    assert torch.allclose(out / g.num_nodes(), out2 / g2.num_nodes())

    assert len(swout) == g.num_nodes()
    assert len(swout1) == g1.num_nodes()
    assert len(swout2) == g2.num_nodes()

    assert torch.allclose(
        torch.unique(torch.round(swout, decimals=4), sorted=True),
        torch.unique(torch.round(swout2, decimals=4), sorted=True)
    )
    assert torch.allclose(
        torch.unique(torch.round(swout, decimals=4), sorted=True),
        torch.unique(torch.round(swout2, decimals=4), sorted=True),
    )
