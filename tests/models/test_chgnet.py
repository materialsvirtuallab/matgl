from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.models import CHGNet


@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("learn_basis", [True, False])
@pytest.mark.parametrize("bond_dim", [None, (16,)])
@pytest.mark.parametrize("angle_dim", [None, (16,)])
@pytest.mark.parametrize("activation", ["swish", "softplus2"])
def test_model(graph_MoS, activation, angle_dim, bond_dim, learn_basis, dropout):
    structure, graph, state = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = CHGNet(
        element_types=["Mo", "S"],
        activation_type=activation,
        bond_update_hidden_dims=bond_dim,
        learn_basis=learn_basis,
        angle_update_hidden_dims=angle_dim,
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
    converter = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)

    g, lattice, _ = converter.get_graph(structure)
    g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lattice[0])
    g.ndata["pos"] = g.ndata["frac_coords"] @ lattice[0]

    g1, lattice2, _ = converter.get_graph(supercell1)
    g1.edata["pbc_offshift"] = torch.matmul(g1.edata["pbc_offset"], lattice2[0])
    g1.ndata["pos"] = g1.ndata["frac_coords"] @ lattice2[0]

    g2, lattice3, _ = converter.get_graph(supercell2)
    g2.edata["pbc_offshift"] = torch.matmul(g2.edata["pbc_offset"], lattice3[0])
    g2.ndata["pos"] = g2.ndata["frac_coords"] @ lattice3[0]

    out, swout = model(g)
    out1, swout1 = model(g1)
    out2, swout2 = model(g2)

    assert not torch.allclose(out, out1)
    assert not torch.allclose(out, out2)

    assert torch.allclose(out / g.num_nodes(), out1 / g1.num_nodes())
    assert torch.allclose(out / g.num_nodes(), out2 / g2.num_nodes())

    assert len(swout) == g.num_nodes()
    assert len(swout1) == g1.num_nodes()
    assert len(swout2) == g2.num_nodes()

    assert torch.allclose(
        torch.unique(torch.round(swout, decimals=4), sorted=True),
        torch.unique(torch.round(swout2, decimals=4), sorted=True),
    )
    assert torch.allclose(
        torch.unique(torch.round(swout, decimals=4), sorted=True),
        torch.unique(torch.round(swout2, decimals=4), sorted=True),
    )
