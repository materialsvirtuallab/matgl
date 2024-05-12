from __future__ import annotations

import os

import matgl
import numpy as np
import pytest
import torch
from matgl.ext.pymatgen import Structure2Graph
from matgl.models import CHGNet


class TestCHGNet:
    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    @pytest.mark.parametrize("learn_basis", [True, False])
    @pytest.mark.parametrize("bond_dim", [None, (16,)])
    @pytest.mark.parametrize("angle_dim", [None, (16,)])
    @pytest.mark.parametrize("activation", ["swish", "softplus2"])
    def test_model(self, graph_MoS, activation, angle_dim, bond_dim, learn_basis, dropout):
        structure, graph, state = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        for readout_field in ["atom_feat", "bond_feat", "angle_feat"]:
            for final_mlp_type in ["gated", "mlp"]:
                model = CHGNet(
                    element_types=["Mo", "S"],
                    activation_type=activation,
                    bond_update_hidden_dims=bond_dim,
                    learn_basis=learn_basis,
                    angle_update_hidden_dims=angle_dim,
                    conv_dropout=dropout,
                    readout_field=readout_field,
                    final_mlp_type=final_mlp_type,
                )
                global_out = model(g=graph)
                assert torch.numel(global_out) == 1
                assert torch.numel(graph.ndata["magmom"]) == graph.num_nodes()
                model.save(".")
                CHGNet.load(".")
                os.remove("model.pt")
                os.remove("model.json")
                os.remove("state.pt")

    def test_exceptions(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            _ = CHGNet(element_types=None, is_intensive=False, activation_type="whatever")

    @pytest.mark.parametrize("structure", ["LiFePO4", "BaNiO3", "MoS"])
    def test_prediction_validity(self, structure, request):
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

        out = model(g)
        out1 = model(g1)
        out2 = model(g2)

        assert not torch.allclose(out, out1)
        assert not torch.allclose(out, out2)

        assert torch.allclose(out / g.num_nodes(), out1 / g1.num_nodes())
        assert torch.allclose(out / g.num_nodes(), out2 / g2.num_nodes())

        assert len(g.ndata["magmom"]) == g.num_nodes()
        assert len(g1.ndata["magmom"]) == g1.num_nodes()
        assert len(g2.ndata["magmom"]) == g2.num_nodes()

        assert torch.allclose(
            torch.unique(torch.round(g.ndata["magmom"], decimals=4), sorted=True),
            torch.unique(torch.round(g2.ndata["magmom"], decimals=4), sorted=True),
        )
        assert torch.allclose(
            torch.unique(torch.round(g.ndata["magmom"], decimals=4), sorted=True),
            torch.unique(torch.round(g2.ndata["magmom"], decimals=4), sorted=True),
        )
