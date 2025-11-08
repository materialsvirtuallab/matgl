from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.layers._atom_ref_pyg import AtomRefPYG


class TestAtomRef:
    def test_atom_ref(self, graph_MoSH_pyg):
        _, g1, _ = graph_MoSH_pyg
        element_ref = AtomRefPYG(torch.tensor([0.5, 1.0, 2.0]))

        atom_ref = element_ref(g1)
        assert atom_ref == 3.5

    def test_atom_ref_without_property_offset(self, graph_MoSH_pyg):
        _, g1, _ = graph_MoSH_pyg
        element_ref = AtomRefPYG()

        atom_ref = element_ref(g1)
        assert atom_ref == 0.0

    def test_atom_ref_property_offset_as_list(self, graph_MoSH_pyg):
        _, g1, _ = graph_MoSH_pyg
        element_ref = AtomRefPYG([0.5, 1.0, 2.0])

        atom_ref = element_ref(g1)
        assert atom_ref == 3.5

    def test_atom_ref_fit(self, graph_MoSH_pyg):
        _, g1, _ = graph_MoSH_pyg
        element_ref = AtomRefPYG(torch.tensor([0.5, 1.0, 2.0]))
        properties = torch.tensor([2.0, 2.0])
        bg = Batch.from_data_list([g1, g1])
        element_ref.fit([g1, g1], properties)

        atom_ref = element_ref(bg)
        assert list(np.round(atom_ref.numpy())) == [2.0, 2.0]

    def test_atom_ref_with_states(self, graph_MoSH_pyg):
        _, g1, _ = graph_MoSH_pyg
        element_ref = AtomRefPYG(torch.tensor([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
        state_label = torch.tensor([1])
        atom_ref = element_ref(g1, state_label)
        assert atom_ref == 10
