from __future__ import annotations

import dgl
import numpy as np
import torch
from matgl.layers._atom_ref import AtomRef


class TestAtomRef:
    def test_atom_ref(self, graph_MoSH):
        _, g1, _ = graph_MoSH
        element_ref = AtomRef(torch.tensor([0.5, 1.0, 2.0]))

        atom_ref = element_ref(g1)
        assert atom_ref == 3.5

    def test_atom_ref_fit(self, graph_MoSH):
        _, g1, _ = graph_MoSH
        element_ref = AtomRef(torch.tensor([0.5, 1.0, 2.0]))
        properties = torch.tensor([2.0, 2.0])
        bg = dgl.batch([g1, g1])
        element_ref.fit([g1, g1], properties)

        atom_ref = element_ref(bg)
        assert list(np.round(atom_ref.numpy())) == [2.0, 2.0]

    def test_atom_ref_with_states(self, graph_MoSH):
        _, g1, _ = graph_MoSH
        element_ref = AtomRef(torch.tensor([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
        state_label = torch.tensor([1])
        atom_ref = element_ref(g1, state_label)
        assert atom_ref == 10
