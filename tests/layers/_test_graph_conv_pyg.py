from __future__ import annotations

from torch import nn

import matgl
from matgl.layers import BondExpansion
from matgl.layers._embedding_pyg import (
    TensorEmbeddingPYG,
)
from matgl.layers._graph_convolution_pyg import (
    TensorNetInteractionPYG,
)


class TestGraphConv:
    def test_tensornet_interaction(self, graph_Mo_pyg):
        s, g1, state = graph_Mo_pyg
        bond_expansion = BondExpansion(rbf_type="SphericalBessel", max_n=3, max_l=3, cutoff=4.0, smooth=True)
        g1.edge_attr = bond_expansion(g1.bond_dist)

        tensor_embedding = TensorEmbeddingPYG(
            units=64, degree_rbf=3, activation=nn.SiLU(), ntypes_node=1, cutoff=5.0, dtype=matgl.float_th
        )

        X, state_feat = tensor_embedding(g1, state)
        interaction = TensorNetInteractionPYG(
            num_rbf=3,
            units=64,
            activation=nn.SiLU(),
            equivariance_invariance_group="O(3)",
            dtype=matgl.float_th,
            cutoff=5.0,
        )
        X = interaction(g1, X)

        assert [X.shape[0], X.shape[1], X.shape[2], X.shape[3]] == [2, 64, 3, 3]

        interaction_so3 = TensorNetInteractionPYG(
            num_rbf=3,
            units=64,
            activation=nn.SiLU(),
            equivariance_invariance_group="SO(3)",
            dtype=matgl.float_th,
            cutoff=5.0,
        )
        X = interaction_so3(g1, X)

        assert [X.shape[0], X.shape[1], X.shape[2], X.shape[3]] == [2, 64, 3, 3]
