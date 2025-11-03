from __future__ import annotations

import torch
import torch.nn as nn

from matgl.layers._readout_pyg import (
    ReduceReadOutPYG,
    WeightedAtomReadOutPYG,
    WeightedReadOutPYG,
)


class TestReadOut:
    def test_weighted_readout(self, graph_MoS_pyg):
        _, g1, _ = graph_MoS_pyg
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        read_out = WeightedReadOutPYG(in_feats=64, dims=[64, 64], num_targets=4)
        atomic_properties = read_out(g1)
        assert [atomic_properties.size(dim=0), atomic_properties.size(dim=1)] == [2, 4]

    def test_reduce_readout(self, graph_MoS_pyg):
        _, g1, _ = graph_MoS_pyg
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        read_out = ReduceReadOutPYG(op="mean", field="node_feat")
        output = read_out(g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 64]

    def test_wighted_atom_readout(self, graph_MoS_pyg):
        _, g1, _ = graph_MoS_pyg
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        read_out = WeightedAtomReadOutPYG(in_feats=64, dims=[64, 64], activation=nn.SiLU())
        output = read_out(g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 64]
