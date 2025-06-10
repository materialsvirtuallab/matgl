from __future__ import annotations

import pytest
import torch

from matgl.layers._readout_pyg import (
    ReduceReadOutPYG,
    Set2SetReadOutPYG,
    WeightedReadOutPYG,
)


class TestReadOut:
    def test_weighted_readout(self, graph_MoS_pyg):
        s, g1, state = graph_MoS_pyg
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        read_out = WeightedReadOutPYG(in_feats=64, dims=[64, 64], num_targets=4)
        atomic_properties = read_out(g1)
        assert [atomic_properties.size(dim=0), atomic_properties.size(dim=1)] == [2, 4]

    def test_reduce_readout(self, graph_MoS_pyg):
        s, g1, state = graph_MoS_pyg
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        read_out = ReduceReadOutPYG(op="mean", field="node_feat")
        output = read_out(g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 64]

    def test_set2set_readout(self, graph_MoS_pyg):
        s, g1, state = graph_MoS_pyg
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        g1.edge_feat = torch.rand(g1.num_edges, 64)
        g1.node_feat = torch.rand(g1.num_nodes, 64)
        g1.edge_feat = torch.rand(g1.num_edges, 64)
        g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
        read_out = Set2SetReadOutPYG(
            in_feats=64,
            n_iters=3,
            n_layers=3,
            field="node_feat",
        )
        output = read_out(g1)
        assert [output.size(dim=0), output.size(dim=1)] == [1, 128]

        with pytest.raises(ValueError, match="Field must be 'node_feat'"):
            Set2SetReadOutPYG(1, 2, 3, field="nonsense")
