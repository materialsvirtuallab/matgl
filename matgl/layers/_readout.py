"""Readout layer for M3GNet."""

from __future__ import annotations

import dgl
import torch
from dgl.nn import Set2Set
from torch import nn

from matgl.layers._core import EdgeSet2Set, GatedMLP


class Set2SetReadOut(nn.Module):
    """The Set2Set readout function."""

    def __init__(
        self,
        n_iters: int,
        n_layers: int,
        field: str,
    ):
        """
        Args:
            n_iters (int): Number of LSTM steps
            n_layers (int): Number of layers.
            field (str): Field of graph to perform the readout.
        """
        super().__init__()
        self.field = field
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.node_s2s: dict[int, Set2Set] = {}
        self.edge_s2s: dict[int, EdgeSet2Set] = {}

    def forward(self, g: dgl.DGLGraph):
        if self.field == "node_feat":
            in_feats = g.ndata["node_feat"].size(dim=1)
            if in_feats not in self.node_s2s:  # init s2s only once to remove stochasticity
                self.node_s2s[in_feats] = Set2Set(in_feats, n_iters=self.n_iters, n_layers=self.n_layers)
            out_tensor = self.node_s2s[in_feats](g, g.ndata["node_feat"])  # type: ignore
        elif self.field == "edge_feat":
            in_feats = g.edata["edge_feat"].size(dim=1)
            if in_feats not in self.edge_s2s:  # init s2s only once to remove stochasticity
                self.edge_s2s[in_feats] = EdgeSet2Set(in_feats, n_iters=self.n_iters, n_layers=self.n_layers)
            out_tensor = self.edge_s2s[in_feats](g, g.edata["edge_feat"])  # type: ignore
        return out_tensor


class ReduceReadOut(nn.Module):
    """Reduce atom or bond attributes into lower dimensional tensors as readout.
    This could be summing up the atoms or bonds, or taking the mean, etc.
    """

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        """
        Args:
            op (str): op for the reduction
            field (str): Field of graph to perform the reduction.
        """
        super().__init__()
        self.op = op
        self.field = field

    def forward(self, g: dgl.DGLGraph):
        """Args:
            g: DGL graph.

        Returns:
            torch.tensor.
        """
        if self.field == "node_feat":
            reduced_tensor = dgl.readout_nodes(g, feat="node_feat", op=self.op)
        elif self.field == "edge_feat":
            reduced_tensor = dgl.readout_edges(g, feat="edge_feat", op=self.op)
        return reduced_tensor


class WeightedReadOut(nn.Module):
    """Feed node features into Gated MLP as readout."""

    def __init__(self, in_feats: int, dims: list[int], num_targets: int):
        """
        Args:
            in_feats: input features (nodes)
            dims: NN architecture for Gated MLP
            num_targets: number of target properties.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, g: dgl.DGLGraph):
        """Args:
            g: DGL graph.

        Returns:
            atomic_properties: torch.Tensor.
        """
        atomic_properties = self.gated(g.ndata["node_feat"])
        return atomic_properties


class WeightedReadOutPair(nn.Module):
    """Feed the average of atomic features i and j into weighted readout layer."""

    def __init__(self, in_feats, dims, num_targets, activation=None):
        super().__init__()
        self.in_feats = in_feats
        self.activation = activation
        self.num_targets = num_targets
        self.dims = [*dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, g: dgl.DGLGraph):
        num_nodes = g.ndata["node_feat"].size(dim=0)
        pair_properties = torch.zeros(num_nodes, num_nodes, self.num_targets)
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                in_features = (g.ndata["node_feat"][i][:] + g.ndata["node_feat"][j][:]) / 2.0
                pair_properties[i][j][:] = self.gated(in_features)[:]
                pair_properties[j][i][:] = pair_properties[i][j][:]
        return pair_properties
