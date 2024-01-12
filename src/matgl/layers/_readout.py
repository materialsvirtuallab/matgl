"""Readout layer for M3GNet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from dgl.nn import Set2Set
from torch import nn

from ._core import EdgeSet2Set, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence


class Set2SetReadOut(nn.Module):
    """The Set2Set readout function."""

    def __init__(
        self,
        in_feats: int,
        n_iters: int,
        n_layers: int,
        field: Literal["node_feat", "edge_feat"],
    ):
        """
        Args:
            in_feats (int): length of input feature vector
            n_iters (int): Number of LSTM steps
            n_layers (int): Number of layers.
            field (str): Field of graph to perform the readout.
        """
        super().__init__()
        self.field = field
        self.n_iters = n_iters
        self.n_layers = n_layers
        if field == "node_feat":
            self.set2set = Set2Set(in_feats, n_iters, n_layers)
        elif field == "edge_feat":
            self.set2set = EdgeSet2Set(in_feats, n_iters, n_layers)
        else:
            raise ValueError("Field must be node_feat or edge_feat")

    def forward(self, g: dgl.DGLGraph):
        if self.field == "node_feat":
            return self.set2set(g, g.ndata["node_feat"])
        return self.set2set(g, g.edata["edge_feat"])


class ReduceReadOut(nn.Module):
    """Reduce atom or bond attributes into lower dimensional tensors as readout.
    This could be summing up the atoms or bonds, or taking the mean, etc.
    """

    def __init__(self, op: str = "mean", field: Literal["node_feat", "edge_feat"] = "node_feat"):
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
            return dgl.readout_nodes(g, feat="node_feat", op=self.op)
        return dgl.readout_edges(g, feat="edge_feat", op=self.op)


class WeightedReadOut(nn.Module):
    """Feed node features into Gated MLP as readout."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
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
