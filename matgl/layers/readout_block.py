"""
Readout layer for M3GNet
"""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn
from dgl.nn import Set2Set

from matgl.layers.core import EdgeSet2Set, GatedMLP


class Set2SetReadOut(nn.Module):
    """
    The Set2Set readout function
    """

    def __init__(
        self,
        num_steps: int,
        num_layers: int,
        field: str,
    ):
        """
        Paramaters:
            num_steps (int): number of LSTM steps
            field (str): the field of MaterialGraph to perform the readout
        """
        super().__init__()
        self.field = field
        self.num_steps = num_steps
        self.num_layers = num_layers

    def forward(self, g):
        s2s_kwargs = {"n_iters": self.num_steps, "n_layers": self.num_layers}
        if self.field == "node_feat":
            in_feats = g.ndata["node_feat"].size(dim=1)
            set2set = Set2Set(in_feats, **s2s_kwargs)
            out_tensor = set2set(g, g.ndata["node_feat"])
        elif self.field == "edge_feat":
            in_feats = g.edata["edge_feat"].size(dim=1)
            set2set = EdgeSet2Set(in_feats, **s2s_kwargs)
            out_tensor = set2set(g, g.edata["edge_feat"])
        return out_tensor


class ReduceReadOut(nn.Module):
    """
    Reduce atom or bond attributes into lower dimensional tensors as readout.
    This could be summing up the atoms or bonds, or taking the mean, etc.
    """

    def __init__(self, op: str = "mean", field: str = "node_feat", device: str = "cpu"):
        super().__init__()
        """
        Parameters:
            op (str): op for the reduction
            field (str): the field of MaterialGraph to perform the reduction
        """
        self.op = op
        self.field = field

    def forward(self, g):
        """
        Args:
            g: DGL graph
        Returns:
            torch.tensor
        """
        if self.field == "node_feat":
            reduced_tensor = dgl.readout_nodes(g, feat="node_feat", op=self.op)
        elif self.field == "edge_feat":
            reduced_tensor = dgl.readout_edges(g, feat="edge_feat", op=self.op)
        return reduced_tensor


class WeightedReadOut(nn.Module):
    """
    Feed node features into Gated MLP as readout.
    """

    def __init__(self, in_feats: int, dims: list[int], num_targets: int, device: str = "cpu"):
        """
        Parameters:
           in_feats: input features (nodes)
           dims: NN architecture for Gated MLP
           num_targets: number of target properties
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims] + [num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False, device=device)

    def forward(self, g):
        """
        Args:
            g: DGL graph
        Returns:
            atomic_prperties: torch.tensor
        """
        atomic_properties = self.gated(g.ndata["node_feat"])
        return atomic_properties


class WeightedReadOutPair(nn.Module):
    """
    Feed the average of atomic features i and j into weighted readout layer.
    """

    def __init__(self, in_feats, dims, num_targets, activation=None):
        super().__init__()
        self.in_feats = in_feats
        self.activation = activation
        self.num_targets = num_targets
        self.dims = [*dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, g):
        num_nodes = g.ndata["node_feat"].size(dim=0)
        pair_properties = torch.zeros(num_nodes, num_nodes, self.num_targets)
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                in_features = (g.ndata["node_feat"][i][:] + g.ndata["node_feat"][j][:]) / 2.0
                pair_properties[i][j][:] = self.gated(in_features)[:]
                pair_properties[j][i][:] = pair_properties[i][j][:]
        return pair_properties
