"""Readout layer in PYG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from matgl.layers import GatedMLP

from ._core_pyg import Set2Set

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data


class ReduceReadOutPYG(nn.Module):
    """Reduce node or edge attributes into lower dimensional tensors as readout in PyTorch Geometric.
    This could be summing up the nodes or edges, or taking the mean, etc.
    """

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        """
        Args:
            op (str): Operation for the reduction ('mean', 'sum', or 'max').
            field (str): Field to perform the reduction ('node_feat' or 'edge_feat').
        """
        super().__init__()
        self.op = op
        self.field = field
        if op not in ["mean", "sum", "max"]:
            raise ValueError("op must be 'mean', 'sum', or 'max'")
        if field not in ["node_feat"]:
            raise ValueError("field must be 'node_feat'")

        # Map operation to PyG pooling function
        self.pool_fn = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[op]

    def forward(self, graph: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            graph (Data): PyG Data object containing x, edge_attr, edge_index, and batch.

        Returns:
            torch.Tensor: Pooled features, shape (num_graphs, feature_dim).
        """
        if not hasattr(graph, "node_feat") or graph.node_feat is None:
            raise ValueError("Data object must contain node features (graph.node_feat)")
        return self.pool_fn(graph.node_feat, graph.batch)


class Set2SetReadOutPYG(nn.Module):
    """Set2Set readout function for PyTorch Geometric."""

    def __init__(
        self,
        in_feats: int,
        n_iters: int,
        n_layers: int,
        field: str,
    ):
        """
        Args:
            in_feats (int): Length of input feature vector.
            n_iters (int): Number of LSTM steps.
            n_layers (int): Number of LSTM layers.
            field (str): Field to perform readout ('node_feat' or 'edge_feat').
        """
        super().__init__()
        self.field = field
        self.n_iters = n_iters
        self.n_layers = n_layers
        if field == "node_feat":
            self.set2set = Set2Set(in_feats, n_iters, n_layers)
        else:
            raise ValueError("Field must be 'node_feat'")

    def forward(self, graph: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            graph: PyG Data object containing x, edge_index, edge_attr, and batch.

        Returns:
            Pooled features, shape (num_graphs, 2 * in_feats).
        """
        return self.set2set(graph.node_feat, graph.batch)


class WeightedReadOutPYG(nn.Module):
    """Feed node features into Gated MLP as readout for atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        """
        Args:
            in_feats: input features (nodes).
            dims: NN architecture for Gated MLP.
            num_targets: number of target properties.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, graph: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            graph (Data): PyG Data object containing node features (data.node_feat).

        Returns:
            atomic_properties (torch.Tensor): Per-node atomic properties, shape (num_nodes, num_targets).
        """
        atomic_properties = self.gated(graph.node_feat)
        return atomic_properties
