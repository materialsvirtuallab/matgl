"""Readout layer in PYG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from torch_geometric.nn import Set2Set, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter

from matgl.layers import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data


class ReduceReadOut(nn.Module):
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


class WeightedReadOut(nn.Module):
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


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties in PyTorch Geometric."""

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """
        Args:
            in_feats: Input features (nodes).
            dims: NN architecture for Gated MLP.
            activation: Activation function for multi-layer perceptrons.
        """
        super().__init__()
        self.dims = [in_feats, *dims]
        self.activation = activation
        self.mlp = MLP(dims=self.dims, activation=self.activation, activate_last=True)
        self.weight = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Args:
            graph: PyG graph Data object.

        Returns:
            atomic_properties: Tensor of shape (num_graphs, output_dim).
        """
        # Apply MLP to node features
        h = self.mlp(graph.node_feat)  # Shape: (num_nodes, output_dim)

        # Compute weights for each node
        w = self.weight(graph.node_feat)  # Shape: (num_nodes, 1)

        # Weighted node features
        weighted_h = h * w  # Element-wise multiplication, shape: (num_nodes, output_dim)

        # Aggregate weighted node features per graph using global_add_pool
        h_g_sum = global_add_pool(weighted_h, graph.batch)  # Shape: (num_graphs, output_dim)

        return h_g_sum


class EdgeSet2Set(nn.Module):
    """Implementation of Set2Set for edges in PyG."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """:param input_dim: The size of each input sample.
        :param n_iters: The number of iterations.
        :param n_layers: The number of recurrent layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, n_layers, batch_first=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, graph: Data, feat: torch.Tensor):
        """Defines the computation performed at every call.

        :param graph: Input graph
        :param feat: Input features (edge features).
        :return: One hot vector
        """
        if hasattr(graph, "batch") and graph.batch is not None:
            # For edge features, we need to get the batch for each edge
            # In PyG, edges don't have a direct batch attribute, so we use the destination node's batch
            edge_batch = graph.batch[graph.edge_index[1]]
            batch_size = graph.batch.max().item() + 1
        else:
            edge_batch = torch.zeros(feat.size(0), dtype=torch.long, device=feat.device)
            batch_size = 1

        h = (
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
        )

        q_star = feat.new_zeros(batch_size, self.output_dim)

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)

            # Compute attention weights
            e = (feat * q[edge_batch]).sum(dim=-1, keepdim=True)
            # Softmax over edges in each graph
            # Subtract max for numerical stability
            e_max = scatter(e.squeeze(-1), edge_batch, dim=0, dim_size=batch_size, reduce="max")
            e_exp = torch.exp(e.squeeze(-1) - e_max[edge_batch])
            e_sum = scatter(e_exp, edge_batch, dim=0, dim_size=batch_size, reduce="sum")
            alpha = (e_exp / (e_sum[edge_batch] + 1e-8)).unsqueeze(-1)

            # Weighted sum
            weighted_feat = feat * alpha
            readout = scatter(weighted_feat, edge_batch, dim=0, dim_size=batch_size, reduce="sum")
            q_star = torch.cat([q, readout], dim=-1)

        return q_star


class Set2SetReadOut(nn.Module):
    """The Set2Set readout function for PyG."""

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

    def forward(self, graph: Data):
        if self.field == "node_feat":
            if not hasattr(graph, "node_feat") or graph.node_feat is None:
                raise ValueError("Data object must contain node features (graph.node_feat)")
            return self.set2set(graph.node_feat, graph.batch)
        if not hasattr(graph, "edge_feat") or graph.edge_feat is None:
            raise ValueError("Data object must contain edge features (graph.edge_feat)")
        return self.set2set(graph, graph.edge_feat)
