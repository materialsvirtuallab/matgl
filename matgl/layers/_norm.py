"""Normalization modules."""


from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import matgl

if TYPE_CHECKING:
    import dgl


class GraphNorm(nn.Module):
    """Graph normalization layer.

    Following the following paper:
        https://proceedings.mlr.press/v139/cai21e.html
    """

    def __init__(self, input_dim: int, eps: float = 1e-5):
        """
        Init GraphNorm layer.

        Args:
            input_dim: dimension of input features
            eps: value added to denominator for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.mean_scale = nn.Parameter(torch.ones(input_dim))

    def forward(self, features: torch.Tensor, graph: dgl.DGLGraph):
        """Forward pass.

        Args:
            features (torch.Tensor): features
            graph (dgl.DGLGraph): graph

        Returns:
            torch.Tensor: normalized features
        """
        batch_list = graph.batch_num_nodes().to(matgl.int_th)
        batch_index = torch.arange(graph.batch_size).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (features.dim() - 1)).expand_as(features)
        mean = torch.zeros(graph.batch_size, *features.shape[1:])
        mean = mean.scatter_add_(0, batch_index, features)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        out = features - mean * self.mean_scale
        std = torch.zeros(graph.batch_size, *features.shape[1:])
        std = std.scatter_add_(0, batch_index, out.pow(2))
        std = ((std.T / batch_list).T + self.eps).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        return self.weight * out / std + self.bias
