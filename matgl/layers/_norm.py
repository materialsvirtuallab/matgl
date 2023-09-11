"""Normalization modules."""


from __future__ import annotations


import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    """Graph normalization layer.

    Following the following paper:
        https://proceedings.mlr.press/v139/cai21e.html
    """

    def __init__(self, hidden_dim=300):
        """
        Init GraphNorm layer.

        Args:
            hidden_dim: dimension of learnable normalization parameters
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, graph, tensor):
        """Forward pass.

        Args:
            graph (dgl.DGLGraph): graph
            tensor:

        Returns:

        """
        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = tensor - mean
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        # return sub / std
        return self.weight * sub / std + self.bias