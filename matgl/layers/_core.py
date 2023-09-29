"""Implementations of multi-layer perceptron (MLP) and other helper classes."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from dgl import DGLGraph, broadcast_edges, softmax_edges, sum_edges
from torch import nn
from torch.nn import LSTM, Linear, Module

from matgl.layers._norm import GraphNorm

if TYPE_CHECKING:
    from collections.abc import Sequence


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: list[int],
        activation: nn.Module | None = None,
        normalization: Literal["graph"] | None = None,
        activate_last: bool = False,
        normalize_last: bool = False,
        normalize_hidden: bool = False,
        use_bias: bool = True,
        bias_last: bool = True,
    ) -> None:
        """
        Args:
            dims: Dimensions of each layer of MLP.
            activation: activation: Activation function.
            normalization: normalization name.
            activate_last: Whether to apply activation to last layer.
            normalize_last: Whether to normalize output of last layer.
            normalize_hidden: Whether to normalize output of hidden layers.
            use_bias: Whether to use bias.
            bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if normalization == "graph" else None
        self.activation = activation if activation is not None else nn.Identity()
        self.activate_last = activate_last
        self.normalize_last = normalize_last

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=use_bias))
                if normalize_hidden and normalization == "graph":
                    self.norm_layers.append(GraphNorm(out_dim))
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=use_bias and bias_last))
                if normalize_last and normalization == "graph":
                    self.norm_layers.append(GraphNorm(out_dim))

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f'MLP({", ".join(dims)})'

    @property
    def last_linear(self) -> Linear | None:
        """:return: The last linear layer."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        raise RuntimeError

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP."""
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs: torch.Tensor, graph: dgl.Graph | None = None) -> torch.Tensor:
        """Applies all layers in turn.

        Args:
            inputs: input feature tensor.
            graph: graph of model.

        Returns:
            output feature tensor.
        """
        x = inputs
        for i in range(self._depth - 1):
            x = self.layers[i](x)
            if self.norm_layers is not None:
                x = self.norm_layers[i](x, graph)
            x = self.activation(x)

        x = self.layers[-1](x)
        if self.normalize_last:
            x = self.norm_layers[-1](x, graph)
        if self.activate_last:
            x = self.activation(x)
        return x


class GatedMLP(nn.Module):
    """An implementation of a Gated multi-layer perceptron."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activation: nn.Module | None = None,
        normalization: Literal["graph"] | None = None,
        activate_last: bool = True,
        normalize_last: bool = False,
        normalize_hidden: bool = False,
        use_bias: bool = True,
        bias_last: bool = True,
    ):
        """
        Args:
            in_feats: Input features.
            dims: Dimensions of each layer of MLP.
            activation: non-linear activation module.
            normalization: normalization name.
            activate_last: Whether to apply activation to last layer.
            normalize_last: Whether to normalize output of last layer.
            normalize_hidden: Whether to normalize hidden layers.
            use_bias: Whether to use a bias in linear layers.
            bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.use_bias = use_bias
        self.activate_last = activate_last

        activation = activation if activation is not None else nn.SiLU()
        self.layers = MLP(
            self.dims,
            activation=activation,
            normalization=normalization,
            activate_last=True,
            normalize_last=normalize_last,
            normalize_hidden=normalize_hidden,
            use_bias=use_bias,
            bias_last=bias_last,
        )
        self.gates = MLP(
                self.dims,
                activation,
                normalization=normalization,
                activate_last=False,
                normalize_last=normalize_last,
                normalize_hidden=normalize_hidden,
                use_bias=use_bias,
                bias_last=bias_last,
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, graph: dgl.Graph | None = None) -> torch.Tensor:
        return self.layers(inputs, graph) * self.sigmoid(self.gates(inputs, graph))


class EdgeSet2Set(Module):
    """Implementation of Set2Set."""

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
        self.lstm = LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, g: DGLGraph, feat: torch.Tensor):
        """Defines the computation performed at every call.

        :param g: Input graph
        :param feat: Input features.
        :return: One hot vector
        """
        with g.local_scope():
            batch_size = g.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_edges(g, q)).sum(dim=-1, keepdim=True)
                g.edata["e"] = e
                alpha = softmax_edges(g, "e")
                g.edata["r"] = feat * alpha
                readout = sum_edges(g, "r")
                q_star = torch.cat([q, readout], dim=-1)

            return q_star
