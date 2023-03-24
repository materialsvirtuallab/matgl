"""
Implementations of multi-layer perceptron (MLP) and other helper classes.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from dgl import DGLGraph, broadcast_edges, softmax_edges, sum_edges
from torch.nn import LSTM, Linear, Module, ModuleList


class MLP(nn.Module):
    """
    An implementation of a multi-layer perceptron.
    """

    def __init__(
        self,
        dims: list[int],
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        :param dims: Dimensions of each layer of MLP.
        :param activation: Activation function.
        :param activate_last: Whether to apply activation to last layer.
        :param bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        device = torch.device(device)
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True, device=device))

                if activation is not None:
                    self.layers.append(activation)  # type: ignore
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last, device=device))

                if activation is not None and activate_last:
                    self.layers.append(activation)  # type: ignore

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
        """
        :return: The last linear layer.
        """
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        return None

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP"""
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs):
        """
        Applies all layers in turn.

        :param inputs: Input tensor
        :return: Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x


class GatedMLP(nn.Module):
    """
    An implementation of a Gated multi-layer perceptron.
    """

    def __init__(
        self,
        in_feats: int,
        dims: list[int],
        activate_last: bool = True,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        """
        :param in_feats: Dimension of input features.
        :param dims: Architecture of neural networks.
        :param activate_last: Whether applying activation to last layer or not.
        :param bias_last: Whether applying bias to last layer or not.
        """
        super().__init__()
        device = torch.device(device)
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.layers = nn.Sequential()
        self.gates = nn.Sequential()
        self.use_bias = use_bias
        self.activate_last = activate_last
        for i, (in_dim, out_dim) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            if i < self._depth - 1:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias, device=device))
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias, device=device))
                self.layers.append(nn.SiLU())
                self.gates.append(nn.SiLU())
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias, device=device))
                if self.activate_last:
                    self.layers.append(nn.SiLU())
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias, device=device))
                self.gates.append(nn.Sigmoid())

    def forward(self, inputs: torch.tensor):
        return self.layers(inputs) * self.gates(inputs)


class EdgeSet2Set(Module):
    """
    Implementation of Set2Set.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """
        :param input_dim: The size of each input sample.
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

    def forward(self, g: DGLGraph, feat: torch.tensor):
        """
        Defines the computation performed at every call.

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
