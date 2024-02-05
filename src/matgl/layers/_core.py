"""Implementations of multi-layer perceptron (MLP) and other helper classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from dgl import DGLGraph, broadcast_edges, softmax_edges, sum_edges
from torch import Tensor, nn
from torch.nn import LSTM, Linear, Module, ModuleList

if TYPE_CHECKING:
    from collections.abc import Sequence


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: Callable[[Tensor], Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        """:param dims: Dimensions of each layer of MLP.
        :param activation: Activation function.
        :param activate_last: Whether to apply activation to last layer.
        :param bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)  # type: ignore
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

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

    def forward(self, inputs):
        """Applies all layers in turn.

        :param inputs: Input tensor
        :return: Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x


class GatedMLP(nn.Module):
    """An implementation of a Gated multi-layer perceptron."""

    def __init__(self, in_feats: int, dims: Sequence[int], activate_last: bool = True, use_bias: bool = True):
        """:param in_feats: Dimension of input features.
        :param dims: Architecture of neural networks.
        :param activate_last: Whether applying activation to last layer or not.
        :param bias_last: Whether applying bias to last layer or not.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.layers = nn.Sequential()
        self.gates = nn.Sequential()
        self.use_bias = use_bias
        self.activate_last = activate_last
        for i, (in_dim, out_dim) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            if i < self._depth - 1:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.layers.append(nn.SiLU())
                self.gates.append(nn.SiLU())
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                if self.activate_last:
                    self.layers.append(nn.SiLU())
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Sigmoid())

    def forward(self, inputs: Tensor):
        return self.layers(inputs) * self.gates(inputs)


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

    def forward(self, g: DGLGraph, feat: Tensor):
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


class GatedEquivariantBlock(nn.Module):
    """
    Gated equivariant block as used for the prediction of tensorial properties by PaiNN.
    Transforms scalar and vector representation using gated nonlinearities.

    References:
    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021 (to appear)

    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation: nn.Module,
        sactivation: nn.Module,
    ):
        """
        Args:
            n_sin: number of input scalar features.
            n_vin: number of input vector features.
            n_sout: number of output scalar features.
            n_vout: number of output vector features.
            n_hidden: number of hidden units.
            activation: internal activation function.
            sactivation: activation function for scalar outputs.
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = MLP(
            dims=[n_vin, 2 * n_vout],
            activation=None,
            activate_last=False,  # No activation in the last layer
            bias_last=True,
        )
        self.scalar_net = MLP(
            dims=[n_sin + n_vout, n_hidden, n_sout + n_vout], activation=activation, activate_last=False, bias_last=True
        )
        self.sactivation = sactivation

    def forward(self, inputs: tuple[Tensor, Tensor]):
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


def build_gated_equivariant_mlp(
    n_in: int,
    n_out: int,
    activation: nn.Module,
    sactivation: nn.Module,
    n_hidden: int | Sequence[int] | None = None,
    n_gating_hidden: int | Sequence[int] | None = None,
    n_layers: int = 2,
):
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s instead of dense layers.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_gating_hidden: number hidden gated layer nodes
            If an integer, same number of node is used for all hidden gated layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: Activation function for gating function.
        sactivation: Activation function for scalar outputs. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for _ in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        n_hidden = [n_hidden] * (n_layers - 1) if isinstance(n_hidden, int) else list(n_hidden)
        #        if type(n_hidden) is int:
        #            n_hidden = [n_hidden] * (n_layers - 1)
        #        else:
        #            n_hidden = list(n_hidden)
        # n_neurons = [n_in] + n_hidden + [n_out]
        n_neurons = [n_in, *n_hidden, n_out]
    if n_gating_hidden is None:
        n_gating_hidden = n_neurons[:-1]
    elif isinstance(n_gating_hidden, int):
        n_gating_hidden = [n_gating_hidden] * n_layers
    else:
        n_gating_hidden = list(n_gating_hidden)
    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        GatedEquivariantBlock(
            n_sin=n_neurons[i],
            n_vin=n_neurons[i],
            n_sout=n_neurons[i + 1],
            n_vout=n_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        GatedEquivariantBlock(
            n_sin=n_neurons[-2],
            n_vin=n_neurons[-2],
            n_sout=n_neurons[-1],
            n_vout=n_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net
