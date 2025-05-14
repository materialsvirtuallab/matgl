"""Implementations of multi-layer perceptron (MLP) and other helper classes without DGL."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import Identity, Linear, Module, ModuleList, Sigmoid
from torch_geometric.nn import Set2Set

from matgl.layers._norm import GraphNorm, LayerNorm

if TYPE_CHECKING:
    from torch import LongTensor


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: Callable[[Tensor], Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            last = i == self._depth - 1
            self.layers.append(Linear(in_dim, out_dim, bias=(bias_last if last else True)))
            if activation is not None and (activate_last or not last):
                self.layers.append(activation)  # type: ignore

    def __repr__(self):
        dims = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} → {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)
        return f"MLP({', '.join(dims)})"

    @property
    def last_linear(self) -> Linear:
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        raise RuntimeError("No linear layer found")

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features  # type: ignore

    @property
    def out_features(self) -> int:
        return self.last_linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MLP_norm(nn.Module):
    """Multi-layer perceptron with optional graph- or layer-normalization."""

    def __init__(
        self,
        dims: list[int],
        activation: nn.Module | None = None,
        activate_last: bool = False,
        use_bias: bool = True,
        bias_last: bool = True,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        norm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._depth = len(dims) - 1
        self.activation = activation or Identity()
        self.activate_last = activate_last
        self.normalize_hidden = normalize_hidden

        norm_kwargs = norm_kwargs or {}
        has_norm = normalization in ("graph", "layer")

        self.layers = ModuleList()
        self.norms: ModuleList | None = ModuleList() if has_norm else None

        for i, (d_in, d_out) in enumerate(itertools.pairwise(dims)):
            last = i == self._depth - 1
            self.layers.append(Linear(d_in, d_out, bias=(use_bias and (bias_last if last else True))))
            if has_norm:
                Norm = GraphNorm if normalization == "graph" else LayerNorm
                self.norms.append(Norm(d_out, **norm_kwargs))
        self._normalization = normalization

    def forward(self, x: Tensor, batch: LongTensor | None = None) -> Tensor:
        for i in range(self._depth):
            x = self.layers[i](x)
            if self.norms is not None:
                # only normalize hidden if requested, but always normalize final if a norm is present
                if i < self._depth - 1 and not self.normalize_hidden:
                    pass
                else:
                    # GraphNorm expects (x, batch), LayerNorm ignores batch
                    x = self.norms[i](x, batch)  # type: ignore
            if i < self._depth - 1 or self.activate_last:
                x = self.activation(x)
        return x


class GatedMLP(nn.Module):
    """An implementation of a Gated multi-layer perceptron."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activate_last: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.activate_last = activate_last

        # build two parallel MLPs: one for values, one for gates
        self.values = ModuleList()
        self.gates = ModuleList()

        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:], strict=False)):
            last = i == self._depth - 1
            # value path
            self.values.append(Linear(d_in, d_out, bias=use_bias))
            # gate path
            self.gates.append(Linear(d_in, d_out, bias=use_bias))
            # activations
            if not last or activate_last:
                self.values.append(nn.SiLU())
            self.gates.append(nn.Sigmoid() if last else nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        v = x
        g = x
        for layer in self.values:
            v = layer(v)
        for layer in self.gates:
            g = layer(g)
        return v * g


class GatedMLP_norm(nn.Module):
    """Gated MLP with normalization, built on top of MLP_norm."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activation: nn.Module | None = None,
        activate_last: bool = True,
        use_bias: bool = True,
        bias_last: bool = True,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        norm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.layers = MLP_norm(
            [in_feats, *dims],
            activation=activation or nn.SiLU(),
            activate_last=True,
            use_bias=use_bias,
            bias_last=bias_last,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        self.gates = MLP_norm(
            [in_feats, *dims],
            activation=activation or nn.SiLU(),
            activate_last=False,
            use_bias=use_bias,
            bias_last=bias_last,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, batch: LongTensor | None = None) -> Tensor:
        return self.layers(x, batch) * self.sigmoid(self.gates(x, batch))


class EdgeSet2Set(Module):
    """Edge-focused Set2Set readout using PyG’s Set2Set underneath."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        super().__init__()
        # PyG's Set2Set: (in_channels, processing_steps, num_layers)
        self.pool = Set2Set(input_dim, processing_steps=n_iters, num_layers=n_layers)
        self.output_dim = 2 * input_dim

    def forward(self, x: Tensor, batch: LongTensor) -> Tensor:
        """
        :param x: edge features, shape [num_edges, input_dim]
        :param batch: edge-to-graph mapping, shape [num_edges]
        """
        return self.pool(x, batch)


class GatedEquivariantBlock(nn.Module):
    """
    Gated equivariant block for tensorial prediction (same as before).
    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation: nn.Module,
        sactivation: nn.Module | None = None,
    ):
        super().__init__()
        self.mix_vectors = MLP(
            dims=[n_vin, 2 * n_vout],
            activation=None,
            activate_last=False,
            bias_last=False,
        )
        self.scalar_net = MLP(
            dims=[n_sin + n_vout, n_hidden, n_sout + n_vout],
            activation=activation,
            activate_last=False,
            bias_last=True,
        )
        self.sactivation = sactivation

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        vV, vW = torch.split(vmix, vmix.shape[-1] // 2, dim=-1)
        vVn = torch.norm(vV, dim=-2)
        ctx = torch.cat([scalars, vVn], dim=-1)
        out = self.scalar_net(ctx)
        s_out, v_part = torch.split(
            out, [self.scalar_net.last_linear.out_features - vW.shape[-1], vW.shape[-1]], dim=-1
        )
        v_out = v_part.unsqueeze(-2) * vW
        if self.sactivation:
            s_out = self.sactivation(s_out)
        return s_out, v_out


def build_gated_equivariant_mlp(
    n_in: int,
    n_out: int,
    activation: nn.Module,
    sactivation: nn.Module | None = None,
    n_hidden: int | Sequence[int] | None = None,
    n_gating_hidden: int | Sequence[int] | None = None,
    n_layers: int = 2,
) -> nn.Sequential:
    # Determine hidden dims
    if n_hidden is None:
        dims = []
        size = n_in
        for _ in range(n_layers - 1):
            dims.append(size)
            size = max(n_out, size // 2)
        dims = [n_in, *dims, n_out]
    else:
        hidden = [n_hidden] * (n_layers - 1) if isinstance(n_hidden, int) else list(n_hidden)
        dims = [n_in, *hidden, n_out]

    # Gating hidden dims
    if n_gating_hidden is None:
        gate_dims = dims[:-1]
    else:
        gate_dims = [n_gating_hidden] * n_layers if isinstance(n_gating_hidden, int) else list(n_gating_hidden)

    layers = []
    for i in range(n_layers):
        layers.append(
            GatedEquivariantBlock(
                n_sin=dims[i],
                n_vin=dims[i],
                n_sout=dims[i + 1],
                n_vout=dims[i + 1],
                n_hidden=gate_dims[i],
                activation=activation,
                sactivation=(sactivation if i < n_layers - 1 else None),
            )
        )
    return nn.Sequential(*layers)
