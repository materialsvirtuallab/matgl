from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from matgl.layers._core import MLP, GatedMLP
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import (
    decompose_tensor,
    new_radial_tensor,
    scatter_add,
    tensor_norm,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data


class TensorNetInteraction(MessagePassing):
    """A Graph Convolution block for TensorNet adapted for PyTorch Geometric."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            num_rbf: Number of radial basis functions.
            units: Number of hidden neurons.
            activation: Activation function.
            cutoff: Cutoff radius for graph construction.
            equivariance_invariance_group: Group action on geometric tensor representations, either O(3) or SO(3).
            dtype: Data type for all variables.
        """
        super().__init__(aggr="add")  # Aggregate messages by summation
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff
        self.equivariance_invariance_group = equivariance_invariance_group

        # Scalar linear layers
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(num_rbf, units, bias=True, dtype=dtype),
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )

        # Tensor linear layers (6 layers for scalar, skew, and traceless components)
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(6)])

        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the parameters."""
        for linear in self.linears_scalar:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.linears_tensor:
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, graph: Data, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph: PyTorch Geometric Data object containing graph structure.
            X: Node tensor representations.

        Returns:
            X: Updated node tensor representations after message passing.
        """
        edge_index = graph.edge_index
        edge_weight = graph.bond_dist  # Assuming bond_dist is stored in graph
        edge_attr = graph.edge_attr  # Assuming edge_attr is stored in graph

        # Process edge attributes
        C = cosine_cutoff(edge_weight, self.cutoff)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(edge_attr.shape[0], self.units, 3)

        # Normalize input tensor
        X = X / (tensor_norm(X) + 1)[..., None, None]

        # Decompose input tensor
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        # Apply tensor linear transformations
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[1](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = scalars + skew_metrices + traceless_tensors

        # Message passing
        graph.x_I = scalars
        graph.x_A = skew_metrices
        graph.x_S = traceless_tensors
        graph.edge_attr_processed = edge_attr

        messages = self.message(edge_index, graph.x_I, graph.x_A, graph.x_S, graph.edge_attr_processed)
        Im, Am, Sm = self.aggregate(messages, edge_index[0], X.size(0))
        # Combine messages
        msg = Im + Am + Sm

        # Apply group action
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            scalars, skew_metrices, traceless_tensors = decompose_tensor(A + B)
        elif self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            scalars, skew_metrices, traceless_tensors = decompose_tensor(2 * B)
        else:
            raise ValueError("equivariance_invariance_group must be 'O(3)' or 'SO(3)'")

        # Normalize and apply final tensor transformations
        normp1 = (tensor_norm(scalars + skew_metrices + traceless_tensors) + 1)[..., None, None]
        scalars = scalars / normp1
        skew_metrices = skew_metrices / normp1
        traceless_tensors = traceless_tensors / normp1

        scalars = self.linears_tensor[3](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[4](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[5](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Compute update
        dX = scalars + skew_metrices + traceless_tensors
        X = X + dX + torch.matmul(dX, dX)

        return X

    def message(self, edge_index, x_I: torch.Tensor, x_A: torch.Tensor, x_S: torch.Tensor, edge_attr: torch.Tensor):
        """Compute messages for each edge."""
        _, dst = edge_index
        x_I_j = x_I[dst]
        x_A_j = x_A[dst]
        x_S_j = x_S[dst]
        scalars, skew_metrices, traceless_tensors = new_radial_tensor(
            x_I_j, x_A_j, x_S_j, edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2]
        )
        return scalars, skew_metrices, traceless_tensors

    def aggregate(self, inputs, index, dim_size):
        """Aggregate messages for node updates."""
        scalars, skew_matrices, traceless_tensors = inputs
        scalars_agg = scatter_add(scalars, index, dim_size=dim_size)
        skew_matrices_agg = scatter_add(skew_matrices, index, dim_size=dim_size)
        traceless_tensors_agg = scatter_add(traceless_tensors, index, dim_size=dim_size)
        return scalars_agg, skew_matrices_agg, traceless_tensors_agg


class MEGNetGraphConv(MessagePassing):
    """A MEGNet graph convolution layer in PyG."""

    def __init__(
        self,
        edge_func: nn.Module,
        node_func: nn.Module,
        state_func: nn.Module,
    ) -> None:
        """
        Args:
            edge_func: Edge update function.
            node_func: Node update function.
            state_func: Global state update function.
        """
        super().__init__(aggr="mean")  # Aggregate messages by mean
        self.edge_func = edge_func
        self.node_func = node_func
        self.state_func = state_func

    @staticmethod
    def from_dims(
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int],
        activation: nn.Module,
    ) -> MEGNetGraphConv:
        """Create a MEGNet graph convolution layer from dimensions.

        Args:
            edge_dims (list[int]): Edge dimensions.
            node_dims (list[int]): Node dimensions.
            state_dims (list[int]): State dimensions.
            activation (nn.Module): Activation function.

        Returns:
            MEGNetGraphConv: MEGNet graph convolution layer.
        """
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        state_update = MLP(state_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, state_update)

    def forward(
        self,
        graph: Data,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform sequence of edge->node->attribute updates.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state)

        Returns:
            (edge features, node features, graph attributes)
        """
        edge_index = graph.edge_index
        # Ensure edge_index is long dtype for indexing and scatter operations
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)

        # Broadcast state features to nodes
        if hasattr(graph, "batch") and graph.batch is not None:
            # For batched graphs, broadcast state_feat to each node based on batch
            # Ensure batch is long dtype
            batch = graph.batch.to(torch.long)
            num_graphs = batch.max().item() + 1
            if state_feat.dim() == 1:
                state_feat = state_feat.unsqueeze(0)
            if state_feat.size(0) == 1:
                state_feat = state_feat.expand(num_graphs, -1)
            node_state = state_feat[batch]
        else:
            # Single graph case
            if state_feat.dim() == 1:
                state_feat = state_feat.unsqueeze(0)
            node_state = state_feat.expand(node_feat.size(0), -1)

        # Edge update
        vi = node_feat[src]
        vj = node_feat[dst]
        u_edge = node_state[src]  # State for edge context
        edge_inputs = torch.hstack([vi, vj, edge_feat, u_edge])
        edge_update = self.edge_func(edge_inputs)

        # Node update - aggregate edge messages
        edge_messages = scatter(edge_update, dst, dim=0, dim_size=node_feat.size(0), reduce="mean")
        ve = edge_messages
        node_inputs = torch.hstack([node_feat, ve, node_state])
        node_update = self.node_func(node_inputs)

        # State update
        if hasattr(graph, "batch") and graph.batch is not None:
            batch = graph.batch.to(torch.long)
            num_graphs = batch.max().item() + 1
            u_edge_mean = scatter(edge_update, batch[src], dim=0, dim_size=num_graphs, reduce="mean")
            u_vertex_mean = scatter(node_update, batch, dim=0, dim_size=num_graphs, reduce="mean")
            if state_feat.size(0) == 1 and num_graphs > 1:
                state_feat_expanded = state_feat.expand(num_graphs, -1)
                state_inputs = torch.hstack([state_feat_expanded, u_edge_mean, u_vertex_mean])
            elif state_feat.size(0) == 1:
                state_inputs = torch.hstack([state_feat.squeeze(0), u_edge_mean.squeeze(0), u_vertex_mean.squeeze(0)])
            else:
                state_inputs = torch.hstack([state_feat, u_edge_mean, u_vertex_mean])
        else:
            u_edge_mean = edge_update.mean(dim=0, keepdim=True).squeeze(0)
            u_vertex_mean = node_update.mean(dim=0, keepdim=True).squeeze(0)
            if state_feat.dim() == 1:
                state_inputs = torch.hstack([state_feat, u_edge_mean, u_vertex_mean])
            else:
                state_inputs = torch.hstack([state_feat.squeeze(0), u_edge_mean, u_vertex_mean])
        state_update = self.state_func(state_inputs)

        return edge_update, node_update, state_update


class MEGNetBlock(nn.Module):
    """A MEGNet block comprising a sequence of update operations."""

    def __init__(
        self, dims: list[int], conv_hiddens: list[int], act: nn.Module, dropout: float | None = None, skip: bool = True
    ) -> None:
        """
        Init the MEGNet block with key parameters.

        Args:
            dims: Dimension of dense layers before graph convolution.
            conv_hiddens: Architecture of hidden layers of graph convolution.
            act: Activation type.
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according
                to a Bernoulli distribution.
            skip: Residual block.
        """
        super().__init__()
        self.has_dense = len(dims) > 1
        self.activation = act
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]

        mlp_kwargs = {
            "dims": dims,
            "activation": self.activation,
            "activate_last": True,
            "bias_last": True,
        }
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else nn.Identity()  # type: ignore
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else nn.Identity()  # type: ignore
        self.state_func = MLP(**mlp_kwargs) if self.has_dense else nn.Identity()  # type: ignore

        # compute input sizes
        edge_in = 2 * conv_dim + conv_dim + conv_dim  # 2*NDIM+EDIM+GDIM
        node_in = out_dim + conv_dim + conv_dim  # EDIM+NDIM+GDIM
        state_in = out_dim + out_dim + conv_dim  # EDIM+NDIM+GDIM
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in, *conv_hiddens],
            node_dims=[node_in, *conv_hiddens],
            state_dims=[state_in, *conv_hiddens],
            activation=self.activation,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None
        self.skip = skip

    def forward(
        self,
        graph: Data,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MEGNetBlock forward pass.

        Args:
            graph (Data): A PyG Data object.
            edge_feat (Tensor): Edge features.
            node_feat (Tensor): Node features.
            state_feat (Tensor): Graph attributes (global state).

        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated (edge features,
                node features, graph attributes)
        """
        inputs = (edge_feat, node_feat, state_feat)
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_feat = self.state_func(state_feat)

        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat, node_feat, state_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            state_feat = self.dropout(state_feat)  # pylint: disable=E1102

        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_feat = state_feat + inputs[2]

        return edge_feat, node_feat, state_feat


class M3GNetGraphConv(MessagePassing):
    """A M3GNet graph convolution layer in PyG."""

    def __init__(
        self,
        include_state: bool,
        edge_update_func: nn.Module,
        edge_weight_func: nn.Module,
        node_update_func: nn.Module,
        node_weight_func: nn.Module,
        state_update_func: nn.Module | None,
    ):
        """Parameters:
        include_state (bool): Whether including state
        edge_update_func (nn.Module): Update function for edges (Eq. 4)
        edge_weight_func (nn.Module): Weight function for radial basis functions (Eq. 4)
        node_update_func (nn.Module): Update function for nodes (Eq. 5)
        node_weight_func (nn.Module): Weight function for radial basis functions (Eq. 5)
        state_update_func (nn.Module): Update function for state feats (Eq. 6).
        """
        super().__init__(aggr="sum")  # Aggregate messages by summation
        self.include_state = include_state
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree,
        include_state,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: nn.Module,
    ) -> M3GNetGraphConv:
        """M3GNetGraphConv initialization.

        Args:
            degree (int): max_n*max_l
            include_state (bool): whether including state or not
            edge_dims (list): NN architecture for edge update function
            node_dims (list): NN architecture for node update function
            state_dims (list): NN architecture for state update function
            activation (nn.Module): activation function

        Returns:
        M3GNetGraphConv (class)
        """
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        state_update_func = MLP(state_dims, activation, activate_last=True) if include_state else None  # type: ignore
        return M3GNetGraphConv(
            include_state, edge_update_func, edge_weight_func, node_update_func, node_weight_func, state_update_func
        )

    def forward(
        self,
        graph: Data,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform sequence of edge->node->states updates.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state).

        Returns:
            (edge features, node features, graph attributes)
        """
        edge_index = graph.edge_index
        # Ensure edge_index is long dtype for indexing and scatter operations
        src = edge_index[0].to(torch.long)
        dst = edge_index[1].to(torch.long)
        rbf = graph.rbf  # Radial basis functions stored in graph

        # Broadcast state features to nodes
        if self.include_state:
            if hasattr(graph, "batch") and graph.batch is not None:
                # Ensure batch is long dtype
                batch = graph.batch.to(torch.long)
                num_graphs = batch.max().item() + 1
                if state_feat.dim() == 1:
                    state_feat = state_feat.unsqueeze(0)
                if state_feat.size(0) == 1:
                    state_feat = state_feat.expand(num_graphs, -1)
                node_state = state_feat[batch]
            else:
                if state_feat.dim() == 1:
                    state_feat = state_feat.unsqueeze(0)
                node_state = state_feat.expand(node_feat.size(0), -1)
        else:
            node_state = None

        # Edge update
        vi = node_feat[src]
        vj = node_feat[dst]
        if self.include_state:
            assert node_state is not None  # Type narrowing for type checker
            u = node_state[src]
            edge_inputs = torch.hstack([vi, vj, edge_feat, u])
        else:
            edge_inputs = torch.hstack([vi, vj, edge_feat])
        edge_update = self.edge_update_func(edge_inputs) * self.edge_weight_func(rbf)

        # Node update
        if self.include_state:
            assert node_state is not None  # Type narrowing for type checker
            u = node_state[dst]
            node_inputs = torch.hstack([vi, vj, edge_feat, u])
        else:
            node_inputs = torch.hstack([vi, vj, edge_feat])
        messages = self.node_update_func(node_inputs) * self.node_weight_func(rbf)

        # Aggregate messages
        node_update = scatter(messages, dst, dim=0, dim_size=node_feat.size(0), reduce="sum")

        # State update
        if self.include_state and self.state_update_func is not None:
            if hasattr(graph, "batch") and graph.batch is not None:
                batch = graph.batch.to(torch.long)
                num_graphs = batch.max().item() + 1
                uv = scatter(node_feat, batch, dim=0, dim_size=num_graphs, reduce="mean")
                if state_feat.size(0) == 1:
                    state_inputs = torch.hstack([state_feat.squeeze(0), uv.squeeze(0)])
                else:
                    state_inputs = torch.hstack([state_feat, uv])
            else:
                uv = node_feat.mean(dim=0, keepdim=True)
                state_inputs = torch.hstack([state_feat.squeeze(0), uv.squeeze(0)])
            state_update = self.state_update_func(state_inputs)
        else:
            state_update = state_feat

        return edge_feat + edge_update, node_feat + node_update, state_update


class M3GNetBlock(nn.Module):
    """A M3GNet block comprising a sequence of update operations."""

    def __init__(
        self,
        degree: int,
        activation: nn.Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        dropout: float | None = None,
    ) -> None:
        """

        Args:
            degree: Number of radial basis functions
            activation: activation
            dim_node_feats: Number of node features
            dim_edge_feats: Number of edge features
            dim_state_feats: Number of state features
            conv_hiddens: Dimension of hidden layers
            activation: Activation type
            include_state: Including state features or not
            dropout: Probability of an element to be zero in dropout layer.
        """
        super().__init__()

        self.activation = activation

        # compute input sizes
        if include_state:
            edge_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats  # type: ignore
            node_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats  # type: ignore
            state_in = dim_node_feats + dim_state_feats  # type: ignore
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=[state_in, *conv_hiddens, dim_state_feats],  # type: ignore
                activation=self.activation,
            )
        else:
            edge_in = 2 * dim_node_feats + dim_edge_feats  # 2*NDIM+EDIM
            node_in = 2 * dim_node_feats + dim_edge_feats  # 2*NDIM+EDIM
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=None,  # type: ignore
                activation=self.activation,
            )

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(
        self,
        graph: Data,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ) -> tuple:
        """
        Args:
            graph: PyG graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features.

        Returns:
            A tuple of updated features
        """
        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat, node_feat, state_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            if state_feat is not None:
                state_feat = self.dropout(state_feat)  # pylint: disable=E1102

        return edge_feat, node_feat, state_feat
