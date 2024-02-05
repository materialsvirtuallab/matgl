"""Graph convolution layer (GCL) implementations."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Identity, Module

import matgl
from matgl.layers._core import MLP, GatedMLP
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import decompose_tensor, new_radial_tensor, tensor_norm


class MEGNetGraphConv(Module):
    """A MEGNet graph convolution layer in DGL."""

    def __init__(
        self,
        edge_func: Module,
        node_func: Module,
        state_func: Module,
    ) -> None:
        """
        Args:
            edge_func: Edge update function.
            node_func: Node update function.
            state_func: Global state update function.
        """
        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.state_func = state_func

    @staticmethod
    def from_dims(
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int],
        activation: Module,
    ) -> MEGNetGraphConv:
        """Create a MEGNet graph convolution layer from dimensions.

        Args:
            edge_dims (list[int]): Edge dimensions.
            node_dims (list[int]): Node dimensions.
            state_dims (list[int]): State dimensions.
            activation (Module): Activation function.

        Returns:
            MEGNetGraphConv: MEGNet graph convolution layer.
        """
        # TODO(marcel): Softplus doesn't exactly match paper's SoftPlus2
        # TODO(marcel): Should we activate last?
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        attr_update = MLP(state_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, attr_update)

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = edges.src["u"]
        eij = edges.data.pop("e")
        inputs = torch.hstack([vi, vj, eij, u])
        mij = {"mij": self.edge_func(inputs)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update.

        Args:
            graph: Input graph

        Returns:
            Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        graph.edata["e"] = graph.edata.pop("mij")
        return graph.edata["e"]

    def node_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform node update.

        Args:
            graph: Input graph

        Returns:
            Output tensor for nodes.
        """
        graph.update_all(fn.copy_e("e", "e"), fn.mean("e", "ve"))
        ve = graph.ndata.pop("ve")
        v = graph.ndata.pop("v")
        u = graph.ndata.pop("u")
        inputs = torch.hstack([v, ve, u])
        graph.ndata["v"] = self.node_func(inputs)
        return graph.ndata["v"]

    def state_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: Input graph
            state_feat: Input attributes

        Returns:
            Output tensor for attributes
        """
        u_edge = dgl.readout_edges(graph, feat="e", op="mean")
        u_vertex = dgl.readout_nodes(graph, feat="v", op="mean")
        u_edge = torch.squeeze(u_edge)
        u_vertex = torch.squeeze(u_vertex)
        inputs = torch.hstack([state_feat.squeeze(), u_edge, u_vertex])
        state_feat = self.state_func(inputs)
        return state_feat

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->attribute updates.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state)

        Returns:
            (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            graph.ndata["u"] = dgl.broadcast_nodes(graph, state_feat)

            edge_feat = self.edge_update_(graph)
            node_feat = self.node_update_(graph)
            state_feat = self.state_update_(graph, state_feat)

        return edge_feat, node_feat, state_feat


class MEGNetBlock(Module):
    """A MEGNet block comprising a sequence of update operations."""

    def __init__(
        self, dims: list[int], conv_hiddens: list[int], act: Module, dropout: float | None = None, skip: bool = True
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
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore
        self.state_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore

        # compute input sizes
        edge_in = 2 * conv_dim + conv_dim + conv_dim  # 2*NDIM+EDIM+GDIM
        node_in = out_dim + conv_dim + conv_dim  # EDIM+NDIM+GDIM
        attr_in = out_dim + out_dim + conv_dim  # EDIM+NDIM+GDIM
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in, *conv_hiddens],
            node_dims=[node_in, *conv_hiddens],
            state_dims=[attr_in, *conv_hiddens],
            activation=self.activation,
        )

        self.dropout = Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout
        self.skip = skip

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """MEGNetBlock forward pass.

        Args:
            graph (dgl.DGLGraph): A DGLGraph.
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


class M3GNetGraphConv(Module):
    """A M3GNet graph convolution layer in DGL."""

    def __init__(
        self,
        include_states: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ):
        """Parameters:
        include_states (bool): Whether including state
        edge_update_func (Module): Update function for edges (Eq. 4)
        edge_weight_func (Module): Weight function for radial basis functions (Eq. 4)
        node_update_func (Module): Update function for nodes (Eq. 5)
        node_weight_func (Module): Weight function for radial basis functions (Eq. 5)
        state_update_func (Module): Update function for state feats (Eq. 6).
        """
        super().__init__()
        self.include_states = include_states
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree,
        include_states,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: Module,
    ) -> M3GNetGraphConv:
        """M3GNetGraphConv initialization.

        Args:
            degree (int): max_n*max_l
            include_states (bool): whether including state or not
            edge_dims (list): NN architecture for edge update function
            node_dims (list): NN architecture for node update function
            state_dims (list): NN architecture for state update function
            activation (nn.Nodule): activation function

        Returns:
        M3GNetGraphConv (class)
        """
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        attr_update_func = MLP(state_dims, activation, activate_last=True) if include_states else None  # type: ignore
        return M3GNetGraphConv(
            include_states, edge_update_func, edge_weight_func, node_update_func, node_weight_func, attr_update_func
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        """Edge update functions.

        Args:
        edges (DGL graph): edges in dgl graph

        Returns:
        mij: message passing between node i and j
        """
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = None
        if self.include_states:
            u = edges.src["u"]
        eij = edges.data.pop("e")
        rbf = edges.data["rbf"]
        inputs = torch.hstack([vi, vj, eij, u]) if self.include_states else torch.hstack([vi, vj, eij])
        mij = {"mij": self.edge_update_func(inputs) * self.edge_weight_func(rbf)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update.

        Args:
        graph: DGL graph

        Returns:
        edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata.pop("mij")
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform node update.

        Args:
            graph: DGL graph
            state_feat: State attributes

        Returns:
            node_update: node features update
        """
        eij = graph.edata["e"]
        src_id = graph.edges()[0]
        vi = graph.ndata["v"][src_id]
        dst_id = graph.edges()[1]
        vj = graph.ndata["v"][dst_id]
        rbf = graph.edata["rbf"]
        if self.include_states:
            u = dgl.broadcast_edges(graph, state_feat)
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        graph.edata["mess"] = self.node_update_func(inputs) * self.node_weight_func(rbf)
        graph.update_all(fn.copy_e("mess", "mess"), fn.sum("mess", "ve"))
        node_update = graph.ndata.pop("ve")
        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: DGL graph
            state_feat: graph features

        Returns:
        state_update: state_features update
        """
        u = state_feat
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, uv])
        state_feat = self.state_update_func(inputs)  # type: ignore
        return state_feat

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            if self.include_states:
                graph.ndata["u"] = dgl.broadcast_nodes(graph, state_feat)

            edge_update = self.edge_update_(graph)
            graph.edata["e"] = edge_feat + edge_update
            node_update = self.node_update_(graph, state_feat)
            graph.ndata["v"] = node_feat + node_update
            if self.include_states:
                state_feat = self.state_update_(graph, state_feat)

        return edge_feat + edge_update, node_feat + node_update, state_feat


class M3GNetBlock(Module):
    """A M3GNet block comprising a sequence of update operations."""

    def __init__(
        self,
        degree: int,
        activation: Module,
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
            attr_in = dim_node_feats + dim_state_feats  # type: ignore
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=[attr_in, *conv_hiddens, dim_state_feats],  # type: ignore
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

        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple:
        """
        Args:
            graph: DGL graph
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


class TensorNetInteraction(nn.Module):
    """A Graph Convolution block for TensorNet. The official implementation can be found in https://github.com/torchmd/torchmd-net."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = matgl.float_th,
    ):
        """

        Args:
            num_rbf: Number of radial basis functions.
            units: number of hidden neurons.
            activation: activation.
            cutoff: cutoff radius for graph construction.
            equivariance_invariance_group: Group action on geometric tensor representations, either O(3) or SO(3).
            dtype: data type for all variables.
        """
        super().__init__()

        self.num_rbf = num_rbf
        self.units = units
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(num_rbf, units, bias=True, dtype=dtype))
        self.linears_scalar.append(nn.Linear(units, 2 * units, bias=True, dtype=dtype))
        self.linears_scalar.append(nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype))
        self.linears_tensor = nn.ModuleList(nn.Linear(units, units, bias=False) for _ in range(6))
        #        self.act = activation()
        self.act = activation
        self.equivariance_invariance_group = equivariance_invariance_group
        self.reset_parameters()
        self.cutoff = cutoff

    def reset_parameters(self):
        """Reinitialize the parameters."""
        for linear in self.linears_scalar:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        """Edge update functions.

        Args:
        edges (DGL graph): edges in dgl graph

        Returns:
        mij: message passing between node i and j
        """
        I_j = edges.dst["I"]
        A_j = edges.dst["A"]
        S_j = edges.dst["S"]
        edge_attr = edges.data["e"]
        scalars, skew_metrices, traceless_tensors = new_radial_tensor(
            I_j, A_j, S_j, edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2]
        )
        mij = {"I": scalars, "A": skew_metrices, "S": traceless_tensors}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """Perform edge update.

        Args:
        graph: DGL graph

        Returns:
        edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        return graph

    def node_update_(self, graph: dgl.DGLGraph) -> tuple[Tensor, Tensor, Tensor]:
        """Perform node update.

        Args:
            graph: DGL graph

        Returns:
            node_update: node features update
        """
        graph.update_all(fn.copy_e("I", "I"), fn.sum("I", "Ie"))
        graph.update_all(fn.copy_e("A", "A"), fn.sum("A", "Ae"))
        graph.update_all(fn.copy_e("S", "S"), fn.sum("S", "Se"))
        scalars = graph.ndata.pop("Ie")
        skew_metrices = graph.ndata.pop("Ae")
        traceless_tensors = graph.ndata.pop("Se")
        return scalars, skew_metrices, traceless_tensors

    def forward(self, g: dgl.DGLGraph, X: Tensor):
        """

        Args:
            g: dgl graph.
            X: node tensor representations.

        Returns:
            X: message passed tensor representations.
        """
        edge_weight = g.edata["bond_dist"]
        edge_attr = g.edata["edge_attr"]
        C = cosine_cutoff(edge_weight, self.cutoff)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(edge_attr.shape[0], self.units, 3)
        X = X / (tensor_norm(X) + 1)[..., None, None]
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[1](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = scalars + skew_metrices + traceless_tensors
        # propagate_type: (I: Tensor, A: Tensor, S: Tensor, edge_attr: Tensor)
        with g.local_scope():
            g.ndata["I"] = scalars
            g.ndata["A"] = skew_metrices
            g.ndata["S"] = traceless_tensors
            g.edata["e"] = edge_attr
            g = self.edge_update_(g)
            Im, Am, Sm = self.node_update_(g)
            msg = Im + Am + Sm
            if self.equivariance_invariance_group == "O(3)":
                A = torch.matmul(msg, Y)
                B = torch.matmul(Y, msg)
                scalars, skew_metrices, traceless_tensors = decompose_tensor(A + B)
            if self.equivariance_invariance_group == "SO(3)":
                B = torch.matmul(Y, msg)
                scalars, skew_metrices, traceless_tensors = decompose_tensor(2 * B)
            normp1 = (tensor_norm(scalars + skew_metrices + traceless_tensors) + 1)[..., None, None]
            scalars, skew_metrices, traceless_tensors = (
                scalars / normp1,
                skew_metrices / normp1,
                traceless_tensors / normp1,
            )
            scalars = self.linears_tensor[3](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            skew_metrices = self.linears_tensor[4](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            traceless_tensors = self.linears_tensor[5](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            dX = scalars + skew_metrices + traceless_tensors
            X = X + dX + torch.matmul(dX, dX)
        return X
