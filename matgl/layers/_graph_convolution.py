"""Graph convolution layer (GCL) implementations."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Identity, Module

from matgl.layers._core import MLP, GatedMLP


class MEGNetGraphConv(Module):
    """A MEGNet graph convolution layer in DGL."""

    def __init__(
        self,
        edge_func: Module,
        node_func: Module,
        state_func: Module,
    ) -> None:
        """:param edge_func: Edge update function.
        :param node_func: Node update function.
        :param state_func: Global state update function.
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

        :param graph: Input graph
        :return: Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        graph.edata["e"] = graph.edata.pop("mij")
        return graph.edata["e"]

    def node_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform node update.

        :param graph: Input graph
        :return: Output tensor for nodes.
        """
        graph.update_all(fn.copy_e("e", "e"), fn.mean("e", "ve"))
        ve = graph.ndata.pop("ve")
        v = graph.ndata.pop("v")
        u = graph.ndata.pop("u")
        inputs = torch.hstack([v, ve, u])
        graph.ndata["v"] = self.node_func(inputs)
        return graph.ndata["v"]

    def state_update_(self, graph: dgl.DGLGraph, state_attrs: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        :param graph: Input graph
        :param state_attrs: Input attributes
        :return: Output tensor for attributes
        """
        u_edge = dgl.readout_edges(graph, feat="e", op="mean")
        u_vertex = dgl.readout_nodes(graph, feat="v", op="mean")
        u_edge = torch.squeeze(u_edge)
        u_vertex = torch.squeeze(u_vertex)
        inputs = torch.hstack([state_attrs.squeeze(), u_edge, u_vertex])
        state_attr = self.state_func(inputs)
        return state_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_attr: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->attribute updates.

        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param state_attr: Graph attributes (global state)
        :return: (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            graph.ndata["u"] = dgl.broadcast_nodes(graph, state_attr)

            edge_feat = self.edge_update_(graph)
            node_feat = self.node_update_(graph)
            state_attr = self.state_update_(graph, state_attr)

        return edge_feat, node_feat, state_attr


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
        state_attr: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """MEGNetBlock forward pass.

        Args:
            graph (dgl.DGLGraph): A DGLGraph.
            edge_feat (Tensor): Edge features.
            node_feat (Tensor): Node features.
            state_attr (Tensor): Graph attributes (global state).

        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated (edge features,
                node features, graph attributes)
        """
        inputs = (edge_feat, node_feat, state_attr)
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_attr = self.state_func(state_attr)

        edge_feat, node_feat, state_attr = self.conv(graph, edge_feat, node_feat, state_attr)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            state_attr = self.dropout(state_attr)  # pylint: disable=E1102

        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_attr = state_attr + inputs[2]

        return edge_feat, node_feat, state_attr


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
        include_state (bool): Whether including state
        edge_update_func (Module): Update function for edges (Eq. 4)
        edge_weight_func (Module): Weight function for radial basis functions (Eq. 4)
        node_update_func (Module): Update function for nodes (Eq. 5)
        node_weight_func (Module): Weight function for radial basis functions (Eq. 5)
        attr_update_func (Module): Update function for state feats (Eq. 6).
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
        if self.include_states:
            u = edges.src["u"]
        eij = edges.data.pop("e")
        rbf = edges.data["rbf"]
        rbf = rbf.float()
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

    def node_update_(self, graph: dgl.DGLGraph, state_attr: Tensor) -> Tensor:
        """Perform node update.

        Args:
            graph: DGL graph
            state_attr: State attributes

        Returns:
            node_update: node features update
        """
        eij = graph.edata["e"]
        src_id = graph.edges()[0]
        vi = graph.ndata["v"][src_id]
        dst_id = graph.edges()[1]
        vj = graph.ndata["v"][dst_id]
        rbf = graph.edata["rbf"]
        rbf = rbf.float()
        if self.include_states:
            u = dgl.broadcast_edges(graph, state_attr)
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        graph.edata["mess"] = self.node_update_func(inputs) * self.node_weight_func(rbf)
        graph.update_all(fn.copy_e("mess", "mess"), fn.sum("mess", "ve"))
        node_update = graph.ndata.pop("ve")
        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_attrs: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: DGL graph
            state_attrs: graph features

        Returns:
        state_update: state_features update
        """
        u = state_attrs
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, uv])
        state_attr = self.state_update_func(inputs)  # type: ignore
        return state_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_attr: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param state_attr: Graph attributes (global state)
        :return: (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            if self.include_states:
                graph.ndata["u"] = dgl.broadcast_nodes(graph, state_attr)

            edge_update = self.edge_update_(graph)
            graph.edata["e"] = edge_feat + edge_update
            node_update = self.node_update_(graph, state_attr)
            graph.ndata["v"] = node_feat + node_update
            if self.include_states:
                state_attr = self.state_update_(graph, state_attr)

        return edge_feat + edge_update, node_feat + node_update, state_attr


class M3GNetBlock(Module):
    """A M3GNet block comprising a sequence of update operations."""

    def __init__(
        self,
        degree: int,
        activation: Module,
        conv_hiddens: list[int],
        num_node_feats: int,
        num_edge_feats: int,
        num_state_feats: int | None = None,
        include_state: bool = False,
        dropout: float | None = None,
    ) -> None:
        """:param degree: Dimension of radial basis functions
        :param num_node_feats: Number of node features
        :param num_edge_feats: Number of edge features
        :param num_state_feats: Number of state features
        :param conv_hiddens: Dimension of hidden layers
        :param activation: Activation type
        :param include_state: Including state features or not
        :param dropout: Probability of an element to be zero in dropout layer
        """
        super().__init__()

        self.activation = activation

        # compute input sizes
        if include_state:
            edge_in = 2 * num_node_feats + num_edge_feats + num_state_feats  # type: ignore
            node_in = 2 * num_node_feats + num_edge_feats + num_state_feats  # type: ignore
            attr_in = num_node_feats + num_state_feats  # type: ignore
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, num_edge_feats],
                node_dims=[node_in, *conv_hiddens, num_node_feats],
                state_dims=[attr_in, *conv_hiddens, num_state_feats],  # type: ignore
                activation=self.activation,
            )
        else:
            edge_in = 2 * num_node_feats + num_edge_feats  # 2*NDIM+EDIM
            node_in = 2 * num_node_feats + num_edge_feats  # 2*NDIM+EDIM
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, num_edge_feats],
                node_dims=[node_in, *conv_hiddens, num_node_feats],
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
        """:param graph: DGL graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param state_attr: State features
        :return: A tuple of updated features
        """
        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat, node_feat, state_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            state_feat = self.dropout(state_feat)  # pylint: disable=E1102

        return edge_feat, node_feat, state_feat


class CHGNetAtomGraphConv(nn.Module):
    """A CHGNet atom graph convolution layer in DGL."""

    def __init__(
        self,
        include_state: bool,
        node_update_func: Module,
        edge_update_func: Module | None,
        node_weight_func: Module | None,
        edge_weight_func: Module | None,
        state_update_func: Module | None,
    ):
        """
        Args:
            include_state: Whether including state
            node_update_func: Update function for nodes
            edge_update_func: Update function for edges. If None is given, the
                edges are not updated.
            node_weight_func: Weight function for radial basis functions.
                If None is given, no layer-wise weights will be used.
            edge_weight_func: Weight function for radial basis functions
                If None is given, no layer-wise weights will be used.
            state_update_func: Update function for state feats
        """
        super().__init__()
        self.include_state = include_state
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @classmethod
    def from_dims(
        cls,
        include_state: bool,
        activation: Module,
        node_dims: list[int],
        edge_dims: list[int] | None = None,
        state_dims: list[int] | None = None,
        layer_node_weights: bool = False,
        layer_edge_weights: bool = False,
        rbf_order: int | None = None,
    ) -> CHGNetAtomGraphConv:
        """Create a CHGNetAtomGraphConv layer from dimensions.

        Args:
            include_state: whether including state or not
            activation: activation function
            node_dims: NN architecture for node update function
            edge_dims: NN architecture for edge update function
            state_dims: NN architecture for state update function
            layer_node_weights: whether to use layer-wise node weights
            layer_edge_weights: whether to use layer-wise edge weights
            rbf_order: number of radial basis functions
                if either layer_node_weights or layer_edge_weights is True then
                rbf_order must be passed to initialize the weight functions

        Returns:
            CHGNetAtomGraphConv
        """
        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = (
            nn.Linear(in_features=rbf_order, out_features=node_dims[-1], bias=False) if layer_node_weights else None
        )
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:]) if edge_dims is not None else None
        edge_weight_func = (
            nn.Linear(in_features=rbf_order, out_features=edge_dims[-1], bias=False) if layer_edge_weights else None
        )
        state_update_func = MLP(state_dims, activation, activate_last=True) if include_state else None

        return cls(
            include_state, node_update_func, edge_update_func, node_weight_func, edge_weight_func, state_update_func
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, Tensor]:
        """Edge update functions.

        Args:
            edges: edges in dgl graph

        Returns:
            mij: message passing between node i and j
        """
        vi = edges.src["features"]
        vj = edges.dst["features"]
        eij = edges.data["features"]

        if self.include_state:
            u = edges.data["global_state"]
            inputs = torch.hstack([vi, eij, vj, u])
        else:
            inputs = torch.hstack([vi, eij, vj])

        eij_ = self.edge_update_func(inputs)
        if self.edge_weight_func is not None:
            rbf = edges.data["rbf"]
            rbf = rbf.float()
            eij_ = eij_ * self.edge_weight_func(rbf)

        return {"features_": eij_}

    def edge_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor) -> Tensor:
        """Perform edge update.

        Args:
            graph: DGL graph
            shared_weights: atom graph edge weights shared between convolution layers

        Returns:
            edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata["features_"] * shared_weights
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor) -> Tensor:
        """Perform node update.

        Args:
            graph: DGL atom graph
            shared_weights: atom graph node weights shared between convolution layers

        Returns:
            node_update: updated node features
        """
        eij = graph.edata["features"]
        src_id = graph.edges()[0]
        vi = graph.ndata["features"][src_id]
        dst_id = graph.edges()[1]
        vj = graph.ndata["features"][dst_id]

        if self.include_state:
            u = graph.edata["global_state"]
            inputs = torch.hstack([vi, eij, vj, u])
        else:
            inputs = torch.hstack([vi, vj, eij])

        messages = self.node_update_func(inputs)
        if self.node_weight_func is not None:
            rbf = graph.edata["rbf"]
            rbf = rbf.float()
            messages = messages * self.edge_weight_func(rbf)

        graph.edata["message"] = messages
        graph.update_all(fn.copy_e("message", "message"), fn.sum("message", "features_"))
        node_update = graph.ndata["features_"] * shared_weights
        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_attr: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: DGL graph
            state_attr: graph features

        Returns:
        state_update: state_features update
        """
        u = state_attr
        uv = dgl.readout_nodes(graph, feat="features", op="mean")
        inputs = torch.hstack([u, uv])
        state_attr = self.state_update_func(inputs)
        return state_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        state_attr: Tensor,
        shared_node_weights: Tensor,
        shared_edge_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        Args:
            graph: atom graph
            node_features: node features
            edge_features: edge features
            state_attr: state attributes
            shared_node_weights: atom graph node weights shared amongst layers
            shared_edge_weights: atom graph edge weights shared amongst layers
        """
        with graph.local_scope():
            graph.ndata["features"] = node_features
            graph.edata["features"] = edge_features

            if self.include_state:
                graph.edata["global_state"] = dgl.broadcast_edges(graph, state_attr)

            if self.edge_update_func is not None:
                edge_update = self.edge_update_(graph, shared_edge_weights)
                new_edge_features = edge_features + edge_update
                graph.edata["features"] = new_edge_features
            else:
                new_edge_features = edge_features

            node_update = self.node_update_(graph, shared_node_weights)
            new_node_features = node_features + node_update
            graph.ndata["features"] = new_node_features

            if self.include_state:
                state_attr = self.state_update_(graph, state_attr)

        return new_node_features, new_edge_features, state_attr


class CHGNetAtomGraphBlock(nn.Module):
    """A CHGNet atom graph block as a sequence of operations involving a message passing layer over the atom graph."""

    def __init__(
        self,
        num_node_feats: int,
        num_edge_feats: int,
        activation: Module,
        conv_hidden_dims: list[int],
        update_edge_feats: bool = False,
        include_state: bool = False,
        num_state_feats: int | None = None,
        layer_node_weights: bool = False,
        layer_edge_weights: bool = False,
        rbf_order: int | None = None,
        dropout: float | None = None,
    ):
        """
        Args:
            num_node_feats: number of node features
            num_edge_feats: number of edge features
            activation: activation function
            conv_hidden_dims: dimensions of hidden layers
            update_edge_feats: whether to update edge features
            include_state: whether to include state attributes
            num_state_feats: number of state features if include_state is True
            layer_node_weights: whether to include layer-wise node weights
            layer_edge_weights: whether to include layer-wise edge weights
            rbf_order: order of radial basis functions
            dropout: dropout probability
        """
        super().__init__()

        node_input_dim = 2 * num_node_feats + num_edge_feats
        if include_state:
            node_input_dim += num_state_feats
            state_dims = [num_node_feats + num_state_feats] + conv_hidden_dims + [num_state_feats]
        else:
            state_dims = None
        node_dims = [node_input_dim] + conv_hidden_dims + [num_node_feats]
        edge_dims = [node_input_dim] + conv_hidden_dims + [num_edge_feats] if update_edge_feats else None

        self.conv_layer = CHGNetAtomGraphConv.from_dims(
            include_state=include_state,
            activation=activation,
            node_dims=node_dims,
            edge_dims=edge_dims,
            state_dims=state_dims,
            layer_node_weights=layer_node_weights,
            layer_edge_weights=layer_edge_weights,
            rbf_order=rbf_order,
        )
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.out_layer = nn.Linear(num_node_feats, num_node_feats)

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_features: Tensor,
        node_features: Tensor,
        state_attr: Tensor,
        shared_node_weights: Tensor,
        shared_edge_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        Args:
            graph: atom graph
            edge_features: edge features
            node_features: node features
            state_attr: state attributes
            shared_node_weights: atom graph node weights shared amongst layers
            shared_edge_weights: atom graph edge weights shared amongst layers
        """
        node_features, edge_features, state_attr = self.conv_layer(
            graph,
            edge_features,
            node_features,
            state_attr,
            shared_node_weights,
            shared_edge_weights,
        )

        if self.dropout is not None:
            node_features = self.dropout(node_features)

        node_features = self.out_layer(node_features)
        return node_features, edge_features, state_attr


class CHGNetBondGraphConv(nn.Module):
    """A CHGNet atom graph convolution layer in DGL."""

    def __init__(
        self,
        node_update_func: Module,
        node_weight_func: Module | None,
        edge_update_func: Module | None,
        edge_weight_func: Module | None,
    ):
        """
        Args:
            node_update_func: node update function (for bond features)
            node_weight_func: layer node weight function
            edge_update_func: edge update function (for angle features)
            edge_weight_func: layer edge weight function
        """
        super().__init__()

        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func

    @classmethod
    def from_dims(
        cls,
        node_dims: list[int],
        edge_dims: list[int],
        node_weight_in_dims: int = 0,
        edge_weight_in_dims: int = 0,
    ) -> CHGNetBondGraphConv:
        """
        Args:
            node_dims:
            edge_dims:
            node_weight_in_dims:
            edge_weight_in_dims:

        Returns:
            CHGNetBondGraphConv
        """
        # TODO make sure this matches with CHGNet GatedMLP
        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(node_weight_in_dims, node_dims[-1]) if node_weight_in_dims > 0 else None

        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(edge_weight_in_dims, edge_dims[-1]) if edge_weight_in_dims > 0 else None

        return cls(node_update_func, edge_update_func, node_weight_func, edge_weight_func)

    def node_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor) -> Tensor:
        """

        Args:
            graph:
            shared_weights:

        Returns:

        """
        pass

    def edge_update_(self, graph: dgl.DGLGraph, shared_weights: Tensor) -> Tensor:
        """

        Args:
            graph:
            shared_weights:

        Returns:

        """
        pass

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        shared_node_weights: Tensor,
        shared_edge_weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pass
