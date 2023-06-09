"""
Graph convolution layer (GCL) implementations.
"""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Identity, Module

from matgl.layers._core import MLP, GatedMLP


class MEGNetGraphConv(Module):
    """
    A MEGNet graph convolution layer in DGL.
    """

    def __init__(
        self,
        edge_func: Module,
        node_func: Module,
        state_func: Module,
    ) -> None:
        """
        :param edge_func: Edge update function.
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
        """
        Perform edge update.

        :param graph: Input graph
        :return: Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        graph.edata["e"] = graph.edata.pop("mij")
        return graph.edata["e"]

    def node_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """
        Perform node update.

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
        """
        Perform attribute (global state) update.

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
        """
        Perform sequence of edge->node->attribute updates.

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
    """
    A MEGNet block comprising a sequence of update operations.
    """

    def __init__(
        self, dims: list[int], conv_hiddens: list[int], act: Module, dropout: float | None = None, skip: bool = True
    ) -> None:
        """
        TODO: Add docs.
        :param dims: architecture of dense layers before graph convolution
        :param conv_hiddens: architecture of graph convolution
        :param act: activation type
        :param dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according
            to a Bernoulli distribution
        :param skip: residual block.
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
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else Identity()
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else Identity()
        self.state_func = MLP(**mlp_kwargs) if self.has_dense else Identity()

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
    """
    A M3GNet graph convolution layer in DGL.
    """

    def __init__(
        self,
        include_states: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ):
        """
        Parameters:
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
        """
        M3GNetGraphConv initialization.

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
        """
        Edge update functions.

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
        """
        Perform edge update.

        Args:
        graph: DGL graph

        Returns:
        edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata.pop("mij")
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, state_attr: Tensor) -> Tensor:
        """
        Perform node update.

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
        """
        Perform attribute (global state) update.

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
        """
        Perform sequence of edge->node->states updates.

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
    """
    A M3GNet block comprising a sequence of update operations.
    """

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
        """
        :param degree: Dimension of radial basis functions
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
        """
        :param graph: DGL graph
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
