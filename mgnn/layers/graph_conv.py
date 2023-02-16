"""
Graph convolution layer (GCL) implementations.
"""
from __future__ import annotations

import dgl
import torch
import torch.nn as nn
from torch.nn import Dropout, Module
from torch_scatter import scatter_sum

from mgnn.layers.core import MLP, GatedMLP
from mgnn.utils.maths import broadcast_states_to_bonds


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
        attr_update_func: Module,
    ) -> None:
        """
        Paramters:
        include_states (bood): Whether including state
        edge_update_func (Module): Update function for edges (Eq. 4)
        edge_weight_func (Module): Weight function for radial basis functions (Eq. 4)
        node_update_func (Module): Update function for nodes (Eq. 5)
        node_weight_func (Module): Weight function for radieal basis functions (Eq. 5)
        attr_update_func (Module): Update function for state feats (Eq. 6)
        """

        super().__init__()
        self.include_states = include_states
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.attr_update_func = attr_update_func

    @staticmethod
    def from_dims(
        degree, include_states, edge_dims: list[int], node_dims: list[int], attr_dims: list[int], activation: Module
    ) -> M3GNetGraphConv:
        """
        M3GNetGraphConv initialization

        Args:
        degree (int): max_n*max_l
        include_states (bool): whether including state or not
        edge_dim (list): NN architecture for edge update function
        node_dim (list): NN architecture for node update function
        state_dim (list): NN architecture for state update function
        activation (nn.Nodule): activation function

        Returns:
        M3GNetGraphConv (class)
        """
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        if include_states:
            attr_update_func = MLP(attr_dims, activation, activate_last=True)
        else:
            attr_update_func = None
        return M3GNetGraphConv(
            include_states, edge_update_func, edge_weight_func, node_update_func, node_weight_func, attr_update_func
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> torch.Tensor:
        """
        Edge update functions

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
        if self.include_states:
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        mij = {"mij": self.edge_update_func(inputs) * self.edge_weight_func(rbf)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> torch.Tensor:
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

    def node_update_(self, graph: dgl.DGLGraph, graph_attr: torch.Tensor) -> torch.Tensor:
        """
        Perform node update.

        Args:
        graph: DGL graph

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
            u = broadcast_states_to_bonds(graph, graph_attr)
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        mess_from_edge_to_node = self.node_update_func(inputs) * self.node_weight_func(rbf)
        num_nodes = graph.num_nodes()
        node_update = scatter_sum(mess_from_edge_to_node, index=src_id, dim=0, dim_size=num_nodes)
        return node_update

    def attr_update_(self, graph: dgl.DGLGraph, attrs: torch.Tensor) -> torch.Tensor:
        """
        Perform attribute (global state) update.

        Args:
        graph: DGL graph
        attrs: graph features

        Returns
        state_update: state_features update
        """
        u = attrs
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, uv])
        graph_attr = self.attr_update_func(inputs)
        return graph_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform sequence of edge->node->states updates.

        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param graph_attr: Graph attributes (global state)
        :return: (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            if self.include_states:
                graph.ndata["u"] = dgl.broadcast_nodes(graph, graph_attr)

            edge_update = self.edge_update_(graph)
            graph.edata["e"] = edge_feat + edge_update
            node_update = self.node_update_(graph, graph_attr)
            graph.ndata["v"] = node_feat + node_update
            if self.include_states:
                graph_attr = self.attr_update_(graph, graph_attr)

        return edge_feat + edge_update, node_feat + node_update, graph_attr


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
        include_states: bool = False,
        dropout: float | None = None,
    ) -> None:
        """
        :param degree: Dimension of radial basis functions
        :param num_node_feats: Number of node features
        :param num_edge_feats: Number of edge features
        :param num_state_feats: Number of state features
        :param conv_hiddens: Dimension of hidden layers
        :param activation: Activation type
        :param include_states: Including state features or not
        :param dropout: Probability of an element to be zero in dropout layer
        """
        super().__init__()

        self.activation = activation

        # compute input sizes
        if include_states:
            edge_in = 2 * num_node_feats + num_edge_feats + num_state_feats  # 2*NDIM+EDIM+GDIM
            node_in = 2 * num_node_feats + num_edge_feats + num_state_feats  # 2*NDIM+EDIM+GDIM
            attr_in = num_node_feats + num_state_feats  # NDIM+GDIM
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_states,
                edge_dims=[edge_in] + conv_hiddens + [num_edge_feats],
                node_dims=[node_in] + conv_hiddens + [num_node_feats],
                attr_dims=[attr_in] + conv_hiddens + [num_state_feats],
                activation=self.activation,
            )
        else:
            edge_in = 2 * num_node_feats + num_edge_feats  # 2*NDIM+EDIM
            node_in = 2 * num_node_feats + num_edge_feats  # 2*NDIM+EDIM
            self.conv = M3GNetGraphConv.from_dims(
                degree,
                include_states,
                edge_dims=[edge_in] + conv_hiddens + [num_edge_feats],
                node_dims=[node_in] + conv_hiddens + [num_node_feats],
                attr_dims=None,
                activation=self.activation,
            )

        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param graph: DGL graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param graph_attr: State features
        :return: A tuple of updated features
        """

        edge_feat, node_feat, graph_feat = self.conv(graph, edge_feat, node_feat, graph_feat)

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            graph_feat = self.dropout(graph_feat)  # pylint: disable=E1102

        return edge_feat, node_feat, graph_feat
