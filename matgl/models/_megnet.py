"""
Implementation of MEGNet model.
"""
from __future__ import annotations

import logging

import dgl
import torch
from dgl.nn import Set2Set
from pymatgen.core import Structure
from torch import nn

from matgl.config import DEFAULT_ELEMENT_TYPES
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.graph.converters import GraphConverter
from matgl.layers import MLP, BondExpansion, EdgeSet2Set, EmbeddingBlock, MEGNetBlock, SoftExponential, SoftPlus2
from matgl.utils.io import IOMixIn

logger = logging.getLogger(__file__)


class MEGNet(nn.Module, IOMixIn):
    """
    DGL implementation of MEGNet.
    """

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: int | None = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float | None = None,
        graph_transformations: list | None = None,
        element_types: tuple[str, ...] | None = None,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ):
        """
        Construct a MEGNet model. Useful defaults for all arguments have been specified based on MEGNet formation energy
        model.

        Args:
            dim_node_embedding: Dimension of node embedding.
            dim_edge_embedding: Dimension of edge embedding.
            dim_state_embedding: Dimension of state embedding.
            ntypes_state: Number of state types.
            nblocks: Number of blocks.
            hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
            hidden_layer_sizes_conv: Architecture of dense layers for message and update functions
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iterations in Set2Set layer
            hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
            activation_type: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            layer_node_embedding: Architecture of embedding layer for node attributes
            layer_edge_embedding: Architecture of embedding layer for edge attributes
            layer_state_embedding: Architecture of embedding layer for state attributes
            include_state: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution
            graph_transformations: Perform a graph transformation, e.g., incorporate three-body interactions, prior to
                performing the GCL updates.
            element_types: Elements included in the training set
            bond_expansion: Gaussian expansion for edge attributes
            cutoff: cutoff for forming bonds
            gauss_width: width of Gaussian function for bond expansion
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        self.element_types = element_types or DEFAULT_ELEMENT_TYPES
        self.cutoff = cutoff
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=dim_edge_embedding, width=gauss_width
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        if activation_type == "swish":
            activation = nn.SiLU()  # type: ignore
        elif activation_type == "sigmoid":
            activation = nn.Sigmoid()  # type: ignore
        elif activation_type == "tanh":
            activation = nn.Tanh()  # type: ignore
        elif activation_type == "softplus2":
            activation = SoftPlus2()  # type: ignore
        elif activation_type == "softexp":
            activation = SoftExponential()  # type: ignore
        else:
            raise Exception("Undefined activation type, please try using swish, sigmoid, tanh, softplus2, softexp")

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        # first block
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)]  # type: ignore
        # other blocks
        for _ in range(nblocks - 1):
            blocks.append(MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args))  # type: ignore
        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2Set(dim_blocks_out, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.is_classification = is_classification
        self.graph_transformations = graph_transformations or [nn.Identity()] * nblocks
        self.include_state_embedding = include_state

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        state_feat: torch.Tensor,
    ):
        """
        Forward pass of MEGnet. Executes all blocks.

        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param state_feat: State features.
        :return: Prediction
        """
        graph_transformations = self.graph_transformations
        node_feat, edge_feat, state_feat = self.embedding(node_feat, edge_feat, state_feat)
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        for gt, block in zip(graph_transformations, self.blocks):
            output = block(gt(graph), edge_feat, node_feat, state_feat)
            edge_feat, node_feat, state_feat = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec, state_feat])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return output

    def predict_structure(
        self,
        structure: Structure,
        state_feats: torch.tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """
        Convenience method to directly predict property from structure.

        Args:
            structure (Structure): Pymatgen structure
            state_feats (torch.tensor): graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, state_feats_default = graph_converter.get_graph(structure)
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)
        return self(g, g.edata["edge_attr"], g.ndata["node_type"], state_feats).detach()
