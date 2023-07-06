"""Implementation of the Crystal Hamiltonian Graph Network (CHGNet) model.

CHGNet is a graph neural network model that includes 3 body interactions in a similar
fashion to m3gnet, and also includes charge information by including training and
prediction of site-wise magnetic moments.

The CHGNet model is described in the following paper: https://arxiv.org/abs/2302.14231
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence, TYPE_CHECKING

import dgl
import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENT_TYPES
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_line_graph,
)
from matgl.layers import (
    MLP,
    FourierExpansion,
    RadialBesselFunction,
    ReduceReadOut,
    WeightedReadOut,
    Set2SetReadOut,
    CHGNetAtomGraphBlock,
    CHGNetBondGraphBlock,
)
from matgl.utils.io import IOMixIn
from matgl.utils.cutoff import polynomial_cutoff


if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__name__)


class CHGNet(nn.Module, IOMixIn):
    """Main CHGNet model."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str] | None = None,
        dim_atom_embedding: int = 64,
        dim_bond_embedding: int = 64,
        dim_angle_embedding: int = 64,
        dim_state_embedding: int | None = None,
        dim_state_feats: int | None = None,
        non_linear_bond_embedding: bool = False,
        non_linear_angle_embedding: bool = False,
        cutoff: float = 5.0,
        threebody_cutoff: float = 3.0,
        cutoff_exponent: int = 5,
        max_n: int = 9,
        max_f: int = 4,
        learn_basis: bool = True,
        num_blocks: int = 4,
        shared_bond_weights: Literal["bond", "three_body_bond", "both"] | None = "both",
        layer_bond_weights: Literal["bond", "three_body_bond", "both"] | None = None,
        atom_conv_hidden_dims: Sequence[int] = (64,),
        bond_conv_hidden_dims: Sequence[int] = (64,),
        angle_layer_hidden_dim: Sequence[int] | None = None,
        conv_dropout: float = 0.0,
        num_targets: int = 1,
        num_site_targets: int = 1,
        readout_type: Literal["sum", "average", "attention", "weighted"] = "sum",
        readout_mlp_first: bool = True,
        readout_field: Literal["node_feat", "edge_feat"] = "node_feat",
        readout_hidden_dims: Sequence[int] = (64, 64),
        readout_dropout: float = 0.0,
        activation_type: str = "swish",
        is_intensive: bool = True,
        task_type: Literal["regression", "classification"] = "regression",
        **kwargs,
    ):
        """
        Args:
            element_types: List of element types to consider in the model. If None, defaults to
                DEFAULT_ELEMENT_TYPES.
            dim_atom_embedding: Dimension of the atom embedding.
            dim_bond_embedding: Dimension of the bond embedding.
            dim_angle_embedding: Dimension of the angle embedding.
            dim_state_embedding: Dimension of the state embedding.
            dim_state_feats: Dimension of the state features.
            non_linear_bond_embedding: Whether to use a non-linear embedding for the bond features (single layer MLP).
            non_linear_angle_embedding: Whether to use a non-linear embedding for the angle features (single layer MLP).
            cutoff: cutoff radius of atom graph.
            threebody_cutoff: cutoff radius for bonds included in bond graph for three body interactions.
            cutoff_exponent: exponent for the smoothing cutoff function.
            max_n: maximum number of basis functions for the bond length radial expansion.
            max_f: maximum number of basis functions for the angle Fourier expansion.
            learn_basis: whether to learn the frequencies used in basis functions or use fixed basis functions.
            num_blocks: number of graph convolution blocks.
            shared_bond_weights: whether to share bond weights among layers in the atom and bond graph blocks.
            layer_bond_weights: whether to use independent weights in each convolution layer.
            atom_conv_hidden_dims: hidden dimensions for the atom graph convolution layers.
            bond_conv_hidden_dims: hidden dimensions for the bond graph convolution layers.
            angle_layer_hidden_dim: hidden dimensions for the angle layer.
            conv_dropout: dropout probability for the graph convolution layers.
            num_targets: number of targets to predict.
            num_site_targets: number of site-wise targets to predict. (ie magnetic moments)
            readout_type: type of readout to use.
            readout_mlp_first: whether to apply the readout MLP before or after the readout function.
            readout_field: field to readout from the graph.
            readout_hidden_dims: hidden dimensions for the readout MLP.
            readout_dropout: dropout probability for the readout MLP.
            activation_type: activation function to use.
            is_intensive: whether the target is intensive or extensive.
            task_type: type of task to perform.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        # TODO implement a "get_activation" function with available activations to avoid
        #  this if/else block
        if activation_type == "swish":
            activation = nn.SiLU()  # type: ignore
        elif activation_type == "tanh":
            activation = nn.Tanh()  # type: ignore
        elif activation_type == "sigmoid":
            activation = nn.Sigmoid()  # type: ignore
        elif activation_type == "softplus2":
            activation = SoftPlus2()  # type: ignore
        elif activation_type == "softexp":
            activation = SoftExponential()  # type: ignore
        else:
            raise Exception("Undefined activation type, please try using swish, sigmoid, tanh, softplus2, softexp")

        if task_type == "classification":
            raise NotImplementedError("classification with CHGNet not yet implemented")

        element_types = element_types or DEFAULT_ELEMENT_TYPES

        # basis expansions for bond lengths, triple interaction bond lengths and angles
        self.bond_expansion = RadialBesselFunction(max_n=max_n, cutoff=cutoff, learnable=learn_basis)
        self.threebody_bond_expansion = RadialBesselFunction(
            max_n=max_n, cutoff=threebody_cutoff, learnable=learn_basis
        )
        self.angle_expansion = FourierExpansion(max_f=max_f, learnable=learn_basis)

        # embedding block for atom, bond, angle, and optional state features
        self.include_states = dim_state_feats is not None
        self.state_embedding = nn.Embedding(dim_state_feats, dim_state_embedding) if self.include_states else None
        self.atom_embedding = nn.Embedding(len(element_types), dim_atom_embedding)
        self.bond_embedding = MLP(
            [max_n, dim_bond_embedding], activation=activation, activate_last=non_linear_bond_embedding
        )
        self.angle_embedding = MLP(
            [2 * max_f + 1, dim_angle_embedding], activation=activation, activate_last=non_linear_angle_embedding
        )

        # shared message bond distance smoothing weights
        self.atom_bond_weights = (
            nn.Linear(max_n, dim_atom_embedding, bias=False) if shared_bond_weights in ["bond", "both"] else None
        )
        self.bond_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False) if shared_bond_weights in ["bond", "both"] else None
        )
        self.threebody_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False)
            if shared_bond_weights in ["three_body_bond", "both"]
            else None
        )

        # operations involving the graph (i.e. atom graph) to update atom and bond features
        self.atom_graph_layers = nn.ModuleList(
            [
                CHGNetAtomGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    #  this activation only applies to state update MLP, gMLP in core has silu hard-coded
                    activation=activation,
                    conv_hidden_dims=atom_conv_hidden_dims,
                    update_edge_feats=False,
                    include_state=self.include_states,
                    num_state_feats=dim_state_embedding,
                    dropout=conv_dropout,
                    rbf_order=max_n if layer_bond_weights in ["bond", "both"] else 0,
                )
                for _ in range(num_blocks)
            ]
        )

        # operations involving the line graph (i.e. bond graph) to update bond and angle features
        self.bond_graph_layers = nn.ModuleList(
            [
                CHGNetBondGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    num_angle_feats=dim_angle_embedding,
                    bond_hidden_dims=bond_conv_hidden_dims,
                    angle_hidden_dims=angle_layer_hidden_dim if angle_layer_hidden_dim is not None else [],
                    bond_dropout=conv_dropout,
                    angle_dropout=conv_dropout,
                    rbf_order=max_n if layer_bond_weights in ["three_body_bond", "both"] else 0,
                )
                for _ in range(num_blocks - 1)
            ]
        )

        self.sitewise_readout = (
            nn.Linear(dim_atom_embedding, num_site_targets) if num_site_targets > 0 else lambda x: None
        )

        # TODO different allowed readouts for intensive/extensive targets and task types
        input_readout_dim = dim_atom_embedding if readout_field == "node_feat" else dim_bond_embedding
        self.readout_mlp = MLP([input_readout_dim, *readout_hidden_dims, num_targets], activation)

        if readout_type == "sum":  # mlp first then reduce
            self.readout = ReduceReadOut("sum", field=readout_field)
        elif readout_type == "average":  # reduce then mlp
            self.readout = ReduceReadOut("mean", field=readout_field)
        elif readout_type == "weighted":  # weighted reduce
            self.readout = WeightedReadOut(
                in_feats=input_readout_dim, dims=readout_hidden_dims, num_targets=num_targets
            )
        elif readout_type == "attention":  # attention reduce
            raise NotImplementedError  # TODO implement attention readout
        else:
            raise Exception("Undefined readout type, please try using mlp_reduce, reduce, weighted, attention")

        self.element_types = element_types
        self.max_n = max_n
        self.max_f = max_f
        self.n_blocks = num_blocks
        self.cutoff = cutoff
        self.cutoff_exponent = cutoff_exponent
        self.three_body_cutoff = threebody_cutoff
        self.readout_type = readout_type
        self.readout_mlp_first = readout_mlp_first
        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self, graph: dgl.DGLGraph, state_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model.

        Args:
            graph (dgl.DGLGraph): Input graph.
            state_features (torch.Tensor, optional): State features. Defaults to None.

        Returns:
            torch.Tensor: Model output.
        """

        # TODO should all ops be graph.local_scope? otherwise we are changing the state of the graph implicitly

        # compute bond vectors and distances and add to graph  # TODO this is better done in the graph converter
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist
        graph.edata["bond_expansion"] = self.bond_expansion(bond_dist)  # TODO smooth here or in block?
        polynomial_cutoff(self.bond_expansion(bond_dist), self.cutoff, self.cutoff_exponent)

        # create bond graph (line graph) with necessary node and edge data
        bond_graph = create_line_graph(graph, self.three_body_cutoff)
        # TODO: this may be moved into create_line_graph
        bond_indices = torch.arange(0, graph.num_edges())[graph.edata["bond_dist"] <= self.three_body_cutoff]
        bond_graph.ndata["bond_index"] = bond_indices
        # TODO smooth here or in block?
        bond_graph.ndata["bond_expansion"] = self.threebody_bond_expansion(bond_dist[bond_graph.ndata["bond_index"]])
        # TODO double check if this is correct
        bond_graph.edata["center_atom_index"] = torch.gather(graph.edges()[1], 0, bond_graph.edges()[1])
        # TODO only compute theta (and not cos_theta)
        bond_graph.apply_edges(compute_theta)
        bond_graph.edata["angle_expansion"] = self.angle_expansion(bond_graph.edata["theta"])

        # compute state, atom, bond and angle embeddings
        if self.state_embedding is not None and state_features is not None:
            state_features = self.state_embedding(state_features)
        else:
            state_features = None
        atom_features = self.atom_embedding(graph.ndata["node_type"])
        bond_features = self.bond_embedding(graph.edata["bond_expansion"])
        angle_features = self.angle_embedding(bond_graph.edata["angle_expansion"])

        # shared message weights
        atom_bond_weights = (
            self.atom_bond_weights(graph.edata["bond_expansion"]) if self.atom_bond_weights is not None else None
        )
        bond_bond_weights = (
            self.bond_bond_weights(graph.edata["bond_expansion"]) if self.bond_bond_weights is not None else None
        )
        threebody_bond_weights = (
            self.threebody_bond_weights(bond_graph.ndata["bond_expansion"])
            if self.threebody_bond_weights is not None
            else None
        )

        # message passing layers
        for i in range(self.n_blocks - 1):
            atom_features, bond_features, state_features = self.atom_graph_layers[i](
                graph, atom_features, bond_features, state_features, atom_bond_weights, bond_bond_weights
            )
            bond_features, angle_features = self.bond_graph_layers[i](
                bond_graph, atom_features, bond_features, angle_features, threebody_bond_weights
            )

        # site wise target readout
        site_properties = self.sitewise_readout(atom_features)

        # last atom graph message passing layer
        atom_features, bond_features, state_features = self.atom_graph_layers[-1](
            graph, atom_features, bond_features, state_features, atom_bond_weights, bond_bond_weights
        )

        graph.ndata["atom_features"] = atom_features
        graph.edata["bond_features"] = bond_features
        # bond_graph.edata["angle_features"] = angle_features

        # readout  # TODO return crystal features?
        if self.readout_mlp_first:
            graph.ndata["node_feat"] = self.readout_mlp(atom_features)
            structure_properties = self.readout(graph)
        else:
            graph.ndata["node_feat"] = atom_features
            crystal_features = self.readout(graph)
            structure_properties = self.readout_mlp(crystal_features) * graph.num_nodes()

        structure_properties = torch.squeeze(structure_properties)

        return structure_properties, site_properties

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        # TODO copied verbatim from m3gnet, should refactor?

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)

        graph, state_feats_default = graph_converter.get_graph(structure)
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        output = self(graph=graph, state_features=state_feats)
        return [out.detach() for out in output]
