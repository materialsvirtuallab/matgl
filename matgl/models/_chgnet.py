"""Implementation of the Crystal Hamiltonian Graph Network (CHGNet) model.

CHGNet is a graph neural network model that includes 3 body interactions in a similar
fashion to m3gnet, and also includes charge information by including training and
prediction of site-wise magnetic moments.

The CHGNet model is described in the following paper: https://arxiv.org/abs/2302.14231
"""

from __future__ import annotations

import logging

import dgl
import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENT_TYPES
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)

from matgl.graph.converters import GraphConverter

from matgl.layers import (
    MLP,
    GatedMLP,
    BondExpansion,
    EmbeddingBlock,
    FourierExpansion,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.io import IOMixIn


logger = logging.getLogger(__name__)


class CHGNet(nn.Module, IOMixIn):
    """Main CHGNet model."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str],
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_angle_embedding: int = 64,
        dim_state_embedding: int | None = None,
        dim_state_types: int | None = None,
        dim_state_feats: int | None = None,
        cutoff: float = 5.0,
        threebody_cutoff: float = 3.0,
        cutoff_exponent: float = 5.0,
        max_n: int = 9,  # number of l = 0 bessel function frequencies
        max_f: int = 4,  # number of Fourier frequencies -> 4 * 2 + 1 = 9 of original CHGNet
        learn_basis_freq: bool = True,
        nblocks: int = 4,

        # missing args from original chgnet
        # atom_conv_hidden_dim: Sequence[int] | int = 64,
        # update_bond: bool = True,
        # bond_conv_hidden_dim: Sequence[int] | int = 64,
        # update_angle: bool = True,
        # angle_layer_hidden_dim: Sequence[int] | int = 0,
        # conv_dropout: float = 0,
        # read_out: str = "ave",
        # mlp_hidden_dims: Sequence[int] | int = (64, 64),
        # mlp_dropout: float = 0,
        # mlp_first: bool = True,


        # additional args from m3gnet mgl
        # units: int = 64,
        # ntargets: int = 1,
        # field: str = "node_feat",
        include_state: bool = False,
        activation_type: str = "swish",
        is_intensive: bool = True,
        # readout_type: str = "average",  # or attention
        # task_type: str = "regression",
        **kwargs
    ):
        """"""
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
            raise Exception(
                "Undefined activation type, please try using swish, sigmoid, tanh, softplus2, softexp"
            )

        if element_types is None:
            self.element_types = DEFAULT_ELEMENT_TYPES  # make sure these match CHGNet
        else:
            self.element_types = element_types

        # TODO change to a simple learnable 0-th order bessel function with max_n roots
        self.bond_expansion = BondExpansion(max_l=1, max_n=max_n, rbf_type="SphericalBessel", cutoff=cutoff)
        # TODO nn.Linear for weights of bond expansion
        # TODO need an rbf for the graph bonds using the threebody cutoff and nn.Linear for weights
        self.angle_expansion = None  # implement this

        # feature embeddings
        self.atom_embedding = nn.Embedding(len(element_types), dim_node_embedding)
        # TODO add option for activation
        self.bond_embedding = MLP([max_n, dim_edge_embedding], activation=activation, activate_last=False)
        self.angle_embedding = MLP([max_f, dim_angle_embedding], activation=activation, activate_last=False)
        self.state_embedding = nn.Embedding(dim_state_types, dim_state_feats) if include_state else None

        # operations involving the line graph (i.e. bond graph) to update bond and angle features
        self.three_body_interactions = nn.ModuleList(
            {
                None  # implement the BondConvolution and AngleUpdate
                # in here calculate the
                for _ in range(nblocks - 1)
            }
        )

        # operations involving the graph (i.e. atom graph) to update atom and bond features
        self.graph_layers = nn.ModuleList(
            {
                None  # implement the AtomConvolution and BondUpdate
                for _ in range(nblocks)
            }
        )

        self.magmom_readout = nn.Linear(dim_node_embedding, 1)

        # TODO implement the readout layer attrs
        self.final_layer = None  # MLP or GatedMLP with Linear readout

        self.max_n = max_n
        self.max_f = max_f
        self.n_blocks = nblocks
        self.cutoff = cutoff
        self.cutoff_exponent = cutoff_exponent
        self.three_body_cutoff = threebody_cutoff
        self.include_states = include_state
        self.is_intensive = is_intensive
