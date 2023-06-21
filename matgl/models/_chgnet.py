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
        cutoff_smoothing_exponent: float = 5.0,
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
        # is_intensive: bool = True,

        # additional args from m3gnet mgl
        # units: int = 64,
        # ntargets: int = 1,
        # field: str = "node_feat",
        # include_state: bool = False,
        # activation_type: str = "swish",
        # is_intensive: bool = True,
        # readout_type: str = "average",  # or attention
        # task_type: str = "regression",
        **kwargs
    ):
        """
        Args:
            element_types (tuple): list of elements appearing in the dataset
            dim_node_embedding (int): number of embedded atomic features
            dim_edge_embedding (int): number of edge features
            dim_state_embedding (int): number of hidden neurons in state embedding
            dim_state_feats (int): number of state features after linear layer
            dim_state_types (int): number of state labels
            max_n (int): number of radial basis expansion
            max_l (int): number of angular expansion
            nblocks (int): number of convolution blocks
            rbf_type (str): radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            is_intensive (bool): whether the prediction is intensive
            readout_type (str): the readout function type. choose from `set2set`,
            `weighted_atom` and `reduce_atom`, default to `weighted_atom`
            task_type (str): `classification` or `regression`, default to
            `regression`
            cutoff (float): cutoff radius of the graph
            threebody_cutoff (float): cutoff radius for 3 body interaction
            units (int): number of neurons in each MLP layer
            ntargets (int): number of target properties
            use_smooth (bool): whether using smooth Bessel functions
            use_phi (bool): whether using phi angle
            field (str): using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            niters_set2set (int): number of set2set iterations
            nlayers_set2set (int): number of set2set layers
            include_state (bool): whether to include states features
            activation_type (str): activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            **kwargs: For future flexibility. Not used at the moment.
        """