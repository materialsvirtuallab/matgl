"""Implementation of Materials 3-body Graph Network (M3GNet) model.

The main improvement over MEGNet is the addition of many-body interactios terms, which improves efficiency of
representation of local interactions for applications such as interatomic potentials. For more details on M3GNet,
please refer to::

    Chen, C., Ong, S.P. _A universal graph deep learning interatomic potential for the periodic table._ Nature
    Computational Science, 2023, 2, 718-728. DOI: 10.1038/s43588-022-00349-3.

"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import dgl
import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENT_TYPES
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.layers import (
    MLP,
    BondExpansion,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,
    ReduceReadOut,
    Set2SetReadOut,
    SoftExponential,
    SoftPlus2,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
    WeightedReadOut,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)


class M3GNet(nn.Module, IOMixIn):
    """The main M3GNet model."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str],
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int | None = None,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        max_n: int = 3,
        max_l: int = 3,
        nblocks: int = 3,
        rbf_type="SphericalBessel",
        is_intensive: bool = True,
        readout_type: str = "weighted_atom",
        task_type: str = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        use_smooth: bool = False,
        use_phi: bool = False,
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: str = "node_feat",
        include_state: bool = False,
        activation_type: str = "swish",
        **kwargs,
    ):
        """
        Args:
            element_types (tuple): list of elements appearing in the dataset
            dim_node_embedding (int): number of embedded atomic features
            dim_edge_embedding (int): number of edge features
            dim_state_embedding (int): number of hidden neurons in state embedding
            dim_state_feats (int): number of state features after linear layer
            ntypes_state (int): number of state labels
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
        super().__init__()

        self.save_args(locals(), kwargs)

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

        if element_types is None:
            self.element_types = DEFAULT_ELEMENT_TYPES
        else:
            self.element_types = element_types  # type: ignore

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=rbf_type, smooth=use_smooth)

        degree = max_n * max_l * max_l if use_phi else max_n * max_l

        degree_rbf = max_n if use_smooth else max_n * max_l

        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            use_phi=use_phi,
            use_smooth=use_smooth,
        )
        self.three_body_interactions = nn.ModuleList(
            {
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            }
        )

        if dim_state_feats is None:
            dim_state_feats = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            {
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    num_node_feats=dim_node_embedding,
                    num_edge_feats=dim_edge_embedding,
                    num_state_feats=dim_state_feats,
                    include_state=include_state,
                )
                for _ in range(nblocks)
            }
        )
        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(num_steps=niters_set2set, num_layers=nlayers_set2set, field=field)
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats + dim_state_feats if include_state else input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding, dims=[units, units], num_targets=ntargets  # type: ignore
            )

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_states = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
    ):
        """Performs message passing and updates node representations.

        Args:
            g : DGLGraph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            l_g : DGLGraph for a batch of line graphs.

        Returns:
            output: Output property for a batch of graphs
        """
        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)

        expanded_dists = self.bond_expansion(g.edata["bond_dist"])
        if l_g is None:
            l_g = create_line_graph(g, self.threebody_cutoff)
        else:
            if l_g.num_nodes() == g.num_edges():
                valid_three_body = g.edata["bond_dist"] <= self.threebody_cutoff
                l_g.ndata["bond_vec"] = g.edata["bond_vec"][valid_three_body]
                l_g.ndata["bond_dist"] = g.edata["bond_dist"][valid_three_body]
                l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][valid_three_body]
            else:
                three_body_id = torch.unique(torch.concatenate(l_g.edges()))
                max_three_body_id = max(torch.cat([three_body_id + 1, torch.tensor([0])]))
                l_g.ndata["bond_vec"] = g.edata["bond_vec"][:max_three_body_id]
                l_g.ndata["bond_dist"] = g.edata["bond_dist"][:max_three_body_id]
                l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][:max_three_body_id]
        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_basis = self.basis_expansion(l_g)
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
        node_feat, edge_feat, state_feat = self.embedding(node_types, g.edata["rbf"], state_attr)
        for i in range(self.n_blocks):
            edge_feat = self.three_body_interactions[i](
                g,
                l_g,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
        g.ndata["node_feat"] = node_feat
        g.edata["edge_feat"] = edge_feat
        if self.is_intensive:
            node_vec = self.readout(g)
            vec = torch.hstack([node_vec, state_feat]) if self.include_states else node_vec  # type: ignore
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            g.ndata["atomic_properties"] = self.final_layer(g)
            output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, state_feats_default = graph_converter.get_graph(structure)
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
