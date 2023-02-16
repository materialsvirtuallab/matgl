"""
The core m3gnet model
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dgl.nn import Set2Set
from torch_scatter import scatter

from mgnn.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from mgnn.layers.atom_ref import AtomRef
from mgnn.layers.bond_expansion import BondExpansion
from mgnn.layers.core import MLP, GatedMLP
from mgnn.layers.cutoff_functions import polynomial_cutoff
from mgnn.layers.embedding_block import EmbeddingBlock
from mgnn.layers.graph_conv import M3GNetBlock
from mgnn.layers.readout_block import ReduceReadOut, WeightedReadOut
from mgnn.layers.three_body import SphericalBesselWithHarmonics, ThreeBodyInteractions
from mgnn.utils.maths import get_segment_indices_from_n


class M3GNet(nn.Module):
    """
    The main M3GNet model
    """

    def __init__(
        self,
        element_types,
        num_node_feats: int = 16,
        num_edge_feats: int = 16,
        num_state_feats: int | None = None,
        max_n: int = 3,
        max_l: int = 3,
        n_blocks: int = 3,
        units: int = 64,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        activation: str = "swish",
        include_states: bool = False,
        state_embedding_dim: int | None = None,
        mean: float = 0.0,
        std: float = 1.0,
        element_refs: np.ndarray | None = None,
        is_intensive: bool = True,
        readout: str = "weighted_atom",
        task_type: str = "regression",
        rbf_type="SphericalBessel",
        use_phi: bool = False,
        use_smooth: bool = True,
        num_s2s_steps: int = 3,
        num_s2s_layers: int = 3,
        num_targets: int = 1,
    ):
        r"""
        Args:
            max_n (int): number of radial basis expansion
            max_l (int): number of angular expansion
            n_blocks (int): number of convolution blocks
            units (int): number of neurons in each MLP layer
            cutoff (float): cutoff radius of the graph
            threebody_cutoff (float): cutoff radius for 3 body interaction
            n_atom_types (int): number of atom types
            include_states (bool): whether to include states calculation
            readout (str): the readout function type. choose from `set2set`,
                `weighted_atom` and `reduce_atom`, default to `weighted_atom`
            task_type (str): `classification` or `regression`, default to
                `regression`
            is_intensive (bool): whether the prediction is intensive
            mean (float): optional `mean` value of the target
            std (float): optional `std` of the target
            element_refs (np.ndarray): element reference values for each
                element
            **kwargs:
        """
        super().__init__()
        self.cutoff = cutoff
        self.rbf_type = rbf_type
        self.use_phi = use_phi
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=self.rbf_type, smooth=use_smooth)

        self.embedding = EmbeddingBlock(
            num_node_feats=num_node_feats,
            num_edge_feats=num_edge_feats,
            num_state_feats=num_state_feats,
            include_states=include_states,
            state_embedding_dim=state_embedding_dim,
            activation=activation,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n, max_l=max_l, cutoff=cutoff, use_phi=use_phi, use_smooth=use_smooth
        )
        if use_phi:
            degree = max_n * max_l * max_l
        else:
            degree = max_n * max_l

        self.three_body_interactions = [
            ThreeBodyInteractions(
                update_network_atom=MLP(dims=[num_node_feats, degree], activation=nn.Sigmoid()),
                update_network_bond=GatedMLP(in_feats=degree, dims=[num_edge_feats], use_bias=False),
            )
            for _ in range(n_blocks)
        ]
        if use_smooth:
            degree_rbf = max_n
        else:
            degree_rbf = max_n * max_l
        layer = M3GNetBlock(
            degree=degree_rbf,
            activation=self.activation,
            conv_hiddens=[units, units],
            num_node_feats=num_node_feats,
            num_edge_feats=num_edge_feats,
            num_state_feats=num_state_feats,
            include_states=include_states,
        )
        self.graph_layers = []

        for i in range(n_blocks):
            self.graph_layers.append(layer)

        if is_intensive:
            if readout == "set2set":
                self.atom_readout = Set2Set(num_node_feats, n_iters=num_s2s_steps, n_layers=num_s2s_layers)
                if include_states:
                    readout_feats = 2 * num_node_feats + num_state_feats
                else:
                    readout_feats = 2 * num_node_feats
            else:
                self.atom_readout = ReduceReadOut("mean", field="node_feat")
                if include_states:
                    readout_feats = num_node_feats + num_state_feats
                else:
                    readout_feats = num_node_feats
            dims_final_layer = [readout_feats] + [units, units] + [num_targets]
            self.final_layer = MLP(dims_final_layer, self.activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive")
            self.final_layer = WeightedReadOut(in_feats=num_node_feats, dims=[units, units], num_targets=num_targets)
        if element_refs is None:
            element_refs = torch.zeros(len(element_types))
            self.element_ref_calc = AtomRef(property_offset=element_refs)
        else:
            self.element_ref_calc = AtomRef(property_offset=element_refs)

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = n_blocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_states = include_states
        self.readout = readout
        self.task_type = task_type
        self.is_intensive = is_intensive
        self.mean = mean
        self.std = std
        self.element_refs = element_refs

    def forward(self, g, state_attr) -> torch.tensor:
        """
        Args:
            graph (list): list repr of a MaterialGraph
            **kwargs:
        Returns:

        """

        bv, bd = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bv
        g.edata["bond_dist"] = bd
        property_offset = self.element_ref_calc(g)
        l_g = create_line_graph(g, self.threebody_cutoff)  # TODO(Kenko): check if we can use load graphs for l_g
        l_g.apply_edges(compute_theta_and_phi)
        bond_basis = self.bond_expansion(bd)
        g.edata["rbf"] = bond_basis
        three_body_basis = self.basis_expansion(g, l_g)
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
        node_attr = g.ndata["attr"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = self.embedding(node_attr, edge_attr, state_attr)

        for i in range(self.n_blocks):
            edge_feat_updated = self.three_body_interactions[i](
                g, l_g, three_body_basis, three_body_cutoff, node_feat, edge_feat
            )
            edge_feat = edge_feat_updated
            edge_feat_new, node_feat_new, state_feat_new = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
            node_feat, edge_feat, state_feat = node_feat_new, edge_feat_new, state_feat_new

        g.ndata["node_feat"] = node_feat
        g.edata["edge_feat"] = edge_feat

        if self.is_intensive:
            node_vec = self.atom_readout(g)
            if self.include_states:
                vec = torch.hsatack([node_vec, state_feat])
            else:
                vec = node_vec
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            atomic_properties = self.final_layer(g)
            segment_index = get_segment_indices_from_n(g.batch_num_nodes())
            output = scatter(atomic_properties, segment_index, dim=0, reduce="sum", dim_size=g.batch_size)
        output = output * self.std + self.mean
        property_offset = property_offset.reshape(property_offset.size(dim=0), -1)
        #        output = output.reshape(-1,)
        output += property_offset
        return output
