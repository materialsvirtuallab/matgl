"""
Core M3GNet model
"""
from __future__ import annotations

import logging
import os

import dgl
import numpy as np
import torch
import torch.nn as nn

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.graph.converters import Pmg2Graph
from matgl.layers.atom_ref import AtomRef
from matgl.layers.bond_expansion import BondExpansion
from matgl.layers.core import MLP, GatedMLP
from matgl.layers.cutoff_functions import polynomial_cutoff
from matgl.layers.embedding_block import EmbeddingBlock
from matgl.layers.graph_conv import M3GNetBlock
from matgl.layers.readout_block import ReduceReadOut, Set2SetReadOut, WeightedReadOut
from matgl.layers.three_body import SphericalBesselWithHarmonics, ThreeBodyInteractions

logger = logging.getLogger(__file__)
CWD = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "m3gnet"

MODEL_PATHS = {"MP-2021.2.8-EFS": os.path.join(CWD, "..", "..", "pretrained", "MP-2021.2.8-EFS")}


class M3GNet(nn.Module):
    """
    The main M3GNet model
    """

    def __init__(
        self,
        element_types: tuple[str],
        num_node_feats: int = 64,
        num_edge_feats: int = 64,
        num_state_feats: int | None = None,
        num_node_types: int | None = None,
        num_state_types: int | None = None,
        state_embedding_dim: int | None = None,
        max_n: int = 3,
        max_l: int = 3,
        n_blocks: int = 3,
        rbf_type="SphericalBessel",
        is_intensive: bool = True,
        readout_type: str = "weighted_atom",
        task_type: str = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        num_targets: int = 1,
        use_smooth: bool = False,
        use_phi: bool = False,
        num_s2s_steps: int = 3,
        num_s2s_layers: int = 3,
        field: str = "node_feat",
        device="cpu",
        include_states: bool = False,
        element_refs: np.ndarray | None = None,
        activation: str = "swish",
        **kwargs,
    ):
        r"""
        Args:
            element_types (tuple): list of elements appearing in the dataset
            num_node_feats (int): number of atomic features
            num_edge_feats (int): number of edge features
            num_state_feats (int): number of state features
            num_node_types (int): number of node types
            num_state_types (int): number of state labels
            state_embedding_dim (int): number of hidden neeurons in state embedding
            max_n (int): number of radial basis expansion
            max_l (int): number of angular expansion
            n_blocks (int): number of convolution blocks
            rbf_type (str): radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            is_intensive (bool): whether the prediction is intensive
            readout (str): the readout function type. choose from `set2set`,
                `weighted_atom` and `reduce_atom`, default to `weighted_atom`
            task_type (str): `classification` or `regression`, default to
                `regression`
            cutoff (float): cutoff radius of the graph
            threebody_cutoff (float): cutoff radius for 3 body interaction
            units (int): number of neurons in each MLP layer
            data_mean (float): optional `mean` value of the target
            data_std (float): optional `std` of the target
            num_targets (int): number of target properties
            use_smooth (bool): whether using smooth Bessel functions
            use_phi (bool): whether using phi angle
            field (str): using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            num_s2s_steps (int): number of set2set iterations
            num_s2s_layers (int): number of set2set layers
            include_states (bool): whether to include states features
            element_refs (np.ndarray): element reference values for each
                element
            activation (str): activation type. choose from 'swish', 'tanh', 'sigmoid'
            **kwargs:
        """

        # Store M3GNet model args for loading trained model
        self.model_args = {k: v for k, v in locals().items() if k not in ["self", "__class__", "kwargs"]}
        self.model_args.update(kwargs)

        super().__init__()

        if activation == "swish":
            self.activation = nn.SiLU()  # type: ignore
        elif activation == "tanh":
            self.activation = nn.Tanh()  # type: ignore
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()  # type: ignore

        self.graph_converter = Pmg2Graph(element_types=element_types, cutoff=cutoff)

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=rbf_type, smooth=use_smooth, device=device)

        degree = max_n * max_l * max_l if use_phi else max_n * max_l

        degree_rbf = max_n if use_smooth else max_n * max_l

        if num_node_types is None:
            num_node_types = len(element_types)

        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            num_node_feats=num_node_feats,
            num_edge_feats=num_edge_feats,
            num_node_types=num_node_types,
            num_state_feats=num_state_feats,
            include_states=include_states,
            state_embedding_dim=state_embedding_dim,
            activation=self.activation,
            device=device,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n, max_l=max_l, cutoff=cutoff, use_phi=use_phi, use_smooth=use_smooth, device=device
        )
        self.three_body_interactions = nn.ModuleList(
            {
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[num_node_feats, degree], activation=nn.Sigmoid(), activate_last=True, device=device
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[num_edge_feats], use_bias=False, device=device),
                )
                for _ in range(n_blocks)
            }
        )

        self.graph_layers = nn.ModuleList(
            {
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=self.activation,
                    conv_hiddens=[units, units],
                    num_node_feats=num_node_feats,
                    num_edge_feats=num_edge_feats,
                    num_state_feats=num_state_feats,
                    include_states=include_states,
                    device=device,
                )
                for _ in range(n_blocks)
            }
        )
        if is_intensive:
            input_feats = num_node_feats if field == "node_feat" else num_edge_feats
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(num_steps=num_s2s_steps, num_layers=num_s2s_layers, field=field)
                readout_feats = 2 * input_feats + num_state_feats if include_states else 2 * input_feats  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field, device=device)
                readout_feats = input_feats + num_state_feats if include_states else input_feats  # type: ignore

            dims_final_layer = [readout_feats] + [units, units] + [num_targets]
            self.final_layer = MLP(dims_final_layer, self.activation, activate_last=False, device=device)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive")
            self.final_layer = WeightedReadOut(
                in_feats=num_node_feats, dims=[units, units], num_targets=num_targets, device=device
            )
        if element_refs is not None:
            self.element_ref_calc = AtomRef(property_offset=element_refs)

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = n_blocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_states = include_states
        self.task_type = task_type
        self.is_intensive = is_intensive
        self.data_mean = data_mean
        self.data_std = data_std
        self.element_refs = element_refs
        self.device = device

    def as_dict(self):
        out = {"state_dict": self.state_dict(), "model_args": self.model_args}
        return out

    @classmethod
    def from_dict(cls, dict, **kwargs):
        """
        build a M3GNet from a saved dictionary
        """
        model = M3GNet(**dict["model_args"])
        model.load_state_dict(dict["state_dict"], **kwargs)
        return model

    @classmethod
    def from_dir(cls, path, **kwargs):
        """
        build a M3GNet from a saved directory
        """
        file_name = os.path.join(path, MODEL_NAME + ".pt")
        state = torch.load(file_name)
        print(state["model"])
        model = M3GNet.from_dict(state["model"], **kwargs)
        return model

    @classmethod
    def load(cls, model_dir: str = "MP-2021.2.8-EFS") -> M3GNet:
        """
        Load the model weights from pre-trained model (m3gnet.pt)
        Args:
            model_dir (str): directory for saved model. Defaults to "MP-2021.2.8-EFS".

        Returns: M3GNet object.
        """
        if model_dir in MODEL_PATHS:
            return cls.from_dir(MODEL_PATHS[model_dir])

        if os.path.isdir(model_dir) and "m3gnet.pt" in os.listdir(model_dir):
            return cls.from_dir(model_dir)

        raise ValueError(f"{model_dir} not found in available pretrained {list(MODEL_PATHS.keys())}")

    def forward(self, g: dgl.Graph, state_attr: torch.tensor | None = None, l_g: dgl.Graph | None = None):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        state_attr : torch.tensor
            State attrs for a batch of graphs.
        l_g : DGLGraph
            DGLGraph for a batch of line graphs.

        Returns
        -------
        ouput : torch.tensor
            Ouput property for a batch of graphs
        """
        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist

        expanded_dists = self.bond_expansion(g.edata["bond_dist"])
        if l_g is None:
            l_g = create_line_graph(g, self.threebody_cutoff)
        else:
            valid_three_body = g.edata["bond_dist"] <= self.threebody_cutoff
            l_g.ndata["bond_vec"] = g.edata["bond_vec"][valid_three_body]
            l_g.ndata["bond_dist"] = g.edata["bond_dist"][valid_three_body]
            l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][valid_three_body]
        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_basis = self.basis_expansion(g, l_g)
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
        num_node_feats, num_edge_feats, num_state_feats = self.embedding(node_types, g.edata["rbf"], state_attr)
        for i in range(self.n_blocks):
            num_edge_feats = self.three_body_interactions[i](
                g, l_g, three_body_basis, three_body_cutoff, num_node_feats, num_edge_feats
            )
            num_edge_feats, num_node_feats, num_state_feat = self.graph_layers[i](
                g, num_edge_feats, num_node_feats, num_state_feats
            )
        g.ndata["node_feat"] = num_node_feats
        g.edata["edge_feat"] = num_edge_feats
        if self.is_intensive:
            node_vec = self.readout(g)
            vec = torch.hstack([node_vec, state_attr]) if self.include_states else node_vec
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            g.ndata["atomic_properties"] = self.final_layer(g)
            output = dgl.readout_nodes(g, "atomic_properties", op="sum")
        output = output * self.data_std + self.data_mean
        output = torch.squeeze(output)
        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_ref_calc(g))
            output += property_offset
        return output
