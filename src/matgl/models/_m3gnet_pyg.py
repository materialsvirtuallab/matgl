"""Implementation of Materials 3-body Graph Network (M3GNet) model for PyG.

The main improvement over MEGNet is the addition of many-body interactios terms, which improves efficiency of
representation of local interactions for applications such as interatomic potentials. For more details on M3GNet,
please refer to::

    Chen, C., Ong, S.P. _A universal graph deep learning interatomic potential for the periodic table._ Nature
    Computational Science, 2023, 2, 718-728. DOI: 10.1038/s43588-022-00349-3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute_pyg import (
    compute_pair_vector_and_distance_pyg,
    compute_theta_and_phi_pyg,
    create_line_graph_pyg,
)
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
    GatedMLP,
    SphericalBesselWithHarmonics,
)
from matgl.layers._embedding_pyg import EmbeddingBlock
from matgl.layers._graph_convolution_pyg import M3GNetBlock
from matgl.layers._readout_pyg import (
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.layers._three_body_pyg import ThreeBodyInteractions
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


class M3GNet(MatGLModel):
    """The main M3GNet model for PyG."""

    __version__ = 2

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        max_n: int = 3,
        max_l: int = 3,
        nblocks: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",
        is_intensive: bool = True,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        use_smooth: bool = False,
        use_phi: bool = False,
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        include_state: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        dropout: float | None = None,
        **kwargs,
    ):
        """
        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            dim_node_embedding (int): Number of embedded atomic features
            dim_edge_embedding (int): Number of edge features
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            ntypes_state (int): Number of state labels
            max_n (int): Number of radial basis expansion
            max_l (int): Number of angular expansion
            nblocks (int): Number of convolution blocks
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            is_intensive (bool): Whether the prediction is intensive
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            cutoff (float): Cutoff radius of the graph
            threebody_cutoff (float): Cutoff radius for 3 body interaction
            units (int): Number of neurons in each MLP layer
            ntargets (int): Number of target properties
            use_smooth (bool): Whether using smooth Bessel functions
            use_phi (bool): Whether using phi angle
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            include_state (bool): Whether to include states features
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            dropout (float): Dropout probability to apply in graph layers during training
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS

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
            [
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            ]
        )

        dim_state_feats = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            [
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            ]
        )
        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats,
                    n_iters=niters_set2set,
                    n_layers=nlayers_set2set,
                    field=field,
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type: ignore[assignment]
                readout_feats = units + dim_state_feats if include_state else units  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats + dim_state_feats if include_state else input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self,
        g: Data,
        state_attr: torch.Tensor | None = None,
        l_g: Data | None = None,
        return_all_layer_output: bool = False,
    ):
        """Performs message passing and updates node representations.

        Args:
            g : PyG Data for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            l_g : PyG Data for a batch of line graphs.
            return_all_layer_output: Whether to return outputs of all M3GNet layers. By default, only the final layer
                output is returned.

        Returns:
            output: Output property for a batch of graphs.
        """
        node_types = g.node_type
        bond_vec, bond_dist = compute_pair_vector_and_distance_pyg(g)
        g.bond_vec = bond_vec
        g.bond_dist = bond_dist

        expanded_dists = self.bond_expansion(g.bond_dist)
        if l_g is None:
            l_g = create_line_graph_pyg(g, self.threebody_cutoff)
        # Compute theta and phi for line graph
        if l_g.edge_index.size(1) > 0:
            cos_theta, phi, triple_bond_lengths = compute_theta_and_phi_pyg(g, l_g)
        else:
            # Empty line graph - create empty tensors
            cos_theta = torch.empty(0, device=g.bond_dist.device)
            phi = torch.empty(0, device=g.bond_dist.device)
            triple_bond_lengths = torch.empty(0, device=g.bond_dist.device)

        # Store as edge attributes for compatibility with SphericalBesselWithHarmonics
        # Create a simple wrapper to make PyG Data compatible with DGL-style access
        class LineGraphWrapper:
            def __init__(self, lg, cos_theta, phi, triple_bond_lengths):
                self.lg = lg
                self.edata = {
                    "cos_theta": cos_theta,
                    "phi": phi,
                    "triple_bond_lengths": triple_bond_lengths,
                }

        l_g_wrapped = LineGraphWrapper(l_g, cos_theta, phi, triple_bond_lengths)
        g.rbf = expanded_dists
        three_body_basis = self.basis_expansion(l_g_wrapped)
        three_body_cutoff = polynomial_cutoff(g.bond_dist, self.threebody_cutoff)
        node_feat, edge_feat, state_feat = self.embedding(node_types, g.rbf, state_attr)
        fea_dict = {
            "bond_expansion": expanded_dists,
            "three_body_basis": three_body_basis,
            "embedding": {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            },
        }
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
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }
        g.node_feat = node_feat
        g.edge_feat = edge_feat
        if self.is_intensive:
            field_vec = self.readout(g)
            if self.include_state and state_feat is not None:
                readout_vec = torch.hstack([field_vec, state_feat])  # type: ignore
            else:
                readout_vec = field_vec
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            g.node_feat = node_feat
            atomic_properties = self.final_layer(g)
            if isinstance(g, Batch) and hasattr(g, "batch") and g.batch is not None:
                if atomic_properties.shape == (1, 1):
                    atomic_properties = atomic_properties.squeeze(-1)
                else:
                    atomic_properties = atomic_properties.squeeze()
                # Ensure batch is long dtype for scatter_add
                batch = g.batch.to(torch.long)
                output = scatter_add(atomic_properties, batch, dim_size=g.num_graphs)
            else:
                output = torch.sum(atomic_properties, dim=0, keepdim=True).squeeze()
            fea_dict["readout"] = atomic_properties
        fea_dict["final"] = output
        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to featurize or predict properties of a structure with M3GNet model.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes.
            graph_converter: Object that implements a get_graph_from_structure.
            output_layers: List of names for the layer of GNN as output. Choose from "bond_expansion", "embedding",
                "three_body_basis", "gc_1", "gc_2", "gc_3", "readout", and "final". By default, all M3GNet layer
                outputs are returned. Ignored if `return_features` is False.
            return_features (bool): If True, return specified layer outputs. If False, only return final output.

        Returns:
            output (dict or torch.tensor): M3GNet intermediate and final layer outputs for a structure, or final
                predicted property if `return_features` is False.
        """
        allowed_output_layers = [
            "bond_expansion",
            "embedding",
            "three_body_basis",
            "readout",
            "final",
        ] + [f"gc_{i + 1}" for i in range(self.n_blocks)]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed_output_layers}.")

        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]

        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)

        if not return_features:
            return model_output["final"].detach()

        return {k: v for k, v in model_output.items() if k in output_layers}
