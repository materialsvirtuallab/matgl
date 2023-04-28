"""
Implementation of MEGNet model.
"""
from __future__ import annotations

import logging
import os

import dgl
import torch
import torch.nn as nn
from dgl.nn import Set2Set
from pymatgen.core import Structure

from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.graph.converters import Pmg2Graph
from matgl.layers.activations import SoftExponential, SoftPlus2
from matgl.layers.bond_expansion import BondExpansion
from matgl.layers.core import MLP, EdgeSet2Set
from matgl.layers.graph_conv import MEGNetBlock

logger = logging.getLogger(__file__)
CWD = os.path.dirname(os.path.abspath(__file__))
# defined the default element types
ELEMENT_TYPES = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]

# These define paths to models that are already pre-trained and ready to use.
PRETRAINED_MODEL_PATHS = {
    "MP-2018.6.1-Eform": os.path.join(CWD, "..", "..", "pretrained_models", "MP-2018.6.1-Eform"),
    "MP-2019.4.1-BandGap-mfi": os.path.join(CWD, "..", "..", "pretrained_models", "MP-2019.4.1-BandGap-mfi"),
}


class TempleMEGNet(nn.Module):
    """
    DGL implementation of MEGNet.
    """

    def __init__(
        self,
        node_embedding_dim: int = 16,
        edge_embedding_dim: int = 100,
        attr_embedding_dim: int = 2,
        nblocks: int = 3,
        hidden_layer: list[int] = [64, 32],
        conv_hidden_layer: list[int] = [64, 64, 32],
        s2s_nlayers: int = 1,
        s2s_niters: int = 2,
        output_hidden_layer: list[int] = [32, 16],
        activation_type: str = "softplus2",
        is_classification: bool = False,
        node_embedding_layer: nn.Module | None = None,
        edge_embedding_layer: nn.Module | None = None,
        attr_embedding_layer: nn.Module | None = None,
        include_states: bool = False,
        dropout: float | None = None,
        graph_transformations: list | None = None,
        element_types: tuple[str] | None = None,
        data_mean: torch.tensor = torch.zeros(1),
        data_std: torch.tensor = torch.ones(1),
        graph_converter: Pmg2Graph | None = None,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ) -> None:
        """
        Construct a MEGNet model.

        Args:
            node_embedding_dim: Dimension of node embedding.
            edge_embedding_dim: Dimension of edge embedding.
            attr_embedding_dim: Dimension of attr (global state) embedding.
            nblocks: Number of blocks.
            hidden_layer: Architecture of dense layers before the graph convolution
            conv_hidden_layer: Architecture of dense layers for message and update functions
            s2s_nlayers: Number of layers in Set2Set layer
            s2s_niters: Number of iteratons in Set2Set layer
            output_hidden_layer: Architecture of dense layers for concatenated features after graph convolution
            activation_types: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            node_embedding_layer: Architecture of embedding layer for node attributes
            edge_embedding_layer: Architecture of embedding layer for edge attributes
            attr_embedding_layer: Architecture of embedding layer for state attributes
            include_states: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to a Bernoulli distribution
            graph_transformations: Perform a graph transformation, e.g., incorporate three-body interactions, prior to
                performing the GCL updates.
            element_types: Elements included in the training set
            data_mean: Mean of target properties in the training set
            data_std: Standard deviation of target properties in the training set
            graph_converter: Pmg2Graph converter
            bond_expansion: Gaussian expansion for edge attributes
            cutoff: cutoff for forming bonds
            gauss_width: width of Gaussian function for bond expansion
            **kwargs:
        """
        # Store MEGNet model args for loading trained model
        self.model_args = {k: v for k, v in locals().items() if k not in ["self", "__class__", "kwargs"]}
        self.model_args.update(kwargs)

        super().__init__()

        if element_types is not None:
            self.element_types = element_types
        else:
            self.element_types = ELEMENT_TYPES
        if graph_converter is not None:
            self.graph_converter = graph_converter
        else:
            self.graph_converter = Pmg2Graph(element_types=self.element_types, cutoff=cutoff)
        if bond_expansion is not None:
            self.bond_expansion = bond_expansion
        else:
            self.bond_expansion = BondExpansion(
                rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=edge_embedding_dim, width=gauss_width
            )
        if data_mean is not None:
            self.data_mean = data_mean
        if data_std is not None:
            self.data_std = data_std

        self.edge_embedding_layer = edge_embedding_layer if edge_embedding_layer else nn.Identity()
        if node_embedding_layer is None:
            self.node_embedding_layer = nn.Embedding(len(self.element_types), node_embedding_dim)
        else:
            self.node_embedding_layer = nn.Identity()
        self.node_embedding_layer = node_embedding_layer if node_embedding_layer else nn.Identity()
        self.attr_embedding_layer = attr_embedding_layer if attr_embedding_layer else nn.Identity()

        node_dims = [node_embedding_dim, *hiddens]
        edge_dims = [edge_embedding_dim, *hiddens]
        attr_dims = [attr_embedding_dim, *hiddens]

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

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.attr_encoder = MLP(attr_dims, activation, activate_last=True)

        blocks_in_dim = hidden_layer[-1]
        block_out_dim = conv_hidden_layer[-1]
        block_args = {
            "conv_hiddens": conv_hiddens,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        blocks = []

        # first block
        blocks.append(MEGNetBlock(dims=[blocks_in_dim], **block_args))  # type: ignore
        # other blocks
        for _ in range(num_blocks - 1):
            blocks.append(MEGNetBlock(dims=[block_out_dim, *hidden_layer], **block_args))  # type: ignore
        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": s2s_niters, "n_layers": s2s_nlayers}
        self.edge_s2s = EdgeSet2Set(block_out_dim, **s2s_kwargs)
        self.node_s2s = Set2Set(block_out_dim, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * block_out_dim + block_out_dim, *output_hidden_layer, 1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout

        self.is_classification = is_classification
        self.graph_transformations = graph_transformations or [nn.Identity()] * nblocks
        self.include_states = include_states

    def as_dict(self):
        out = {"state_dict": self.state_dict(), "model_args": self.model_args}
        return out

    @classmethod
    def from_dict(cls, dict, **kwargs):
        """
        build a MEGNet from a saved dictionary
        """
        model = MEGNet(**dict["model_args"])
        model.load_state_dict(dict["state_dict"], **kwargs)
        return model

    @classmethod
    def from_dir(cls, path, **kwargs):
        """
        build a MEGNet from a saved directory
        """
        file_name = os.path.join(path, "megnet.pt")
        if torch.cuda.is_available() is False:
            state = torch.load(file_name, map_location=torch.device("cpu"))
        else:
            state = torch.load(file_name)
        model = MEGNet.from_dict(state["model"], strict=False, **kwargs)
        return model

    @classmethod
    def load(cls, model_dir: str) -> MEGNet:
        """
        Load the model weights from pre-trained model (megnet.pt)
        Args:
            model_dir (str): directory for saved model. Defaults to "MP-2018.6.1-Eform".

        Returns: MEGNet object.
        """
        if model_dir in PRETRAINED_MODEL_PATHS:
            return cls.from_dir(PRETRAINED_MODEL_PATHS[model_dir])

        if os.path.isdir(model_dir) and "megnet.pt" in os.listdir(model_dir):
            return cls.from_dir(model_dir)

        raise ValueError(f"{model_dir} not found in available pretrained_models {list(PRETRAINED_MODEL_PATHS.keys())}.")

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ):
        """
        TODO: Add docs.
        :param graph: Input graph
        :param edge_feat: Edge features
        :param node_feat: Node features
        :param graph_attr: Graph attributes / state features.
        :return: Prediction
        """
        graph_transformations = self.graph_transformations
        edge_feat = self.edge_encoder(self.edge_embedding_layer(edge_feat))
        node_feat = self.node_encoder(self.node_embedding_layer(node_feat))
        if self.include_states:
            graph_attr = self.attr_embedding_layer(graph_attr)
        else:
            graph_attr = self.attr_encoder(self.attr_embedding_layer(graph_attr))

        for i, block in enumerate(self.blocks):
            graph = graph_transformations[i](graph)
            output = block(graph, edge_feat, node_feat, graph_attr)
            edge_feat, node_feat, graph_attr = output

        node_vec = self.node_s2s(graph, node_feat)
        edge_vec = self.edge_s2s(graph, edge_feat)

        node_vec = torch.squeeze(node_vec)
        edge_vec = torch.squeeze(edge_vec)
        graph_attr = torch.squeeze(graph_attr)

        vec = torch.hstack([node_vec, edge_vec, graph_attr])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return output

    def predict_structure(self, structure: Structure, attrs: torch.tensor | None = None):
        """
        Convenience method to directly predict property from structure.
        Args:
            structure (Structure): Pymatgen structure
            attrs (torch.tensor): graph attributes
        Returns:
            output (torch.tensor): output property
        """
        g, attrs_default = self.graph_converter.get_graph_from_structure(structure)
        if attrs is None:
            attrs = torch.tensor(attrs_default)

        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)

        output = self.data_std * self(g, g.edata["edge_attr"], g.ndata["node_type"], attrs) + self.data_mean

        return output.detach()
