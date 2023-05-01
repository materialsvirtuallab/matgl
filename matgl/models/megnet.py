"""
Implementation of MEGNet model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dgl
import torch
import torch.nn as nn
from dgl.nn import Set2Set
from pymatgen.core import Structure

from matgl.config import PRETRAINED_MODELS_PATH
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.graph.converters import Pmg2Graph
from matgl.layers.activations import SoftExponential, SoftPlus2
from matgl.layers.bond_expansion import BondExpansion
from matgl.layers.core import MLP, EdgeSet2Set
from matgl.layers.graph_conv import MEGNetBlock

logger = logging.getLogger(__file__)

DEFAULT_ELEMENT_TYPES = (
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
)


class MEGNet(nn.Module):
    """
    DGL implementation of MEGNet.
    """

    def __init__(
        self,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        attr_embedding_dim: int,
        num_blocks: int,
        hiddens: list[int],
        conv_hiddens: list[int],
        s2s_num_layers: int,
        s2s_num_iters: int,
        output_hiddens: list[int],
        act: str = "swish",
        is_classification: bool = True,
        node_embed: nn.Module | None = None,
        edge_embed: nn.Module | None = None,
        attr_embed: nn.Module | None = None,
        include_states: bool = False,
        dropout: float | None = None,
        graph_transformations: list | None = None,
        element_types: tuple[str] | None = None,
        data_mean: torch.tensor | None = None,
        data_std: torch.tensor | None = None,
        graph_converter: Pmg2Graph | None = None,
        bond_expansion: BondExpansion | None = None,
        **kwargs,
    ) -> None:
        """
        Construct a MEGNet model.

        Args:
            node_embedding_dim: Dimension of node embedding.
            edge_embedding_dim: Dimension of edge embedding.
            attr_embedding_dim: Dimension of attr (global state) embedding.
            num_blocks: Number of blocks.
            hiddens:
            conv_hiddens:
            s2s_num_layers:
            s2s_num_iters:
            output_hiddens:
            act:
            is_classification:
            node_embed:
            edge_embed:
            attr_embed:
            include_states:
            dropout:
            graph_transformations:
            element_types:
            data_mean:
            data_std:
            graph_converter: Perform a graph transformation, e.g., incorporate three-body interactions, prior to
                performing the GCL updates.
            bond_expansion:
            **kwargs:
        """
        # Store MEGNet model args for loading trained model
        self.model_args = {k: v for k, v in locals().items() if k not in ["self", "__class__", "kwargs"]}
        self.model_args.update(kwargs)

        super().__init__()

        if element_types is not None:
            self.element_types = element_types
        if graph_converter is not None:
            self.graph_converter = graph_converter
        if bond_expansion is not None:
            self.bond_expansion = bond_expansion
        if data_mean is not None:
            self.data_mean = data_mean
        if data_std is not None:
            self.data_std = data_std

        self.edge_embed = edge_embed if edge_embed else nn.Identity()
        self.node_embed = node_embed if node_embed else nn.Identity()
        self.attr_embed = attr_embed if attr_embed else nn.Identity()

        node_dims = [node_embedding_dim, *hiddens]
        edge_dims = [edge_embedding_dim, *hiddens]
        attr_dims = [attr_embedding_dim, *hiddens]

        if act == "swish":
            activation = nn.SiLU()  # type: ignore
        elif act == "sigmoid":
            activation = nn.Sigmoid()  # type: ignore
        elif act == "tanh":
            activation = nn.Tanh()  # type: ignore
        elif act == "softplus2":
            activation = SoftPlus2()  # type: ignore
        elif act == "softexp":
            activation = SoftExponential()  # type: ignore
        else:
            raise Exception("Undefined activation type, please try using swish, sigmoid, tanh, softplus2, softexp")

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.attr_encoder = MLP(attr_dims, activation, activate_last=True)

        blocks_in_dim = hiddens[-1]
        block_out_dim = conv_hiddens[-1]
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
            blocks.append(MEGNetBlock(dims=[block_out_dim, *hiddens], **block_args))  # type: ignore
        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": s2s_num_iters, "n_layers": s2s_num_layers}
        self.edge_s2s = EdgeSet2Set(block_out_dim, **s2s_kwargs)
        self.node_s2s = Set2Set(block_out_dim, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            dims=[2 * 2 * block_out_dim + block_out_dim, *output_hiddens, 1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None
        # TODO(marcel): should this be an 1D dropout

        self.is_classification = is_classification
        self.graph_transformations = graph_transformations or [nn.Identity()] * num_blocks
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
    def from_dir(cls, path: str | Path, **kwargs):
        """
        build a MEGNet from a saved directory
        """
        path = Path(path)
        file_name = path / "megnet.pt"
        if torch.cuda.is_available() is False:
            state = torch.load(file_name, map_location=torch.device("cpu"))
        else:
            state = torch.load(file_name)
        model = MEGNet.from_dict(state["model"], strict=False, **kwargs)
        return model

    @classmethod
    def load(cls, model_dir: str | Path) -> MEGNet:
        """
        Load the model weights from pre-trained model (megnet.pt)
        Args:
            model_dir (str): directory for saved model.

        Returns: MEGNet object.
        """
        if (PRETRAINED_MODELS_PATH / model_dir).exists():
            return cls.from_dir(PRETRAINED_MODELS_PATH / model_dir)
        model_dir = Path(model_dir)
        try:
            return cls.from_dir(model_dir)
        except FileNotFoundError:
            raise ValueError(f"{model_dir} not found in available pretrained_models {PRETRAINED_MODELS_PATH}.")

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
        edge_feat = self.edge_encoder(self.edge_embed(edge_feat))
        node_feat = self.node_encoder(self.node_embed(node_feat))
        if self.include_states:
            graph_attr = self.attr_embed(graph_attr)
        else:
            graph_attr = self.attr_encoder(self.attr_embed(graph_attr))

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


class MEGNetTemp(nn.Module):
    """
    DGL implementation of MEGNet.
    """

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_attr_embedding: int = 2,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        layer_node_embedding: nn.Module | None = None,
        layer_edge_embedding: nn.Module | None = None,
        layer_attr_embedding: nn.Module | None = None,
        include_states: bool = False,
        dropout: float | None = None,
        graph_transformations: list | None = None,
        element_types: tuple[str, ...] | None = None,
        data_mean: torch.tensor | None = None,
        data_std: torch.tensor | None = None,
        graph_converter: Pmg2Graph | None = None,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ) -> None:
        """
        Construct a MEGNet model. Useful defaults for all arguments have been specified based on MEGNet formation energy
        model.

        Args:
            dim_node_embedding: Dimension of node embedding.
            dim_edge_embedding: Dimension of edge embedding.
            dim_attr_embedding: Dimension of attr (global state) embedding.
            nblocks: Number of blocks.
            hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
            hidden_layer_sizes_conv: Architecture of dense layers for message and update functions
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iteratons in Set2Set layer
            hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
            activation_types: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            layer_node_embedding: Architecture of embedding layer for node attributes
            layer_edge_embedding: Architecture of embedding layer for edge attributes
            layer_attr_embedding: Architecture of embedding layer for state attributes
            include_states: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution
            graph_transformations: Perform a graph transformation, e.g., incorporate three-body interactions, prior to
                performing the GCL updates.
            element_types: Elements included in the training set
            data_mean: Mean of target properties in the training set. Defaults to 0.
            data_std: Standard deviation of target properties in the training set. Defaults to 1.
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

        self.element_types = element_types or DEFAULT_ELEMENT_TYPES
        self.graph_converter = graph_converter or Pmg2Graph(element_types=self.element_types, cutoff=cutoff)
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=dim_edge_embedding, width=gauss_width
        )
        self.data_mean = data_mean or torch.zeros(1)
        self.data_std = data_std or torch.ones(1)

        self.edge_embedding_layer = layer_edge_embedding if layer_edge_embedding else nn.Identity()
        if layer_node_embedding is None:
            self.node_embedding_layer = nn.Embedding(len(self.element_types), dim_node_embedding)
        else:
            self.node_embedding_layer = nn.Identity()
        self.node_embedding_layer = layer_node_embedding or nn.Identity()
        self.attr_embedding_layer = layer_attr_embedding or nn.Identity()

        node_dims = [dim_node_embedding, *hidden_layer_sizes_conv]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_conv]
        attr_dims = [dim_attr_embedding, *hidden_layer_sizes_conv]

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
        self.include_states = include_states

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: torch.Tensor,
        node_feat: torch.Tensor,
        graph_attr: torch.Tensor,
    ):
        """
        Forward pass of MEGnet. Executes all blocks.

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

        for gt, block in zip(graph_transformations, self.blocks):
            output = block(gt(graph), edge_feat, node_feat, graph_attr)
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
        attrs = attrs or torch.tensor(attrs_default)

        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.bond_expansion(bond_dist)

        output = self.data_std * self(g, g.edata["edge_attr"], g.ndata["node_type"], attrs) + self.data_mean

        return output.detach()

    def as_dict(self):
        out = {"state_dict": self.state_dict(), "model_args": self.model_args}
        return out

    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> MEGNet:
        r"""
        Load MEGNet model from dict.

        Args:
            d: dict
            **kwargs: Additional kwargs.

        Returns:
            MEGNet model
        """
        model = MEGNet(**d["model_args"])
        model.load_state_dict(d["state_dict"], **kwargs)
        return model

    @classmethod
    def load(cls, model_dir: str | Path, **kwargs) -> MEGNet:
        """
        Load a MEGNet model from a directory or the name of a pre-trained model.

        Args:
            model_dir (str): String or Path object for location of saved model. It can also be one of the pre-trained
                models listed in matgl.config.PRETRAINED_MODELS_PATH.
            **kwargs: Additional kwargs.

        Returns: MEGNet object.
        """
        model_dir = Path(model_dir)
        if (PRETRAINED_MODELS_PATH / model_dir).exists():
            path = PRETRAINED_MODELS_PATH / model_dir
        try:
            if torch.cuda.is_available() is False:
                state = torch.load(path / "megnet.pt", map_location=torch.device("cpu"))
            else:
                state = torch.load(path / "megnet.pt")
            model = MEGNet.from_dict(state["model"], strict=False, **kwargs)
            return model
        except FileNotFoundError:
            raise ValueError(
                f"{model_dir} does not appear to be a valid model. Provide a valid path or use one of "
                f"the following pretrained models in {list(PRETRAINED_MODELS_PATH.iterdir())}."
            )
