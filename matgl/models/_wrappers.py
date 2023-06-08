"""
Implementations of pseudomodels that wraps other models.
"""
from __future__ import annotations

import torch
from pymatgen.core import Structure
from torch import nn

from matgl.data.transformer import Transformer
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.graph.converters import GraphConverter
from matgl.utils.io import IOMixIn


class TransformedTargetModel(nn.Module, IOMixIn):
    """
    A model where the target is first transformed prior to training and the reverse transformation is performed for
    predictions. This is modelled after scikit-learn's TransformedTargetRegressor
    """

    def __init__(self, model: nn.Module, target_transformer: Transformer):
        """
        Args:
            model: Input model
            target_transformer: Transformer for target.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.transformer = target_transformer

    def forward(self, *args, **kwargs):
        r"""
        Args:
            *args: Passthrough to parent model.forward method.
            **kwargs: Passthrough to parent model.forward method.

        Returns:
            Inverse transformed output.
        """
        output = self.model.forward(*args, **kwargs)
        return self.transformer.inverse_transform(output)

    def __repr__(self):
        return f"TransformedTargetModel:\nModel: {self.model.__repr()}\nTransformer: {self.transformer.__repr__()}"

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
            data_mean: Mean of the data. Used when the original data has been scaled.
            data_std: Std of the data. Used when the original data has been scaled.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            graph_converter = Structure2Graph(element_types=self.model.element_types, cutoff=self.model.cutoff)
        g, state_feats_default = graph_converter.get_graph(structure)
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["edge_attr"] = self.model.bond_expansion(bond_dist)
        return self(g, g.edata["edge_attr"], g.ndata["node_type"], state_feats).detach()
