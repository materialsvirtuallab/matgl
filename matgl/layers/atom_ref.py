"""
atomic energy offset. Used for predicting extensive properties.
"""
from __future__ import annotations

import dgl
import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Molecule, Structure


class AtomRef(nn.Module):
    """
    Get total property offset for a system:
    """

    def __init__(
        self,
        property_offset: np.array,  # type: ignore
    ) -> None:
        """
        Parameters:
        -----------
        property_offset (np.array): a array of elemental property offset
        """
        super().__init__()
        self.property_offset = torch.tensor(property_offset)
        self.max_z = self.property_offset.size(dim=0)

    def get_feature_matrix(self, structs_or_graphs: list, element_list: tuple[str]) -> np.typing.NDArray:
        """
        Get the number of atoms for different elements in the structure

        Args:
        structs_or_graphs (list): a list of pymatgen Structure or dgl graph
        element_list: a dictionary containing element types in the training set

        Returns:
        features (np.array): a matrix (num_structures, num_elements)
        """
        n = len(structs_or_graphs)
        features = np.zeros(shape=(n, self.max_z))
        for i, s in enumerate(structs_or_graphs):
            if isinstance(s, (Structure, Molecule)):
                atomic_numbers = [element_list.index(site.specie.symbol) for site in s.sites]
            else:
                one_hot_vecs = s.ndata["attr"]
                atomic_numbers = ((one_hot_vecs == 1).nonzero(as_tuple=True)[0]).tolist()
            features[i] = np.bincount(atomic_numbers, minlength=self.max_z)
        return features

    def fit(self, structs_or_graphs: list, element_list: tuple[str], properties: np.typing.NDArray) -> bool:
        """
        Fit the elemental reference values for the properties

        Args:
        structs_or_graphs: pymatgen Structures or dgl graphs
        properties (np.ndarray): array of extensive properties

        Returns:
        """
        features = self.get_feature_matrix(structs_or_graphs, element_list)
        self.property_offset = np.linalg.pinv(features.T.dot(features)).dot(features.T.dot(properties))
        self.property_offset = torch.tensor(self.property_offset)
        return True

    def forward(self, g: dgl.DGLGraph, state_attr=None):
        """
        Get the total property offset for a system

        Args:
        g: a batch of dgl graphs
        state_attr: state label

        Returns:
        offset_per_graph:
        """
        if self.property_offset.ndim > 1:
            offset_batched_with_state = []
            for i in range(0, self.property_offset.size(dim=0)):
                property_offset_batched = self.property_offset[i].repeat(g.num_nodes(), 1)
                offset = property_offset_batched.to(g.device) * g.ndata["attr"]
                g.ndata["atomic_offset"] = torch.sum(offset, 1)
                offset_batched = dgl.readout_nodes(g, "atomic_offset")
                offset_batched_with_state.append(offset_batched)
            offset_batched_with_state = torch.stack(offset_batched_with_state)  # type: ignore
            return offset_batched_with_state[state_attr]
        else:
            property_offset_batched = self.property_offset.repeat(g.num_nodes(), 1)
            offset = property_offset_batched.to(g.device) * g.ndata["attr"]
            g.ndata["atomic_offset"] = torch.sum(offset, 1)
            offset_batched = dgl.readout_nodes(g, "atomic_offset")
            return offset_batched
