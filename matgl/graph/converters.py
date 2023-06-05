"""
Tools to convert materials representations from Pymatgen and other codes to DGLGraphs.
"""
from __future__ import annotations

import abc

import dgl


class GraphConverter(metaclass=abc.ABCMeta):
    """
    Abstract base class for converters from input structures/molecules to graphs.
    """

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[dgl.DGLGraph, list]:
        """

        Args:
            structure: Input structure or molecule.

        Returns:
            DGLGraph object, state_attr
        """
