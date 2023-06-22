"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dgl


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[dgl.DGLGraph, list]:
        """Args:
            structure: Input crystals or molecule.

        Returns:
            DGLGraph object, state_attr
        """
