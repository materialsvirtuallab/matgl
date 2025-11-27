"""Tools to convert materials representations from Pymatgen and other codes to Graphs."""

from __future__ import annotations

from matgl.config import BACKEND

if BACKEND == "DGL":
    from ._converters_dgl import GraphConverter
else:
    from ._converters_pyg import GraphConverter  # type: ignore[assignment]

__all__ = [
    "GraphConverter",
]
