"""Tools to construct a dataset of DGL graphs."""

from __future__ import annotations

from matgl.config import BACKEND

if BACKEND == "DGL":
    from ._data_dgl import MGLDataLoader, MGLDataset, collate_fn_graph, collate_fn_pes
else:
    from ._data_pyg import (  # type: ignore[assignment]
        MGLDataLoader,
        MGLDataset,
        collate_fn_graph,
        collate_fn_pes,
        split_dataset,
    )

__all__ = [
    "MGLDataLoader",
    "MGLDataset",
    "collate_fn_graph",
    "collate_fn_pes",
    "split_dataset",
]
