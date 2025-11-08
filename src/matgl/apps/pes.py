"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from matgl.config import BACKEND

# ruff: noqa

if BACKEND == "DGL":
    from ._pes_dgl import Potential
else:
    from ._pes_pyg import Potential  # type: ignore[assignment]
