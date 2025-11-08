"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from matgl.config import BACKEND

if BACKEND == "dgl" or BACKEND == "pyg":
    pass
