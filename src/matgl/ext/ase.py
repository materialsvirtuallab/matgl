"""ASE interface for MatGL."""

from __future__ import annotations

import matgl

if matgl.config.BACKEND == "DGL":
    from ._ase_dgl import OPTIMIZERS, MolecularDynamics, PESCalculator, Relaxer
else:
    from ._ase_pyg import OPTIMIZERS, MolecularDynamics, PESCalculator, Relaxer  # type: ignore[assignment]

__all__ = ["OPTIMIZERS", "MolecularDynamics", "PESCalculator", "Relaxer"]
