"""ASE interface for MatGL."""

from __future__ import annotations

import matgl

if matgl.config.BACKEND == "DGL":
    from ._ase_dgl import OPTIMIZERS, MolecularDynamics, PESCalculator, Relaxer, TrajectoryObserver
else:
    from ._ase_pyg import OPTIMIZERS, MolecularDynamics, PESCalculator, Relaxer, TrajectoryObserver  # type: ignore[assignment]

__all__ = ["OPTIMIZERS", "MolecularDynamics", "PESCalculator", "Relaxer", "TrajectoryObserver"]
