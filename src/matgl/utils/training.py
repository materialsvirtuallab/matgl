"""Utils for training MatGL models.

This module provides a unified interface for training MatGL models, automatically
selecting the appropriate backend implementation (DGL or PyG) based on the configured backend.
"""

from __future__ import annotations

from matgl.config import BACKEND

if BACKEND == "DGL":
    from matgl.utils._training_dgl import (
        MatglLightningModuleMixin,
        ModelLightningModule,
        PotentialLightningModule,
        xavier_init,
    )
else:
    from matgl.utils._training_pyg import (  # type: ignore[assignment]
        MatglLightningModuleMixin,
        ModelLightningModule,
        PotentialLightningModule,
        xavier_init,
    )

__all__ = [
    "MatglLightningModuleMixin",
    "ModelLightningModule",
    "PotentialLightningModule",
    "xavier_init",
]
