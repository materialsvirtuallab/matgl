"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from matgl.config import BACKEND

if BACKEND == "DGL":
    try:
        import dgl  # noqa
    except ImportError as err:
        raise ImportError("Please install DGL to use this backend.") from err
    from ._pes_dgl import Potential
else:
    try:
        import torch_geometric  # noqa
    except ImportError as err:
        raise ImportError("Please install torch_geometric to use this backend.") from err
    from ._pes_pyg import Potential  # noqa  # type: ignore[assignment]
