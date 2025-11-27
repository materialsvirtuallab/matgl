"""Integration with pymatgen for graph construction."""

from __future__ import annotations

import matgl

if matgl.config.BACKEND == "DGL":
    from ._pymatgen_dgl import Molecule2Graph, Structure2Graph, get_element_list
else:
    from ._pymatgen_pyg import Molecule2Graph, Structure2Graph, get_element_list  # type: ignore[assignment]


__all__ = ["Molecule2Graph", "Structure2Graph", "get_element_list"]
