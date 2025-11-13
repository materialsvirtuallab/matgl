"""Integration with pymatgen for graph construction."""

from __future__ import annotations

import matgl

if matgl.config.BACKEND == "DGL":
    from matgl.apps._pymatgen_dgl import Atom2Graph, Structure2Graph, get_element_list
else:
    from matgl.apps._pymatgen_pyg import Atom2Graph, Structure2Graph, get_element_list  # type: ignore[assignment]


__all__ = ["Atom2Graph", "Structure2Graph", "get_element_list"]
