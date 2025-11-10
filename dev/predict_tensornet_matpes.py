"""Example script showing how to load the TensorNet MatPES PBE v2025.1 potential and
predict the energy of a structure.

Usage
-----
python examples/predict_tensornet_matpes.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.util.testing import PymatgenTest


def main() -> None:
    """Load the pretrained TensorNet model and predict the energy of an example structure."""
    lattice = Lattice.cubic(3.62)
    structure = Structure.from_spacegroup(
        "Fm-3m",
        lattice,
        ["Cu"],
        [[0, 0, 0]],
    )

    map_location = torch.device("cpu")
    model_path = Path("pretrained_models/TensorNetPYG-MatPES-PBE-v2025.1-PES")
    state = torch.load(model_path / "state.pt", map_location=map_location, weights_only=False)
    d = torch.load(model_path / "model.pt", map_location=map_location, weights_only=False)
    init_args = d["model"]["init_args"]
    from matgl.models._tensornet_pyg import TensorNet

    model_pyg = TensorNet(**init_args)
    model_pyg.load_state_dict(state, strict=False)  # type: ignore

    model_path = Path("pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES")
    state = torch.load(model_path / "state.pt", map_location=map_location, weights_only=False)
    d = torch.load(model_path / "model.pt", map_location=map_location, weights_only=False)
    init_args = d["model"]["init_args"]

    from matgl.models._tensornet_dgl import TensorNet

    model_dgl = TensorNet(**init_args)
    model_dgl.load_state_dict(state, strict=False)  # type: ignore

    for f in ["Li2O", "Si", "LiFePO4", "CsCl", "SiO2"]:
        structure = PymatgenTest.get_structure(f)
        energy = model_pyg.predict_structure(structure)
        print(f"Predicted energy PYG {f}: {float(energy):.6f} eV")
        energy = model_dgl.predict_structure(structure)
        print(f"Predicted energy DGL {f}: {float(energy):.6f} eV")


if __name__ == "__main__":
    main()
