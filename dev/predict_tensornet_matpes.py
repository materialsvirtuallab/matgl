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

    model = TensorNet(**init_args)
    model.load_state_dict(state, strict=False)  # type: ignore

    energy = model.predict_structure(structure)
    print(f"Predicted energy PYG: {float(energy):.6f} eV")

    model_path = Path("pretrained_models/TensorNet-MatPES-PBE-v2025.1-PES")
    state = torch.load(model_path / "state.pt", map_location=map_location, weights_only=False)
    d = torch.load(model_path / "model.pt", map_location=map_location, weights_only=False)
    init_args = d["model"]["init_args"]

    from matgl.models._tensornet_dgl import TensorNet

    model = TensorNet(**init_args)
    model.load_state_dict(state, strict=False)  # type: ignore

    energy = model.predict_structure(structure)
    print(f"Predicted energy DGL: {float(energy):.6f} eV")


if __name__ == "__main__":
    main()
