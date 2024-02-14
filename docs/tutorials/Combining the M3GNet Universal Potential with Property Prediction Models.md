---
layout: default
title: Combining the M3GNet Universal Potential with Property Prediction Models.md
nav_exclude: true
---

# Introduction

There may be instances where you do not have access to a DFT relaxed structure. For instance, you may have a generated hypothetical structure or a structure obtained from an experimental source. In this notebook, we demonstrate how you can use the M3GNet universal potential to relax a crystal prior to property predictions.

This provides a pathway to "DFT-free" property predictions using ML models. It should be cautioned that this is not a substitute for DFT and errors can be expected. But it is sufficiently useful in some cases as a pre-screening tool for massive scale exploration of materials.


```python
from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.ext.ase import Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

For the purposes of demonstration, we will use the perovskite SrTiO3 (STO). We will create a STO with an arbitrary lattice parameter of 4.5 A.


```python
sto = Structure.from_spacegroup(
    "Pm-3m", Lattice.cubic(4.5), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
)
print(sto)
```

As a ground truth reference, we will also obtain the Materials Project DFT calculated SrTiO3 structure (mpid: mp-???) using pymatgen's interface to the Materials API.


```python
mpr = MPRester()
doc = mpr.summary.search(material_ids=["mp-5229"])[0]
sto_dft = doc.structure
sto_dft_bandgap = doc.band_gap
sto_dft_forme = doc.formation_energy_per_atom
```

# Relaxing the crystal


```python
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
```


```python
relaxer = Relaxer(potential=pot)
relax_results = relaxer.relax(sto, fmax=0.01)
relaxed_sto = relax_results["final_structure"]
print(relaxed_sto)
```

You can compare the lattice parameter with the DFT one from MP. Quite clearly, the M3GNet universal potential does a reasonably good job on relaxing STO.


```python
print(sto_dft)
```

# Formation energy prediction

To demonstrate the difference between making predictions with a unrelaxed vs a relaxed crystal, we will load the M3GNet formation energy model.


```python
# Load the pre-trained MEGNet formation energy model.
model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
eform_sto = model.predict_structure(sto)
eform_relaxed_sto = model.predict_structure(relaxed_sto)

print(f"The predicted formation energy for the unrelaxed SrTiO3 is {float(eform_sto):.3f} eV/atom.")
print(f"The predicted formation energy for the relaxed SrTiO3 is {float(eform_relaxed_sto):.3f} eV/atom.")
print(f"The Materials Project formation energy for DFT-relaxed SrTiO3 is {sto_dft_forme:.3f} eV/atom.")
```

The predicted formation energy from the M3GNet relaxed STO is in fairly good agreement with the DFT value.

# Band gap prediction

We will repeat the above exericse but for the band gap.


```python
model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
    graph_attrs = torch.tensor([i])
    bandgap_sto = model.predict_structure(structure=sto, state_attr=graph_attrs)
    bandgap_relaxed_sto = model.predict_structure(structure=relaxed_sto, state_attr=graph_attrs)

    print(f"{method} band gap")
    print(f"\tUnrelaxed STO = {float(bandgap_sto):.2f} eV.")
    print(f"\tRelaxed STO = {float(bandgap_relaxed_sto):.2f} eV.")
print(f"The PBE band gap for STO from Materials Project is {sto_dft_bandgap:.2f} eV.")
```

Again, you can see that using the unrelaxed SrTiO3 leads to large errors, predicting SrTiO3 to have very small band agps. Using the relaxed STO leads to predictions that are much closer to expectations. In particular, the predicted PBE band gap is quite close to the Materials Project PBE value. The experimental band gap is around 3.2 eV, which is reproduced very well by the GLLB-SC prediction!