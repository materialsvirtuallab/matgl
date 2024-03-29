---
layout: default
title: Property Predictions using MEGNet or M3GNet Models.md
nav_exclude: true
---

# Introduction

This notebook demonstrates the use of pre-trained MEGNet and M3GNet models to predict properties.

Author: Tsz Wai Ko (Kenko)
Email: t1ko@ucsd.edu



```python
from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure

import matgl

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# MP Formation energy

The pre-trained models are based on the Materials Project mp.2018.6.1.json dataset. There are two models available - MEGNet and M3GNet.


We create the structure first. This is based on the relaxed structure obtained from the Materials Project. Alternatively, one can use the Materials Project API to obtain the structure.


```python
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
```

## Using the MEGNet-MP-2018.6.1-Eform model


```python
# Load the pre-trained MEGNet formation energy model.
model = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform):.3f} eV/atom.")
```

## Using the M3GNet-MP-2018.6.1-Eform model


```python
# Load the pre-trained M3GNet formation energy model
model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform):.3f} eV/atom.")
```

# MP Band gap

This is the multi-fidelity band gap model, discussed in Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. Learning Properties of Ordered and Disordered Materials from Multi-Fidelity Data. Nature Computational Science 2021, 1, 46–53. https://doi.org/10.1038/s43588-020-00002-x.




```python
model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
    graph_attrs = torch.tensor([i])
    bandgap = model.predict_structure(structure=struct, state_attr=graph_attrs)
    print(f"The predicted {method} band gap for CsCl is {float(bandgap):.3f} eV.")
```