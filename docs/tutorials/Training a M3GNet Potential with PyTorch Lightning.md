---
layout: default
title: tutorials/Training a M3GNet Potential with PyTorch Lightning.md
nav_exclude: true
---
# Introduction

This notebook demonstrates how to fit a M3GNet potential using PyTorch Lightning with MatGL.


```python
from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
from dgl.data.utils import split_dataset
from pymatgen.ext.matproj import MPRester
from pytorch_lightning.loggers import CSVLogger

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import M3GNetDataset, MGLDataLoader, collate_fn_efs
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

For the purposes of demonstration, we will download all Si-O compounds in the Materials Project via the MPRester. The forces and stresses are set to zero, though in a real context, these would be non-zero and obtained from DFT calculations.


```python
mpr = MPRester()

entries = mpr.get_entries_in_chemsys(["Si", "O"])
structures = [e.structure for e in entries]
energies = [e.energy for e in entries]
forces = [np.zeros((len(s), 3)).tolist() for s in structures]
stresses = [np.zeros((3, 3)).tolist() for s in structures]

print(f"{len(structures)} downloaded from MP.")
```


    Retrieving ThermoDoc documents:   0%|          | 0/407 [00:00<?, ?it/s]


    407 downloaded from MP.


We will first setup the M3GNet model and the LightningModule.


```python
element_types = get_element_list(structures)
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = M3GNetDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    energies=energies,
    forces=forces,
    stresses=stresses,
)
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_efs,
    batch_size=2,
    num_workers=1,
)
model = M3GNet(
    element_types=element_types,
    is_intensive=False,
)
lit_module = PotentialLightningModule(model=model)
```

    100%|███████████████████████████████████████████████████████████████████████████████████████| 407/407 [00:02<00:00, 202.56it/s]


Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Here, the max_epochs is set to 2 just for demonstration purposes. In a real fitting, this would be a much larger number. Also, the `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training. 


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_training")
trainer = pl.Trainer(max_epochs=2, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    
      | Name  | Type              | Params
    --------------------------------------------
    0 | model | Potential         | 282 K 
    1 | mae   | MeanAbsoluteError | 0     
    2 | rmse  | MeanSquaredError  | 0     
    --------------------------------------------
    282 K     Trainable params
    0         Non-trainable params
    282 K     Total params
    1.130     Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=2` reached.



```python
# This code just performs cleanup for this notebook.

for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

shutil.rmtree("lightning_logs")
```
