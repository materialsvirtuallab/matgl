---
layout: default
title: Training a M3GNet Potential with PyTorch Lightning.md
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
import lightning as pl
from functools import partial
from dgl.data.utils import split_dataset
from mp_api.client import MPRester
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

For the purposes of demonstration, we will download all Si-O compounds in the Materials Project via the MPRester. The forces and stresses are set to zero, though in a real context, these would be non-zero and obtained from DFT calculations.


```python
# Obtain your API key here: https://next-gen.materialsproject.org/api
mpr = MPRester(api_key="YOUR_API_KEY")
entries = mpr.get_entries_in_chemsys(["Si", "O"])
structures = [e.structure for e in entries]
energies = [e.energy for e in entries]
forces = [np.zeros((len(s), 3)).tolist() for s in structures]
stresses = [np.zeros((3, 3)).tolist() for s in structures]
labels = {
    "energies": energies,
    "forces": forces,
    "stresses": stresses,
}

print(f"{len(structures)} downloaded from MP.")
```

We will first setup the M3GNet model and the LightningModule.


```python
element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels=labels,
    include_line_graph=True,
)
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
# if you are not intended to use stress for training, switch include_stress=False!
my_collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=2,
    num_workers=0,
)
model = M3GNet(
    element_types=element_types,
    is_intensive=False,
)
# if you are not intended to use stress for training, set stress_weight=0.0!
lit_module = PotentialLightningModule(model=model, include_line_graph=True, stress_weight=0.01)
```

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Here, the max_epochs is set to 2 just for demonstration purposes. In a real fitting, this would be a much larger number. Also, the `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training.


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_training")
# Inference mode = False is required for calculating forces, stress in test mode and prediction mode
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```


```python
# test the model, remember to set inference_mode=False in trainer (see above)
trainer.test(dataloaders=test_loader)
```


```python
# save trained model
model_export_path = "./trained_model/"
lit_module.model.save(model_export_path)

# load trained model
model = matgl.load_model(path=model_export_path)
```

## Finetuning a pre-trained M3GNet
In the previous cells, we demonstrated the process of training an M3GNet from scratch. Next, let's see how to perform additional training on an M3GNet that has already been trained using Materials Project data.


```python
# download a pre-trained M3GNet
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model
# obtain element energy offset
property_offset = m3gnet_nnp.element_refs.property_offset
# you should test whether including the original property_offset helps improve training and validation accuracy
lit_module_finetune = PotentialLightningModule(model=model_pretrained, element_refs=property_offset, lr=1e-4, include_line_graph=True)
```


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)
```


```python
# save trained model
model_save_path = "./finetuned_model/"
lit_module_finetune.model.save(model_save_path)
# load trained model
trained_model = matgl.load_model(path=model_save_path)
```


```python
# This code just performs cleanup for this notebook.

for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

shutil.rmtree("logs")
shutil.rmtree("trained_model")
shutil.rmtree("finetuned_model")
```