---
layout: default
title: Training a M3GNet Formation Energy Model with PyTorch Lightning.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to refit a MEGNet formation energy model using PyTorch Lightning with MatGL.


```python
from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn
from matgl.models import M3GNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# Dataset Preparation

We will download the original dataset used in the training of the MEGNet formation energy model (MP.2018.6.1) from figshare. To make it easier, we will also cache the data.


```python
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    if not os.path.exists("mp.2018.6.1.json"):
        f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
        with zipfile.ZipFile(f.local_path) as zf:
            zf.extractall(".")
    data = pd.read_json("mp.2018.6.1.json")
    structures = []
    mp_ids = []
    i = 0
    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
        i = i + 1
        if i > 1000:
            break
    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


structures, mp_ids, eform_per_atom = load_dataset()
```

For demonstration purposes, we are only going to select 100 structures from the entire set of structures to shorten the training time.


```python
structures = structures[:100]
eform_per_atom = eform_per_atom[:100]
```

Here, we set up the dataset.


```python
# get element types in the dataset
elem_list = get_element_list(structures)
# setup a graph converter
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into MEGNetDataset
mp_dataset = MGLDataset(
    threebody_cutoff=4.0, structures=structures, converter=converter, labels={"eform": eform_per_atom}
)
```

We will then split the dataset into training, validation and test data.


```python
train_data, val_data, test_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
my_collate_fn = partial(collate_fn, include_line_graph=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=2,
    num_workers=1,
)
```

# Model setup

In the next step, we setup the model and the ModelLightningModule. Here, we have initialized a MEGNet model from scratch. Alternatively, you can also load one of the pre-trained models for transfer learning, which may speed up the training.


```python
# setup the architecture of MEGNet model
model = M3GNet(
    element_types=elem_list,
    is_intensive=True,
    readout_type="set2set",
)
# setup the MEGNetTrainer
lit_module = ModelLightningModule(model=model)
```

# Training

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Note that the max_epochs is set at 20 to demonstrate the fitting on a laptop. A real fitting should use max_epochs > 100 and be run in parallel on GPU resources. For the formation energy, it should be around 2000. The `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training. You may also need to use `torch.set_default_device("cuda")` or `with torch.device("cuda")` to ensure all data are loaded onto the GPU for training.

We have also initialized the Pytorch Lightning Trainer with a `CSVLogger`, which provides a detailed log of the loss metrics at each epoch.


```python
logger = CSVLogger("logs", name="M3GNet_training")
trainer = pl.Trainer(max_epochs=20, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

# Visualizing the convergence

Finally, we can plot the convergence plot for the loss metrics. You can see that the MAE is already going down nicely with 20 epochs. Obviously, this is nowhere state of the art performance for the formation energies, but a longer training time should lead to results consistent with what was reported in the original MEGNet work.


```python
metrics = pd.read_csv("logs/M3GNet_training/version_0/metrics.csv")
metrics["train_MAE"].dropna().plot()
metrics["val_MAE"].dropna().plot()

_ = plt.legend()
```


```python
# This code just performs cleanup for this notebook.

for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

shutil.rmtree("logs")
```