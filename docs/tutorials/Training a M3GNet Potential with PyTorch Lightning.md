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
import pytorch_lightning as pl
from dgl.data.utils import split_dataset
from mp_api.client import MPRester
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import M3GNetDataset, MGLDataLoader, collate_fn_efs
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule

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

    Retrieving ThermoDoc documents: 100%|██████████| 407/407 [00:00<00:00, 4962446.88it/s]


    407 downloaded from MP.


We will first setup the M3GNet model and the LightningModule.


```python
element_types = get_element_list(structures)
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = M3GNetDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels=labels,
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

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Here, the max_epochs is set to 2 just for demonstration purposes. In a real fitting, this would be a much larger number. Also, the `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training.


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_training")
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader, inference_mode=False)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

      | Name  | Type              | Params
    --------------------------------------------
    0 | mae   | MeanAbsoluteError | 0
    1 | rmse  | MeanSquaredError  | 0
    2 | model | Potential         | 282 K
    --------------------------------------------
    282 K     Trainable params
    0         Non-trainable params
    282 K     Total params
    1.130     Total estimated model params size (MB)


    Epoch 0: 100%|██████████| 163/163 [00:39<00:00,  4.16it/s, v_num=1, val_Total_Loss=0.815, val_Energy_MAE=0.674, val_Force_MAE=0.289, val_Stress_MAE=0.000, val_Site_Wise_MAE=0.000, val_Energy_RMSE=0.730, val_Force_RMSE=0.419, val_Stress_RMSE=0.000, val_Site_Wise_RMSE=0.000, train_Total_Loss=21.60, train_Energy_MAE=2.500, train_Force_MAE=0.349, train_Stress_MAE=0.000, train_Site_Wise_MAE=0.000, train_Energy_RMSE=2.660, train_Force_RMSE=0.487, train_Stress_RMSE=0.000, train_Site_Wise_RMSE=0.000]

    `Trainer.fit` stopped: `max_epochs=1` reached.


    Epoch 0: 100%|██████████| 163/163 [00:40<00:00,  3.99it/s, v_num=1, val_Total_Loss=0.815, val_Energy_MAE=0.674, val_Force_MAE=0.289, val_Stress_MAE=0.000, val_Site_Wise_MAE=0.000, val_Energy_RMSE=0.730, val_Force_RMSE=0.419, val_Stress_RMSE=0.000, val_Site_Wise_RMSE=0.000, train_Total_Loss=21.60, train_Energy_MAE=2.500, train_Force_MAE=0.349, train_Stress_MAE=0.000, train_Site_Wise_MAE=0.000, train_Energy_RMSE=2.660, train_Force_RMSE=0.487, train_Stress_RMSE=0.000, train_Site_Wise_RMSE=0.000]

```python
# test the model, remember to set inference_mode=False in trainer (see above)
trainer.test(dataloaders=test_loader)
```

    Restoring states from the checkpoint path at logs/M3GNet_training/version_0/checkpoints/epoch=0-step=163.ckpt
    Loaded model weights from the checkpoint at logs/M3GNet_training/version_0/checkpoints/epoch=0-step=163.ckpt
    Testing DataLoader 0: 100%|██████████| 21/21 [00:04<00:00,  4.25it/s]
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃        Test metric        ┃       DataLoader 0        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │      test_Energy_MAE      │    0.39233624935150146    │
    │     test_Energy_RMSE      │    0.47101688385009766    │
    │      test_Force_MAE       │    0.2673814296722412     │
    │      test_Force_RMSE      │    0.3861512243747711     │
    │    test_Site_Wise_MAE     │            0.0            │
    │    test_Site_Wise_RMSE    │            0.0            │
    │      test_Stress_MAE      │            0.0            │
    │     test_Stress_RMSE      │            0.0            │
    │      test_Total_Loss      │     0.754666268825531     │
    └───────────────────────────┴───────────────────────────┘

```python
# save trained model
model_export_path = "./trained_model/"
model.save(model_export_path)

# load trained model
model = matgl.load_model(path = model_export_path)
```

## Finetuning a pre-trained M3GNet
In the previous cells, we demonstrated the process of training an M3GNet from scratch. Next, let's see how to perform additional training on an M3GNet that has already been trained using Materials Project data.


```python
# download a pre-trained M3GNet
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model
lit_module_finetune = PotentialLightningModule(model=model_pretrained, lr=1e-4)
```


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

      | Name  | Type              | Params
    --------------------------------------------
    0 | mae   | MeanAbsoluteError | 0
    1 | rmse  | MeanSquaredError  | 0
    2 | model | Potential         | 288 K
    --------------------------------------------
    288 K     Trainable params
    0         Non-trainable params
    288 K     Total params
    1.153     Total estimated model params size (MB)


    Epoch 0: 100%|██████████| 163/163 [00:37<00:00,  4.34it/s, v_num=1, val_Total_Loss=5.420, val_Energy_MAE=1.070, val_Force_MAE=0.407, val_Stress_MAE=0.000, val_Site_Wise_MAE=0.000, val_Energy_RMSE=1.330, val_Force_RMSE=0.615, val_Stress_RMSE=0.000, val_Site_Wise_RMSE=0.000, train_Total_Loss=21.80, train_Energy_MAE=3.270, train_Force_MAE=0.572, train_Stress_MAE=0.000, train_Site_Wise_MAE=0.000, train_Energy_RMSE=3.430, train_Force_RMSE=0.870, train_Stress_RMSE=0.000, train_Site_Wise_RMSE=0.000]

    `Trainer.fit` stopped: `max_epochs=1` reached.


    Epoch 0: 100%|██████████| 163/163 [00:39<00:00,  4.17it/s, v_num=1, val_Total_Loss=5.420, val_Energy_MAE=1.070, val_Force_MAE=0.407, val_Stress_MAE=0.000, val_Site_Wise_MAE=0.000, val_Energy_RMSE=1.330, val_Force_RMSE=0.615, val_Stress_RMSE=0.000, val_Site_Wise_RMSE=0.000, train_Total_Loss=21.80, train_Energy_MAE=3.270, train_Force_MAE=0.572, train_Stress_MAE=0.000, train_Site_Wise_MAE=0.000, train_Energy_RMSE=3.430, train_Force_RMSE=0.870, train_Stress_RMSE=0.000, train_Site_Wise_RMSE=0.000]



```python
# save trained model
model_save_path = "./finetuned_model/"
model_pretrained.save(model_save_path)
# load trained model
trained_model = matgl.load_model(path = model_save_path)
```


```python
# This code just performs cleanup for this notebook.

for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

shutil.rmtree("logs")
shutil.rmtree("trained_model")
shutil.rmtree("finetuned_model")
```