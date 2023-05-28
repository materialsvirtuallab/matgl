# Simple training of MEGNet formation energy model for material projects (version:mp.2018.6.1.json)
# Author: Tsz Wai Ko (Kenko)
# Email: t1ko@ucsd.edu
from __future__ import annotations

import logging
import math
import os
import zipfile

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import split_dataset

# Import megnet related modules
from pymatgen.core import Structure
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, _collate_fn
from matgl.layers._bond import BondExpansion
from matgl.models import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelTrainer

SEED = 42
EPOCHS = 2000

# define the default device for torch tensor. Either 'cuda' or 'cpu'
torch.set_default_device("cpu")
# define the torch.Generator. Either 'cuda' or 'cpu'
generator = torch.Generator(device="cpu")

logging.basicConfig(level=logging.INFO)


# define a raw data loading function
def load_dataset():
    if not os.path.exists("mp.2018.6.1.json"):
        logging.info("Downloading...")
        f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
        with zipfile.ZipFile(f.local_path) as zf:
            zf.extractall(".")
    logging.info("Loading json...")
    data = pd.read_json("mp.2018.6.1.json")
    structures = []
    mp_id = []
    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_id.append(mid)

    return structures, mp_id, data["formation_energy_per_atom"].tolist()


# load the MP raw dataset
structures, mp_id, Eform_per_atom = load_dataset()
# get element types in the dataset
elem_list = get_element_list(structures)
# setup a graph converter
cry_graph = Structure2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into MEGNetDataset
mp_dataset = MEGNetDataset(
    structures, Eform_per_atom, "Eform", converter=cry_graph, initial=0.0, final=5.0, num_centers=100, width=0.5
)
# separate the dataset into training, validation and test data
train_data, val_data, test_data = split_dataset(
    mp_dataset,
    frac_list=[0.9, 0.05, 0.05],
    shuffle=True,
    random_state=SEED,
)
logging.info(f"Train, Valid, Test size = {len(train_data)}, {len(val_data)}, {len(test_data)}")
# get the average and standard deviation from the training set
# setup the embedding layer for node attributes
node_embed = nn.Embedding(len(elem_list), 16)
# define the bond expansion
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)

# setup the architecture of MEGNet model
model = MEGNet(
    dim_node_embedding=16,
    dim_edge_embedding=100,
    dim_state_embedding=2,
    nblocks=3,
    hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32),
    nlayers_set2set=1,
    niters_set2set=2,
    hidden_layer_sizes_output=(32, 16),
    is_classification=False,
    layer_node_embedding=node_embed,
    activation_type="softplus2",
    graph_converter=cry_graph,
    bond_expansion=bond_expansion,
    cutoff=4.0,
    gauss_width=0.5,
)


# define the weight initialization using Xavier scheme
def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if param.dim() < 2:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[0])
                param.data.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                param.data.uniform_(-bound, bound)


xavier_init(model)
# setup the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=5.0e-5, amsgrad=True)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * 10, eta_min=1.0e-4)
# define the loss function for training and validation
train_loss_function = F.l1_loss
validate_loss_function = F.l1_loss
# using GraphDataLoader for batched graphs
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=_collate_fn,
    batch_size=128,
    num_workers=0,
    generator=generator,
)

print(model)

# setup the MEGNetTrainer
trainer = ModelTrainer(model=model, optimizer=optimizer, scheduler=scheduler)
# Train !
trainer.train(
    n_epochs=EPOCHS,
    train_loss_func=train_loss_function,
    val_loss_func=validate_loss_function,
    train_loader=train_loader,
    val_loader=val_loader,
)
