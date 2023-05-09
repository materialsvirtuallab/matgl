# Simple training of multi-fidelity MEGNet bandgap model for materials project (version:mp.2019.4.1.json)
# Author: Tsz Wai Ko (Kenko)
# Email: t1ko@ucsd.edu
import gzip
import json
from typing import List

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

import math

# Import megnet related modules
from pymatgen.core import Structure
from matgl.ext.pymatgen import get_element_list, Structure2Graph
from matgl.layers._bond import BondExpansion
from torch.optim.lr_scheduler import CosineAnnealingLR
from matgl.utils.training import ModelTrainer
from matgl.graph.data import MEGNetDataset, _collate_fn, MGLDataLoader
from matgl.models import MEGNet

ALL_FIDELITIES = ["pbe", "gllb-sc", "hse", "scan"]
TRAIN_FIDELITIES = ["pbe", "gllb-sc", "hse", "scan"]
VAL_FIDELITIES = ["gllb-sc", "hse", "scan"]
TEST_FIDELITIES = ["pbe", "gllb-sc", "hse", "scan"]

SEED = 42
EPOCHS = 2000
path = os.getcwd()

# define the default device for torch tensor. Either 'cuda' or 'cpu'
torch.set_default_device("cpu")
# define the torch.Generator. Either 'cuda' or 'cpu'
generator = torch.Generator(device="cpu")


# define a function for computating the statistics of dataset
def compute_data_stats(dataset):
    graphs, targets, attrs = zip(*dataset)
    targets = torch.stack(targets)

    data_std, data_mean = torch.std_mean(targets)

    return data_std, data_mean


# load materials project structures

if not os.path.isfile("mp.2019.04.01.json"):
    raise RuntimeError("Please download the data first! Use runall.sh in this directory if needed.")

with open("mp.2019.04.01.json") as f:
    structure_data = {i["material_id"]: i["structure"] for i in json.load(f)}
print("All structures in mp.2019.04.01.json contain %d structures" % len(structure_data))


#  Band gap data
with gzip.open("data_no_structs.json.gz", "rb") as f:  # type: ignore
    bandgap_data = json.loads(f.read())

useful_ids = set.union(*[set(bandgap_data[i].keys()) for i in ALL_FIDELITIES])  # mp ids that are used in training
print("Only %d structures are used" % len(useful_ids))
print("Calculating the graphs for all structures... this may take minutes.")
structure_data = {i: structure_data[i] for i in useful_ids}
structure_data = {i: Structure.from_str(j, fmt="cif") for i, j in structure_data.items()}

#  Generate graphs with fidelity information
structures = []
targets = []
material_ids = []
graph_attrs = []
for fidelity_id, fidelity in enumerate(ALL_FIDELITIES):
    for mp_id in bandgap_data[fidelity]:
        structure = deepcopy(structure_data[mp_id])

        # The fidelity information is included here by changing the state attributes
        # PBE: 0, GLLB-SC: 1, HSE: 2, SCAN: 3
        graph_attrs.append(fidelity_id)
        structures.append(structure)
        targets.append(bandgap_data[fidelity][mp_id])
        # the new id is of the form mp-id_fidelity, e.g., mp-1234_pbe
        material_ids.append(f"{mp_id}_{fidelity}")

final_structures = {i: j for i, j in zip(material_ids, structures)}
final_targets = {i: j for i, j in zip(material_ids, targets)}
final_graph_attrs = {i: j for i, j in zip(material_ids, graph_attrs)}


# split the dataset

from sklearn.model_selection import train_test_split

# train:val:test = 8:1:1
fidelity_list = [i.split("_")[1] for i in material_ids]
train_val_ids, test_ids = train_test_split(material_ids, stratify=fidelity_list, test_size=0.1, random_state=SEED)
fidelity_list = [i.split("_")[1] for i in train_val_ids]
train_ids, val_ids = train_test_split(train_val_ids, stratify=fidelity_list, test_size=0.1 / 0.9, random_state=SEED)

# remove pbe from validation
val_ids = [i for i in val_ids if not i.endswith("pbe")]


print("Train, val and test data sizes are ", len(train_ids), len(val_ids), len(test_ids))


# get the train, val and test graph-target pairs
def get_graphs_targets(ids):
    """
    Get graphs and targets list from the ids

    Args:
        ids (List): list of ids

    Returns:
        list of graphs and list of target values
    """
    ids = [i for i in ids if i in final_structures]
    return [final_structures[i] for i in ids], [final_graph_attrs[i] for i in ids], [final_targets[i] for i in ids]


train_structures, train_graph_attrs, train_targets = get_graphs_targets(train_ids)
val_structures, val_graph_attrs, val_targets = get_graphs_targets(val_ids)

elem_list = get_element_list(train_structures)
cry_graph = Structure2Graph(element_types=elem_list, cutoff=5.0)


## Load the dataset using MEGNetDataset
training_set = MEGNetDataset(
    train_structures,
    train_targets,
    "bandgap",
    converter=cry_graph,
    initial=0.0,
    final=6.0,
    num_centers=100,
    width=0.5,
    graph_labels=train_graph_attrs,
)

validation_set = MEGNetDataset(
    val_structures,
    val_targets,
    "bandgap",
    converter=cry_graph,
    initial=0.0,
    final=6.0,
    num_centers=100,
    width=0.5,
    graph_labels=val_graph_attrs,
)

# obtain the average and standard deviation from the training data
train_std, train_mean = compute_data_stats(training_set)
# define the embedding layer for nodel and state attributes
node_embed = nn.Embedding(len(elem_list), 16)
attr_embed = nn.Embedding(len(ALL_FIDELITIES), 16)
# define the bond expansion
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=6.0, num_centers=100, width=0.5)

# define the achitecture of multi-fidelity MEGNet model
model = MEGNet(
    dim_node_embedding=16,
    dim_edge_embedding=100,
    dim_state_embedding=16,
    nblocks=3,
    hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32),
    nlayers_set2set=1,
    niters_set2set=3,
    hidden_layer_sizes_output=(32, 16),
    is_classification=False,
    layer_node_embedding=node_embed,
    layer_state_embedding=attr_embed,
    activation_type="softplus2",
    include_state_embedding=True,
    graph_converter=cry_graph,
    bond_expansion=bond_expansion,
    cutoff=5.0,
    gauss_width=0.5,
    data_mean=train_mean,
    data_std=train_std,
)


# setup the weight initialization using Xavier scheme
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2, amsgrad=True)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * 10, eta_min=1.0e-4)

# define the loss functions
train_loss_function = F.l1_loss
validate_loss_function = F.l1_loss
## using GraphDataLoader for batched graphs
train_loader, val_loader = MGLDataLoader(
    train_data=training_set,
    val_data=validation_set,
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
    nepochs=EPOCHS,
    train_loss_func=train_loss_function,
    val_loss_func=validate_loss_function,
    train_loader=train_loader,
    val_loader=val_loader,
)
