"""Refactoring script to convert saved models to new names."""
from __future__ import annotations

import torch
from matgl.models import MEGNet

# model_path = "pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi"

model_path = "pretrained_models/MEGNet-MP-2018.6.1-Eform"

# state_path = model_path + "/model.pt"
state_path = model_path + "/state.pt"

d = torch.load(state_path)

newd = {}

name_mappings = {
    "attr_encoder": "state_encoder",
    "attr_func": "state_func",
    "state_embedding_layer": "layer_state_embedding",
    "node_embedding_layer": "layer_node_embedding",
    "edge_embedding_layer": "layer_edge_embedding",
    "dim_attr_embedding": "dim_state_embedding",
    "layer_attr_embedding": "layer_state_embedding",
    "layer_node_embedding": "embedding.layer_node_embedding",
    "layer_edge_emebedding": "embedding.layer_edge_embedding",
    "layer_state_embedding": "embedding.layer_state_embedding",
}

for k, v in d.items():
    for n1, n2 in name_mappings.items():
        k = k.replace(n1, n2)
    newd[k] = v

torch.save(newd, state_path)

# This last step is just to check that the new model loads correctly.
model = MEGNet.from_dir_new(model_path)
