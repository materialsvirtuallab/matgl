"""
Refactoring script to convert saved models to new names.
"""
from __future__ import annotations

import torch

from matgl.models import MEGNet

# model_path = 'pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi'

model_path = "pretrained_models/MEGNet-MP-2018.6.1-Eform"
state_path = model_path + "/state.pt"

d = torch.load(state_path)

newd = {}

for k, v in d.items():
    k = k.replace("attr_encoder", "state_encoder")
    k = k.replace("attr_func", "state_func")
    k = k.replace("state_embedding_layer", "layer_state_embedding")
    k = k.replace("node_embedding_layer", "layer_node_embedding")
    k = k.replace("edge_embedding_layer", "layer_edge_embedding")
    if "attr" in k:
        print(k)
    newd[k] = v

torch.save(newd, state_path)

model = MEGNet.from_dir_new(model_path)
