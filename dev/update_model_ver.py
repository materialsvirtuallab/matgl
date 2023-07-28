"""This script is used to update model version numbers ex-post if the versions were updated manually."""
from __future__ import annotations

import json
import sys

import torch

version = int(sys.argv[1])

d = torch.load("model.pt", map_location=torch.device("cpu"))
d["model"]["@model_version"] = version

torch.save(d, "model.pt")

with open("model.json") as f:
    d = json.load(f)

d["kwargs"]["model"]["@model_version"] = version

with open("model.json", "w") as f:
    json.dump(d, f, default=lambda o: str(o), indent=4)
