# Prediction of bandgap for a FCC Cu crystal from pretrained model (version:mp.2019.4.1.json)
# Author: Tsz Wai Ko (Kenko)
# Email: t1ko@ucsd.edu

import os
import torch
import dgl
from pymatgen.core import Element, Structure


## Import megnet related modules
from matgl.models.megnet import MEGNet, MEGNetCalculator
from matgl.layers.bond_expansion import BondExpansion
from matgl.graph.converters import get_element_list, Pmg2Graph
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.layers.bond_expansion import BondExpansion

# define the current working directory
path = os.getcwd()

# define the device, either "cuda" or "cpu"
device = torch.device("cpu")
# load the pre-trained MEGNet model
model = MEGNet.load("MP-2019.4.1-BandGap")
# map the model to CPU or GPU
model = model.to(device)
# read structure
struct = Structure.from_file(path + "/examples/pretrained_model/MP-2019.4.1-BandGap/Cu.cif")
# create a graph converter
cry_graph = Pmg2Graph(element_types=model.element_types, cutoff=model.cutoff)
# convert pymatgen structure into dgl graph
graph, graph_attrs = cry_graph.get_graph_from_structure(structure=struct)
# define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN) for multi-fidelity model
graph_attrs = torch.tensor([0])
# define the Gaussian expansion
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=6.0, num_centers=100, width=0.5)
# compute bond vectors and distances
bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
# expand the bond distance into edge attributes through Gaussian expansion
graph.edata["edge_attr"] = bond_expansion(bond_dist)
# move all necessary inputs into device
graph.ndata["node_type"].to(device)
model.data_mean.to(device)
model.data_std.to(device)
graph_attrs = torch.tensor(graph_attrs).to(device)
# define MEGNet calculator
predictor = MEGNetCalculator(model=model, data_std=model.data_std, data_mean=model.data_mean)
BandGap = predictor(graph, graph_attrs)
print("The predicted PBE BandGap for a FCC Cu crystal is ", float(BandGap.detach().numpy()), "eV")
