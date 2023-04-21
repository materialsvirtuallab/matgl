# Simple training of formation energy from material projects (version:mp.2018.6.1.json)
# Author: Tsz Wai Ko (Kenko)
# Email: t1ko@ucsd.edu

import torch
import dgl
import os
from pymatgen.core import Element, Structure


## Import megnet related modules
from matgl.graph.converters import get_element_list, Pmg2Graph
from matgl.graph.compute import compute_pair_vector_and_distance
from matgl.models.megnet import MEGNet, MEGNetCalculator

# from matgl.utils.predictors import MEGNetCalculator
from matgl.layers.bond_expansion import BondExpansion

# define the CWD path
path = os.getcwd()
# define the device, either "cuda" or "cpu"
device = torch.device("cpu")
# load the pre-trained MEGNet model
model = MEGNet.load()
# map the model to CPU or GPU
model = model.to(device)
# read structure
struct = Structure.from_file(path + "/examples/pretrained_model/MP-2018.6.1-Eform/Cu.cif")
# create a graph converter
cry_graph = Pmg2Graph(element_types=model.element_types, cutoff=model.cutoff)
# convert pymatgen structure into dgl graph
graph, graph_attrs = cry_graph.get_graph_from_structure(structure=struct)
# define the Gaussian expansion
bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
# compute bond vectors and distances
bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
# expand the bond distance into edge attributes through Gaussian expansion
graph.edata["edge_attr"] = bond_expansion(bond_dist)
# move all necessary inputs into device
graph = graph.to(device)
graph.edata["edge_attr"] = graph.edata["edge_attr"].to(device)
graph.ndata["node_type"] = graph.ndata["node_type"].to(device)
graph_attrs = torch.tensor(graph_attrs).to(device)
# define MEGNet calculator
predictor = MEGNetCalculator(model=model, data_std=model.data_std, data_mean=model.data_mean)
Eform_pred = predictor(graph, graph_attrs)
print("The predicted formation energy for a FCC crystal is", float(Eform_pred.detach().numpy()), "eV/atom")
