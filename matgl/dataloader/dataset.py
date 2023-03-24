"""
Tools to construct a dataloader for DGL grphs
"""
from __future__ import annotations

import json
import os
from typing import Callable

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from pymatgen.core import Structure
from tqdm import trange

from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.layers.bond_expansion import BondExpansion


def _collate_fn(batch):
    """
    Merge a list of dgl graphs to form a batch
    """
    graphs, labels, graph_attr = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    graph_attr = torch.stack(graph_attr)
    return g, labels, graph_attr


def _collate_fn_efs(batch):
    """
    Merge a list of dgl graphs to form a batch
    """
    graphs, line_graphs, graph_attr, energies, forces, stresses = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.tensor(energies, dtype=torch.float32)
    f = torch.vstack(forces)
    s = torch.vstack(stresses)
    graph_attr = torch.stack(graph_attr)
    return g, l_g, graph_attr, e, f, s


def MGLDataLoader(
    train_data: dgl.data.utils.Subset,
    val_data: dgl.data.utils.Subset,
    test_data: dgl.data.utils.Subset,
    collate_fn: Callable,
    batch_size: int,
    num_workers: int,
    use_ddp: bool = False,
):
    """
    Dataloader for MEGNet training
    Args:
    train_data: training data
    val_data: validation data
    test_data: testing data
    collate_fn:
    """
    train_loader = GraphDataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        use_ddp=use_ddp,
    )

    val_loader = GraphDataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = GraphDataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class MEGNetDataset(DGLDataset):
    """
    Create a dataset including dgl graphs
    """

    def __init__(
        self,
        structures: list,
        labels: list,
        label_name: str,
        converter,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 20,
        width: float = 0.5,
        name: str = "MEGNETDataset",
    ):
        """
        Args:
        structures: Pymatgen strutcure
        labels: property values
        label: label name
        converter: Transformer for converting structures to DGL graphs, e.g., Pmg2Graph.
        initial: initial distance for Gaussian expansions
        final: final distance for Gaussian expansions
        num_centers: number of Gaussian functions
        width: width of Gaussian functions
        """
        self.converter = converter
        self.structures = structures
        self.labels = torch.FloatTensor(labels)
        self.label_name = label_name
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width
        super().__init__(name=name)

    def has_cache(self, filename: str = "dgl_graph.bin") -> bool:
        """
        Check if the dgl_graph.bin exists or not
        Args:
            :filename: Name of file storing dgl graphs
        Returns: True if file exists.
        """
        return os.path.exists(filename)

    def process(self) -> tuple:
        """
        Convert Pymatgen structure into dgl graphs
        """
        num_graphs = self.labels.shape[0]
        self.graphs = []
        self.graph_attr = []
        bond_expansion = BondExpansion(
            rbf_type="Gaussian", initial=self.initial, final=self.final, num_centers=self.num_centers, width=self.width
        )
        for idx in trange(num_graphs):
            structure = self.structures[idx]
            if isinstance(structure, Structure):
                graph, state_attr = self.converter.get_graph_from_structure(structure=structure)
            else:
                graph, state_attr = self.converter.get_graph_from_molecule(mol=structure)
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["edge_attr"] = bond_expansion(bond_dist)
            self.graphs.append(graph)
            self.graph_attr.append(state_attr)
        self.graph_attr = torch.tensor(self.graph_attr)  # type: ignore

        return self.graphs, self.graph_attr

    def save(self, filename: str = "dgl_graph.bin", filename_graph_attr: str = "graph_attr.pt"):
        """
        Save dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename_graph_attr: Name of file storing graph attrs
        """
        labels_with_key = {self.label_name: self.labels}
        save_graphs(filename, self.graphs, labels_with_key)
        torch.save(self.graph_attr, filename_graph_attr)

    def load(self, filename: str = "dgl_graph.bin", filename_graph_attr: str = "graph_attr.pt"):
        """
        Load dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename: Name of file storing state attrs
        """
        self.graphs, label_dict = load_graphs(filename)
        self.label = torch.stack([label_dict[key] for key in self.label_keys], dim=1)
        self.graph_attr = torch.load("graph_attr.pt")

    def __getitem__(self, idx: int):
        """
        Get graph and label with idx
        """
        return self.graphs[idx], self.labels[idx], self.graph_attr[idx]

    def __len__(self):
        """
        Get size of dataset
        """
        return len(self.graphs)


class M3GNetDataset(DGLDataset):
    """
    Create a dataset including dgl graphs
    """

    def __init__(
        self,
        structures: list,
        energies: list,
        forces: list,
        stresses: None | list,
        converter,
        threebody_cutoff: float,
        name="M3GNETDataset",
    ):
        """
        Args:
        structures: Pymatgen strutcure
        labels: property values
        label: label name
        converter: Transformer for converting structures to DGL graphs, e.g., Pmg2Graph.
        initial: initial distance for Gaussian expansions
        final: final distance for Gaussian expansions
        num_centers: number of Gaussian functions
        width: width of Gaussian functions
        """
        self.converter = converter
        self.structures = structures
        self.energies = energies
        self.forces = forces
        self.threebody_cutoff = threebody_cutoff
        self.stresses = np.zeros(len(self.energies)) if stresses is None else stresses
        super().__init__(name=name)

    def has_cache(self, filename: str = "dgl_graph.bin") -> bool:
        """
        Check if the dgl_graph.bin exists or not
        Args:
            :filename: Name of file storing dgl graphs
        Returns: True if file exists.
        """
        return os.path.exists(filename)

    def process(self) -> tuple:
        """
        Convert Pymatgen structure into dgl graphs
        """
        num_graphs = len(self.energies)
        self.graphs = []
        self.line_graphs = []
        self.graph_attr = []
        for idx in trange(num_graphs):
            structure = self.structures[idx]
            if isinstance(structure, Structure):
                graph, state_attr = self.converter.get_graph_from_structure(structure=structure)
            else:
                graph, state_attr = self.converter.get_graph_from_molecule(mol=structure)
            self.graphs.append(graph)
            self.graph_attr.append(state_attr)
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            line_graph = create_line_graph(graph, self.threebody_cutoff)
            line_graph.ndata.pop("bond_vec")
            line_graph.ndata.pop("bond_dist")
            line_graph.ndata.pop("pbc_offset")
            self.line_graphs.append(line_graph)
        self.graph_attr = torch.tensor(self.graph_attr)  # type: ignore

        return self.graphs, self.line_graphs, self.graph_attr

    def save(
        self,
        filename: str = "dgl_graph.bin",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_graph_attr: str = "graph_attr.pt",
    ):
        """
        Save dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename_graph_attr: Name of file storing graph attrs
        """
        labels_with_key = {"energies": self.energies, "forces": self.forces, "stresses": self.stresses}
        save_graphs(filename, self.graphs)
        save_graphs(filename_line_graph, self.line_graphs)
        torch.save(self.graph_attr, filename_graph_attr)
        with open("labels.json", "w") as file:
            file.write("".join(str(labels_with_key).split("\n")))

    def load(
        self,
        filename: str = "dgl_graph.bin",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_graph_attr: str = "graph_attr.pt",
    ):
        """
        Load dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename: Name of file storing state attrs
        """
        self.graphs = load_graphs(filename)
        self.line_graphs = load_graphs(filename_line_graph)
        with open("labels.json") as file:
            labels = json.load(file)
        self.energies = labels["energies"]
        self.forces = labels["forces"]
        self.stresses = labels["stresses"]
        self.graph_attr = torch.load("graph_attr.pt")

    def __getitem__(self, idx: int):
        """
        Get graph and label with idx
        """
        return (
            self.graphs[idx],
            self.line_graphs[idx],
            self.graph_attr[idx],
            self.energies[idx],
            torch.tensor(self.forces[idx]),
            torch.tensor(self.stresses[idx]),
        )

    def __len__(self):
        """
        Get size of dataset
        """
        return len(self.graphs)
