"""Tools to construct a dataset of DGL graphs."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from tqdm import trange

from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.layers import BondExpansion

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter


def collate_fn(batch, include_line_graph: bool = False):
    """Merge a list of dgl graphs to form a batch."""
    if include_line_graph:
        graphs, line_graphs, state_attr, labels = map(list, zip(*batch))
    else:
        graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    state_attr = torch.stack(state_attr)
    if include_line_graph:
        l_g = dgl.batch(line_graphs)
        return g, l_g, state_attr, labels
    return g, labels, state_attr


def collate_fn_efs(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, line_graphs, state_attr, energies, forces, stresses = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.tensor(energies, dtype=torch.float32)
    f = torch.vstack(forces)
    s = torch.vstack(stresses)
    state_attr = torch.stack(state_attr)
    return g, l_g, state_attr, e, f, s


def MGLDataLoader(
    train_data: dgl.data.utils.Subset,
    val_data: dgl.data.utils.Subset,
    collate_fn: Callable,
    batch_size: int,
    num_workers: int,
    use_ddp: bool = False,
    pin_memory: bool = False,
    test_data: dgl.data.utils.Subset | None = None,
    generator: torch.Generator | None = None,
) -> tuple[GraphDataLoader, ...]:
    """Dataloader for MEGNet training.

    Args:
        train_data (dgl.data.utils.Subset): Training dataset.
        val_data (dgl.data.utils.Subset): Validation dataset.
        collate_fn (Callable): Collate function.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        use_ddp (bool, optional): Whether to use DDP. Defaults to False.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        test_data (dgl.data.utils.Subset | None, optional): Test dataset. Defaults to None.
        generator (torch.Generator | None, optional): Random number generator. Defaults to None.

    Returns:
        tuple[GraphDataLoader, ...]: Train, validation and test data loaders. Test data
            loader is None if test_data is None.
    """
    train_loader = GraphDataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_ddp=use_ddp,
        generator=generator,
    )

    val_loader = GraphDataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if test_data is not None:
        test_loader = GraphDataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class MEGNetDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        structures: list,
        labels: list,
        label_name: str,
        converter: GraphConverter,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
        name: str = "MEGNETDataset",
        graph_labels: list | None = None,
    ):
        """
        Args:
            structures: Pymatgen structure
            labels: property values
            label_name: label name
            converter: Transformer for converting structures to DGL graphs, e.g., Pmg2Graph.
            initial: initial distance for Gaussian expansions
            final: final distance for Gaussian expansions
            num_centers: number of Gaussian functions
            width: width of Gaussian functions
            name: Name of dataset
            graph_labels: graph attributes either integers and floating point numbers.
        """
        self.converter = converter
        self.structures = structures
        self.labels = torch.FloatTensor(labels)
        self.label_name = label_name
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width
        self.graph_labels = graph_labels

        super().__init__(name=name)

    def has_cache(self, filename: str = "dgl_graph.bin") -> bool:
        """Check if the dgl_graph.bin exists or not
        Args:
            :filename: Name of file storing dgl graphs
        Returns: True if file exists.
        """
        return os.path.exists(filename)

    def process(self) -> tuple:
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = self.labels.shape[0]
        graphs = []
        state_attrs = []
        bond_expansion = BondExpansion(
            rbf_type="Gaussian", initial=self.initial, final=self.final, num_centers=self.num_centers, width=self.width
        )
        for idx in trange(num_graphs):
            structure = self.structures[idx]
            graph, state_attr = self.converter.get_graph(structure)
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["edge_attr"] = bond_expansion(bond_dist)
            graphs.append(graph)
            state_attrs.append(state_attr)
        if self.graph_labels is not None:
            if np.array(self.graph_labels).dtype == "int64":
                state_attrs = torch.tensor(self.graph_labels).long()  # type: ignore
            else:
                state_attrs = torch.tensor(self.graph_labels)  # type: ignore
        else:
            state_attrs = torch.tensor(state_attrs)  # type: ignore
        self.graphs = graphs
        self.state_attr = state_attrs
        return self.graphs, self.state_attr

    def save(self, filename: str = "dgl_graph.bin", filename_state_attr: str = "state_attr.pt"):
        """Save dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename_state_attr: Name of file storing graph attrs.
        """
        labels_with_key = {self.label_name: self.labels}
        save_graphs(filename, self.graphs, labels_with_key)
        torch.save(self.state_attr, filename_state_attr)

    def load(self, filename: str = "dgl_graph.bin", filename_state_attr: str = "state_attr.pt"):
        """Load dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename: Name of file storing state attrs.
        """
        self.graphs, label_dict = load_graphs(filename)
        self.label = torch.stack([label_dict[key] for key in self.label_keys], dim=1)
        self.state_attr = torch.load("state_attr.pt")

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        return self.graphs[idx], self.state_attr[idx], self.labels[idx]

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)


class M3GNetDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        converter: GraphConverter,
        threebody_cutoff: float,
        structures: list,
        energies: list | None = None,
        forces: list | None = None,
        stresses: list | None = None,
        labels: list | None = None,
        name="M3GNETDataset",
        label_name: str | None = None,
        graph_labels: list | None = None,
    ):
        """
        Args:
            converter: dgl graph converter
            threebody_cutoff: cutoff for three body
            structures: Pymatgen structure
            energies: Target energies
            forces: Target forces
            stresses: Target stresses
            labels: target properties
            name: name of dataset
            label_name: name of target properties
            graph_labels: state attributes.
        """
        self.converter = converter
        self.structures = structures
        self.energies = energies
        self.forces = forces
        self.labels = labels
        self.label_name = label_name
        self.threebody_cutoff = threebody_cutoff
        self.stresses = np.zeros(len(self.structures)) if stresses is None else stresses
        self.graph_labels = graph_labels
        super().__init__(name=name)

    def has_cache(self, filename: str = "dgl_graph.bin") -> bool:
        """Check if the dgl_graph.bin exists or not
        Args:
            :filename: Name of file storing dgl graphs
        Returns: True if file exists.
        """
        return os.path.exists(filename)

    def process(self) -> tuple:
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)
        graphs = []
        line_graphs = []
        state_attrs = []
        for idx in trange(num_graphs):
            structure = self.structures[idx]
            graph, state_attr = self.converter.get_graph(structure)
            graphs.append(graph)
            state_attrs.append(state_attr)
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            line_graph = create_line_graph(graph, self.threebody_cutoff)
            for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                line_graph.ndata.pop(name)
            line_graphs.append(line_graph)
        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()  # type: ignore
        else:
            state_attrs = torch.tensor(state_attrs)  # type: ignore

        self.graphs = graphs
        self.line_graphs = line_graphs
        self.state_attr = state_attrs

        return self.graphs, self.line_graphs, self.state_attr

    def save(
        self,
        filename: str = "dgl_graph.bin",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
    ):
        """Save dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        :filename_state_attr: Name of file storing graph attrs.
        """
        if self.labels is None:
            labels_with_key = {"energies": self.energies, "forces": self.forces, "stresses": self.stresses}
        else:
            labels_with_key = {self.label_name: self.labels}  # type: ignore
        save_graphs(filename, self.graphs)
        save_graphs(filename_line_graph, self.line_graphs)
        torch.save(self.state_attr, filename_state_attr)
        with open("labels.json", "w") as file:
            file.write("".join(str(labels_with_key).split("\n")))

    def load(
        self,
        filename: str = "dgl_graph.bin",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
    ):
        """
        Load dgl graphs from files.

        Args:
            filename: Name of file storing dgl graphs
            filename_line_graph: Name of file storing dgl line graphs
            filename_state_attr: Name of file storing state attrs.
        """
        self.graphs = load_graphs(filename)
        self.line_graphs = load_graphs(filename_line_graph)
        with open("labels.json") as file:
            labels: dict = json.load(file)
        if self.labels is None:
            self.energies = labels["energies"]
            self.forces = labels["forces"]
            self.stresses = labels["stresses"]
            self.state_attr = torch.load("state_attr.pt")
        else:
            self.labels = labels  # type: ignore

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        if self.labels is None:
            return (
                self.graphs[idx],
                self.line_graphs[idx],
                self.state_attr[idx],
                self.energies[idx],  # type: ignore
                torch.tensor(self.forces[idx]).float(),  # type: ignore
                torch.tensor(self.stresses[idx]).float(),  # type: ignore
            )
        return (self.graphs[idx], self.line_graphs[idx], self.state_attr[idx], self.labels[idx])

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)
