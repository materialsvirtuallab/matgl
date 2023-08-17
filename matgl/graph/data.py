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
        filename: str = "dgl_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        structures: list | None = None,
        labels: list[float] | None = None,
        label_name: str | None = None,
        converter: GraphConverter | None = None,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
        name: str = "MEGNETDataset",
        graph_labels: list[int | float] | None = None,
    ):
        """
        Args:
            filename: file name for storing dgl graphs and target properties
            filename_state_attr: file name for storing state attributes
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
        self.filename = filename
        self.filename_state_attr = filename_state_attr
        self.converter = converter
        self.structures = structures
        self.labels = None if labels is None else torch.FloatTensor(labels)
        self.label_name = label_name
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width
        self.graph_labels = graph_labels

        super().__init__(name=name)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not.

        Returns: True if file exists.
        """
        return os.path.exists(self.filename) and os.path.exists(self.filename_state_attr)

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = self.labels.shape[0]
        graphs = []
        state_attrs = []
        bond_expansion = BondExpansion(
            rbf_type="Gaussian",
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, state_attr = self.converter.get_graph(structure)  # type: ignore
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["edge_attr"] = bond_expansion(bond_dist)
            graphs.append(graph)
            state_attrs.append(state_attr)
        if self.graph_labels is not None:
            if np.array(self.graph_labels).dtype == "int64":
                state_attrs = torch.tensor(self.graph_labels).long()
            else:
                state_attrs = torch.tensor(self.graph_labels)
        else:
            state_attrs = torch.tensor(state_attrs)
        self.graphs = graphs
        self.state_attr = state_attrs
        return self.graphs, self.state_attr

    def save(self):
        """Save dgl graphs and labels."""
        labels_with_key = {self.label_name: torch.tensor(self.labels)}
        save_graphs(self.filename, self.graphs, labels_with_key)
        torch.save(self.state_attr, self.filename_state_attr)

    def load(self):
        """Load dgl graphs and labels."""
        self.graphs, label_dicts = load_graphs(self.filename)
        self.labels = torch.stack([label_dicts[key] for key in label_dicts], dim=1)
        self.state_attr = torch.load(self.filename_state_attr)

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        return self.graphs[idx], self.state_attr[idx], self.labels[idx]  # type: ignore

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)


class M3GNetDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        filename: str = "dgl_graph.bin",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_energies: str = "energies.json",
        filename_forces: str = "forces.json",
        filename_stresses: str = "stresses.json",
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        structures: list | None = None,
        energies: list[float] | None = None,
        forces: list[list[float]] | None = None,
        stresses: list[list[float]] | None = None,
        labels: list[float] | None = None,
        name="M3GNETDataset",
        label_name: str | None = None,
        graph_labels: list[int | float] | None = None,
    ):
        """
        Args:
            filename: file name for storaging dgl graphs
            filename_line_graph: file name for storaging dgl line graphs
            filename_state_attr: file name for storaging state attributes
            filename_energies: file name for storaging energies
            filename_forces: file name for storaging forces
            filename_stresses: file name for storagning stresses
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
        self.filename = filename
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_energies = filename_energies
        self.filename_forces = filename_forces
        self.filename_stresses = filename_stresses
        self.converter = converter
        self.structures = structures
        self.energies = energies.tolist() if isinstance(energies, np.ndarray) else energies
        self.forces = forces.tolist() if isinstance(forces, np.ndarray) else forces
        self.labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
        self.label_name = label_name
        self.threebody_cutoff = threebody_cutoff
        # it only happens when loading the data
        if self.energies is None and self.labels is None:
            self.stresses = None
        else:
            if stresses is None:
                self.stresses = np.zeros(len(self.structures)).tolist()  # type: ignore
            else:
                self.stresses = stresses.tolist() if type(stresses) is np.ndarray else stresses
        self.graph_labels = graph_labels
        super().__init__(name=name)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not."""
        return all(os.path.exists(f) for f in (self.filenamem, self.filename_line_graph, self.filename_state_attr))

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)  # type: ignore
        graphs = []
        line_graphs = []
        state_attrs = []
        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, state_attr = self.converter.get_graph(structure)  # type: ignore
            graphs.append(graph)
            state_attrs.append(state_attr)
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            line_graph = create_line_graph(graph, self.threebody_cutoff)  # type: ignore
            for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                line_graph.ndata.pop(name)
            line_graphs.append(line_graph)
        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()
        else:
            state_attrs = torch.tensor(state_attrs)

        self.graphs = graphs
        self.line_graphs = line_graphs
        self.state_attr = state_attrs

        return self.graphs, self.line_graphs, self.state_attr

    def save(self):
        """Save dgl graphs."""
        if self.labels is None:
            labels_with_key = {"energies": self.energies, "forces": self.forces, "stresses": self.stresses}
            save_graphs(self.filename, self.graphs)
            with open(self.filename_energies, "w") as f:
                json.dump(labels_with_key["energies"], f)
            with open(self.filename_forces, "w") as f:
                json.dump(labels_with_key["forces"], f)
            with open(self.filename_stresses, "w") as f:
                json.dump(labels_with_key["stresses"], f)
        else:
            labels_with_key = {self.label_name: torch.tensor(self.labels)}
            save_graphs(self.filename, self.graphs, labels_with_key)
        save_graphs(self.filename_line_graph, self.line_graphs)
        torch.save(self.state_attr, self.filename_state_attr)

    def load(self):
        """Load dgl graphs from files."""
        self.line_graphs, _ = load_graphs(self.filename_line_graph)
        self.state_attr = torch.load(self.filename_state_attr)
        if self.label_name is None:
            self.graphs, _ = load_graphs(self.filename)
            with open(self.filename_energies) as f:
                self.energies = json.load(f)
            with open(self.filename_forces) as f:
                self.forces = json.load(f)
            with open(self.filename_stresses) as f:
                self.stresses = json.load(f)
        else:
            self.graphs, label_dicts = load_graphs(self.filename)
            self.labels = torch.stack([label_dicts[key] for key in label_dicts], dim=1)  # type: ignore

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        if self.label_name is None:
            return (
                self.graphs[idx],
                self.line_graphs[idx],
                self.state_attr[idx],
                self.energies[idx],
                torch.tensor(self.forces[idx]).float(),
                torch.tensor(self.stresses[idx]).float(),  # type: ignore
            )
        return self.graphs[idx], self.line_graphs[idx], self.state_attr[idx], self.labels[idx]

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)
