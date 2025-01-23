"""Tools to construct a dataset of DGL graphs."""

from __future__ import annotations

import json
import os
from functools import partial
from typing import TYPE_CHECKING, Callable

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from tqdm import trange

import matgl
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter


def collate_fn_graph(batch, include_line_graph: bool = False, multiple_values_per_target: bool = False):
    """Merge a list of dgl graphs to form a batch."""
    line_graphs = None
    if include_line_graph:
        graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch))
    else:
        graphs, lattices, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = (
        torch.vstack([next(iter(d.values())) for d in labels])
        if multiple_values_per_target
        else torch.tensor([next(iter(d.values())) for d in labels], dtype=matgl.float_th)
    )
    state_attr = torch.stack(state_attr)
    lat = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))
    if include_line_graph:
        l_g = dgl.batch(line_graphs)
        return g, lat, l_g, state_attr, labels
    return g, lat, state_attr, labels


def collate_fn_pes(batch, include_stress: bool = True, include_line_graph: bool = False, include_magmom: bool = False):
    """Merge a list of dgl graphs to form a batch."""
    l_g = None
    if include_line_graph:
        graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch))
        l_g = dgl.batch(line_graphs)
    else:
        graphs, lattices, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    e = torch.tensor([d["energies"] for d in labels])  # type: ignore
    f = torch.vstack([d["forces"] for d in labels])  # type: ignore
    s = (
        torch.vstack([d["stresses"] for d in labels])  # type: ignore
        if include_stress is True
        else torch.tensor(np.zeros(e.size(dim=0)), dtype=matgl.float_th)
    )
    m = (
        torch.vstack([d["magmoms"] for d in labels])
        if include_magmom is True
        else torch.tensor(np.zeros(e.size(dim=0)), dtype=matgl.float_th)
    )
    state_attr = torch.stack(state_attr)
    lat = torch.stack(lattices)
    if include_line_graph:
        if include_magmom:
            return g, torch.squeeze(lat), l_g, state_attr, e, f, s, m
        return g, torch.squeeze(lat), l_g, state_attr, e, f, s
    return g, torch.squeeze(lat), state_attr, e, f, s


def MGLDataLoader(
    train_data: dgl.data.utils.Subset,
    val_data: dgl.data.utils.Subset,
    collate_fn: Callable | None = None,
    test_data: dgl.data.utils.Subset = None,
    **kwargs,
) -> tuple[GraphDataLoader, ...]:
    """Dataloader for MatGL training.

    Args:
        train_data (dgl.data.utils.Subset): Training dataset.
        val_data (dgl.data.utils.Subset): Validation dataset.
        collate_fn (Callable): Collate function.
        test_data (dgl.data.utils.Subset | None, optional): Test dataset. Defaults to None.
        **kwargs: Pass-through kwargs to dgl.dataloading.GraphDataLoader. Common ones you may want to set are
            batch_size, num_workers, use_ddp, pin_memory and generator.

    Returns:
        tuple[GraphDataLoader, ...]: Train, validation and test data loaders. Test data
            loader is None if test_data is None.
    """
    if collate_fn is None:
        if "forces" not in train_data.dataset.labels:
            collate_fn = collate_fn_graph
        else:
            if "stresses" not in train_data.dataset.labels:
                collate_fn = partial(collate_fn_pes, include_stress=False)
            else:
                if "magmoms" not in train_data.dataset.labels:
                    collate_fn = collate_fn_pes
                else:
                    collate_fn = partial(collate_fn_pes, include_stress=True, include_magmom=True)

    train_loader = GraphDataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = GraphDataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    if test_data is not None:
        test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class MGLDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        filename: str = "dgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        include_line_graph: bool = False,
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        directed_line_graph: bool = False,
        structures: list | None = None,
        labels: dict[str, list] | None = None,
        directory_name: str = "MGLDataset",
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        raw_dir: str = "./",
        save_dir: str = "./",
    ):
        """
        Args:
            filename: file name for storing dgl graphs.
            filename_lattice: file name for storing lattice matrixs.
            filename_line_graph: file name for storing dgl line graphs.
            filename_state_attr: file name for storing state attributes.
            filename_labels: file name for storing labels.
            include_line_graph: whether to include line graphs.
            converter: dgl graph converter.
            threebody_cutoff: cutoff for three body.
            directed_line_graph (bool): Whether to create a directed line graph (CHGNet), or an
                undirected 3body line graph (M3GNet)
                Default: False (for M3GNet)
            structures: Pymatgen structure.
            labels: targets, as a dict of {name: list of values}.
            directory_name: name of the generated directory that stores the dataset.
            graph_labels: state attributes.
            clear_processed: Whether to clear the stored structures after processing into graphs. Structures
                are not really needed after the conversion to DGL graphs and can take a significant amount of memory.
                Setting this to True will delete the structures from memory.
            save_cache: whether to save the processed dataset. The dataset can be reloaded from save_dir
                Default: True
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
                Default: current working directory
            save_dir : directory to save the processed dataset. Default: same as raw_dir.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        super().__init__(name=directory_name, raw_dir=raw_dir, save_dir=save_dir)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        if self.include_line_graph:
            files_to_check.append(self.filename_line_graph)
        return all(os.path.exists(os.path.join(self.save_path, f)) for f in files_to_check)

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)  # type: ignore
        graphs, lattices, line_graphs, state_attrs = [], [], [], []

        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, lattice, state_attr = self.converter.get_graph(structure)  # type: ignore
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
            graph.ndata["pos"] = torch.tensor(structure.cart_coords)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            if self.include_line_graph:
                line_graph = create_line_graph(graph, self.threebody_cutoff, directed=self.directed_line_graph)  # type: ignore
                for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                    line_graph.ndata.pop(name)
                line_graphs.append(line_graph)
            graph.ndata.pop("pos")
            graph.edata.pop("pbc_offshift")
        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()
        else:
            state_attrs = torch.tensor(np.array(state_attrs), dtype=matgl.float_th)

        if self.clear_processed:
            del self.structures
            self.structures = []

        self.graphs = graphs
        self.lattices = lattices
        self.state_attr = state_attrs
        if self.include_line_graph:
            self.line_graphs = line_graphs
            return self.graphs, self.lattices, self.line_graphs, self.state_attr
        return self.graphs, self.lattices, self.state_attr

    def save(self):
        """Save dgl graphs and labels to self.save_path."""
        if self.save_cache is False:
            return

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.labels:
            with open(os.path.join(self.save_path, self.filename_labels), "w") as file:
                json.dump(self.labels, file)
        save_graphs(os.path.join(self.save_path, self.filename), self.graphs)
        torch.save(self.lattices, os.path.join(self.save_path, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.save_path, self.filename_state_attr))
        if self.include_line_graph:
            save_graphs(os.path.join(self.save_path, self.filename_line_graph), self.line_graphs)

    def load(self):
        """Load dgl graphs from files."""
        self.graphs, _ = load_graphs(os.path.join(self.save_path, self.filename))
        self.lattices = torch.load(os.path.join(self.save_path, self.filename_lattice))
        if self.include_line_graph:
            self.line_graphs, _ = load_graphs(os.path.join(self.save_path, self.filename_line_graph))
        self.state_attr = torch.load(os.path.join(self.save_path, self.filename_state_attr))
        with open(os.path.join(self.save_path, self.filename_labels)) as f:
            self.labels = json.load(f)

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        items = [
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {
                k: torch.tensor(v[idx], dtype=matgl.float_th)
                for k, v in self.labels.items()
                if not isinstance(v[idx], str)
            },
        ]
        if self.include_line_graph:
            items.insert(2, self.line_graphs[idx])
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)
