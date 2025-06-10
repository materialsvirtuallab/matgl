"""Tools to construct a dataset of DGL graphs."""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch_geometric.loader as pyg_loader
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data, Dataset
from tqdm import trange

import matgl
from matgl.graph._compute_pyg import compute_pair_vector_and_distance_pyg, create_line_graph

if TYPE_CHECKING:
    from collections.abc import Callable

    from matgl.graph._converters_pyg import GraphConverter


def ensure_batch_attribute(data: Data) -> Data:
    """
    Ensure a PyG Data object has a batch attribute.

    Args:
        data: PyG Data object.

    Returns:
        Data object with batch attribute set.
    """
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
    return data


def split_dataset(self, frac_list=None, shuffle=False, random_state=None):
    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    num_graphs = len(self)
    num_train = int(frac_list[0] * num_graphs)
    num_val = int(frac_list[1] * num_graphs)

    indices = torch.randperm(num_graphs) if shuffle else torch.arange(num_graphs)
    train_idx = indices[:num_train]
    val_idx = indices[num_train : num_train + num_val]
    test_idx = indices[num_train + num_val :]

    return (Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx))


def collate_fn_graph(batch, include_line_graph: bool = False, multiple_values_per_target: bool = False):
    """
    Merge a list of PyG graphs to form a batch.

    Args:
        batch: List of tuples, each containing (graph, lattice, [line_graph,] state_attr, labels).
        include_line_graph: Whether to include line graphs.
        multiple_values_per_target: Whether labels are tensors (True) or scalars (False).

    Returns:
        Tuple containing:
        - g: PyG Data (single graph) or Batch (multiple graphs) object.
        - lat: Lattice tensor (batch_size, 3, 3) or (3, 3) for single graph.
        - l_g: Batched line graph (Batch object, if include_line_graph=True, else None).
        - state_attr: Stacked state attributes (batch_size, state_dim).
        - labels: Stacked or tensorized labels (batch_size, ...) or (batch_size,).
    """
    l_g = None
    if include_line_graph:
        graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch, strict=False))
    else:
        graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))

    g = Batch.from_data_list(graphs)  # Batch main graphs
    labels = (
        torch.vstack([next(iter(d.values())) for d in labels])  # type:ignore[assignment]
        if multiple_values_per_target
        else torch.tensor([next(iter(d.values())) for d in labels], dtype=matgl.float_th)
    )
    state_attr = torch.stack(state_attr)  # type:ignore[assignment]
    lat = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))
    if include_line_graph:
        l_g = Batch.from_data_list(line_graphs)  # Batch line graphs
        return g, lat, l_g, state_attr, labels
    return g, lat, state_attr, labels


def collate_fn_pes(batch, include_stress: bool = True, include_line_graph: bool = False, include_magmom: bool = False):
    """Merge a list of PyG Data objects to form a batch.

    Args:
        batch: List of tuples, each containing (graph, lattices, [line_graphs,] state_attr, labels)
        include_stress (bool): Whether to include stress tensors in the output
        include_line_graph (bool): Whether to include line graphs in the batch
        include_magmom (bool): Whether to include magnetic moments in the output

    Returns:
        Tuple containing:
        - g: Batched PyG graph (Batch object)
        - lat: Stacked lattice tensors (batch_size, ...)
        - l_g: Batched line graph (Batch object, if include_line_graph=True, else None)
        - state_attr: Stacked state attributes (batch_size, state_dim)
        - e: Energies (batch_size,)
        - f: Forces (num_atoms, 3)
        - s: Stresses (batch_size, 6) or zeros if include_stress=False
        - m: Magnetic moments (batch_size, ...) or zeros if include_magmom=False
    """
    l_g = None
    if include_line_graph:
        graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch, strict=False))
        l_g = Batch.from_data_list(line_graphs)  # Batch line graphs
    else:
        graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))

    g = Batch.from_data_list(graphs)  # Batch main graphs
    e = torch.tensor([d["energies"] for d in labels], dtype=matgl.float_th)
    f = torch.vstack([d["forces"] for d in labels])
    s = (
        torch.vstack([d["stresses"] for d in labels])
        if include_stress
        else torch.zeros(e.size(0), dtype=matgl.float_th)
    )
    m = torch.vstack([d["magmoms"] for d in labels]) if include_magmom else torch.zeros(e.size(0), dtype=matgl.float_th)
    state_attr = torch.stack(state_attr)  # type:ignore[assignment]
    lat = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))

    if include_line_graph:
        if include_magmom:
            return g, lat.squeeze(), l_g, state_attr, e, f, s, m
        return g, lat.squeeze(), l_g, state_attr, e, f, s
    return g, lat.squeeze(), state_attr, e, f, s


def MGLDataLoader(
    train_data: MGLDataset,
    val_data: MGLDataset,
    collate_fn: Callable | None = None,
    test_data: MGLDataset | None = None,
    **kwargs,
) -> tuple[pyg_loader.DataLoader, ...]:
    """Dataloader for MatGL training in PyTorch Geometric.

    Args:
        train_data (Dataset): Training dataset (PyG Dataset or subset).
        val_data (Dataset): Validation dataset (PyG Dataset or subset).
        collate_fn (Callable, optional): Collate function for batching.
        test_data (Dataset, optional): Test dataset (PyG Dataset or subset). Defaults to None.
        **kwargs: Pass-through kwargs to torch_geometric.loader.DataLoader. Common ones you may want to set are
            batch_size, num_workers, pin_memory, and generator.

    Returns:
        Tuple[DataLoader, ...]: Train, validation, and test data loaders. Test data loader is None if test_data is None.
    """
    train_loader = pyg_loader.DataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = pyg_loader.DataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    if test_data is not None:
        test_loader = pyg_loader.DataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


class MGLDataset(Dataset):
    """Create a dataset including PyTorch Geometric graphs."""

    def __init__(
        self,
        filename: str = "pyg_graph.pt",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "pyg_line_graph.pt",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        include_line_graph: bool = False,
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        directed_line_graph: bool = False,
        structures: list | None = None,
        labels: dict[str, list] | None = None,
        root: str = "MGLDataset",
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Args:
            filename: File name for storing PyG graphs.
            filename_lattice: File name for storing lattice matrices.
            filename_line_graph: File name for storing PyG line graphs.
            filename_state_attr: File name for storing state attributes.
            filename_labels: File name for storing labels.
            include_line_graph: Whether to include line graphs.
            converter: Graph converter for PyG (converts structures to Data objects).
            threebody_cutoff: Cutoff for three-body interactions.
            directed_line_graph: Whether to create a directed line graph (CHGNet) or undirected (M3GNet).
            structures: Pymatgen structures.
            labels: Targets as a dict of {name: list of values}.
            root: Root directory where the dataset should be saved.
            transform: A function/transform that takes in a Data or HeteroData object and returns a transformed version.
            pre_transform: A function/transform that takes in a Data or HeteroData object
                           and returns a transformed version.
            pre_filter: A function that takes in a Data or HeteroData object and returns a boolean value.
            directory_name: Name of the directory to store the dataset.
            graph_labels: State attributes.
            clear_processed: Whether to clear stored structures after processing.
            save_cache: Whether to save the processed dataset.
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
        self.root = root

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load or process data

        if self.has_cache():
            self.load()

        shutil.rmtree(self.root + "/processed/")

    def has_cache(self) -> bool:
        """Check if the processed files exist."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        if self.include_line_graph:
            files_to_check.append(self.filename_line_graph)
        return all(os.path.exists(os.path.join(self.root, f)) for f in files_to_check)

    def process(self):
        """Convert Pymatgen structures into PyG Data objects."""
        if self.has_cache():
            pass
        else:
            num_graphs = len(self.structures)
            graphs, lattices, line_graphs, state_attrs = [], [], [], []

            for idx in trange(num_graphs):
                structure = self.structures[idx]
                # Converter returns (Data, lattice, state_attr)
                data, lattice, state_attr = self.converter.get_graph(structure)

                # Add position coordinates
                data.pos = torch.tensor(structure.cart_coords, dtype=matgl.float_th)

                # Compute bond vectors and distances
                bond_vec, bond_dist = compute_pair_vector_and_distance_pyg(data)
                data.bond_vec = bond_vec
                data.bond_dist = bond_dist

                # Compute PBC offsets (if not already in edge_attr)
                if not hasattr(data, "edge_attr") or data.edge_attr is None:
                    data.pbc_offshift = torch.zeros_like(bond_vec)
                else:
                    data.pbc_offshift = torch.matmul(data.edge_attr, lattice[0])

                graphs.append(data)
                lattices.append(lattice)
                state_attrs.append(state_attr)

                if self.include_line_graph:
                    line_graph = create_line_graph(data, self.threebody_cutoff, directed=self.directed_line_graph)
                    # Remove unnecessary attributes from line graph
                    for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                        if hasattr(line_graph, name):
                            delattr(line_graph, name)
                    line_graphs.append(line_graph)

                # Remove temporary attributes
                del data.pos
                if hasattr(data, "pbc_offshift"):
                    del data.pbc_offshift

            if self.graph_labels is not None:
                state_attrs = torch.tensor(self.graph_labels, dtype=torch.long)
            else:
                state_attrs = torch.tensor(np.array(state_attrs), dtype=matgl.float_th)

            if self.clear_processed:
                del self.structures
                self.structures = []
            self.graphs = graphs
            self.lattices = lattices
            self.state_attr = state_attrs
            self.line_graphs = line_graphs if self.include_line_graph else []

            # Validate loaded or processed data
            if not self.graphs:
                raise ValueError("Dataset is empty after loading or processing")
            self.save()

    def save(self):
        """Save PyG graphs and labels to processed_dir."""
        if not self.save_cache:
            return

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if self.labels:
            with open(os.path.join(self.root, self.filename_labels), "w") as file:
                json.dump(self.labels, file)

        torch.save(self.graphs, os.path.join(self.root, self.filename))
        torch.save(self.lattices, os.path.join(self.root, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.root, self.filename_state_attr))
        if self.include_line_graph:
            torch.save(self.line_graphs, os.path.join(self.root, self.filename_line_graph))

    def load(self):
        """Load PyG graphs from files."""
        self.graphs = torch.load(os.path.join(self.root, self.filename))
        self.lattices = torch.load(os.path.join(self.root, self.filename_lattice))
        self.state_attr = torch.load(os.path.join(self.root, self.filename_state_attr))
        if self.include_line_graph:
            self.line_graphs = torch.load(os.path.join(self.root, self.filename_line_graph))
        else:
            self.line_graphs = []
        with open(os.path.join(self.root, self.filename_labels)) as f:
            self.labels = json.load(f)

    def __getitem__(self, idx: int) -> tuple:
        """Get graph and associated data with idx."""
        if idx >= len(self.graphs):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.graphs)} graphs")
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

    def __len__(self) -> int:
        """Get size of dataset."""
        return len(self.graphs)

    @property
    def processed_file_names(self) -> list[str]:
        """List of processed file names."""
        return []

    @property
    def raw_file_names(self) -> list[str]:
        """List of raw file names (not used in this case)."""
        return []
