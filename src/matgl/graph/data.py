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


def collate_fn_efs(batch, include_stress: bool = True, include_line_graph: bool = False):
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
    state_attr = torch.stack(state_attr)
    lat = torch.stack(lattices)
    if include_line_graph:
        return g, torch.squeeze(lat), l_g, state_attr, e, f, s
    return g, torch.squeeze(lat), state_attr, e, f, s


def collate_fn_efsm(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.tensor([d["energies"] for d in labels], dtype=torch.float32)
    f = torch.vstack([d["forces"] for d in labels])
    s = (
        torch.vstack([d["stresses"] for d in labels])  # type: ignore
        if "stresses" in labels[0]
        else torch.tensor(np.zeros(e.size(dim=0)), dtype=matgl.float_th)
    )
    m = torch.vstack([d["magmoms"] for d in labels])
    state_attr = torch.stack(state_attr)
    lat = torch.stack(lattices)
    return g, torch.squeeze(lat), l_g, state_attr, e, f, s, m


def collate_fn_efsm(batch):
    """Merge a list of dgl graphs to form a batch."""
    graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    l_g = dgl.batch(line_graphs)
    e = torch.tensor([d["energy"] for d in labels], dtype=torch.float32)
    f = torch.vstack([d["force"] for d in labels])
    s = torch.vstack([d["stress"] for d in labels])
    m = torch.vstack([d["magmom"] for d in labels])
    state_attr = torch.stack(state_attr)
    lat = torch.stack(lattices)
    return g, torch.squeeze(lat), l_g, state_attr, e, f, s, m


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
                collate_fn = partial(collate_fn_efs, include_stress=False)
            else:
                if "magmoms" not in train_data.dataset.labels:  # noqa: SIM108
                    collate_fn = collate_fn_efs
                else:
                    collate_fn = collate_fn_efsm

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
        name: str = "MGLDataset",
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        raw_dir: str | None = None,
        save_dir: str | None = None,
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
            name: name of dataset.
            graph_labels: state attributes.
            clear_processed: Whether to clear the stored structures after processing into graphs. Structures
                are not really needed after the conversion to DGL graphs and can take a significant amount of memory.
                Setting this to True will delete the structures from memory.
            save_cache: whether to save the processed dataset. The dataset can be reloaded from save_dir
                Default: True
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
                Default: ~/.dgl/
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
        super().__init__(name=name, raw_dir=raw_dir, save_dir=save_dir)

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
            state_attrs = torch.tensor(np.array(state_attrs))

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
            {k: torch.tensor(v[idx], dtype=matgl.float_th) for k, v in self.labels.items()},
        ]
        if self.include_line_graph:
            items.insert(2, self.line_graphs[idx])
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)


class CHGNetDataset(DGLDataset):
    """Create a CHGNet dataset including dgl graphs."""

    def __init__(
        self,
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        structures: list | None = None,
        labels: dict[str, list] | None = None,
        graph_labels: list[int | float] | None = None,
        filename_graphs: str = "dgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graphs: str = "dgl_line_graph.bin",
        filename_labels: str = "labels.json",
        filename_state_attr: str = "state_attr.pt",
        skip_label_keys: tuple[str] | None = None,
        name="CHGNETDataset",
        raw_dir: str | None = None,
        save_dir: str | None = None,
    ):
        """
        Args:
            converter: dgl graph converter
            threebody_cutoff: cutoff for three body
            structures: list of structures
            labels: target properties
            graph_labels: state attributes.
            filename_graphs: filename of dgl graphs
            filename_lattice: file name for storing lattice matrixs
            filename_line_graphs: filename of dgl line graphs
            filename_labels: filename of target labels file
            filename_state_attr: filename of state attributes.
            skip_label_keys: keys of labels to skip when getting dataset items.
            name: name of dataset
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data. Default: ~/.dgl/
            save_dir : directory to save the processed dataset. Default: same as raw_dir.
        """
        self.converter = converter
        self.threebody_cutoff = threebody_cutoff
        self.structures = structures
        self.labels = labels or {}

        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v

        self.graph_labels = graph_labels
        self.graphs = None
        self.lattices = None
        self.line_graphs = None
        self.state_attr = None
        self.skip_label_keys = skip_label_keys or ()
        self.filename_graphs = filename_graphs
        self.filename_lattice = filename_lattice
        self.filename_line_graphs = filename_line_graphs
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.graphs: list[dgl.DGLGraph] = []
        self.line_graphs: list[dgl.DGLGraph] = []
        self.state_attr: torch.Tensor = torch.tensor([])
        super().__init__(name=name, raw_dir=raw_dir, save_dir=save_dir)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not
        Args:
            :filename: Name of file storing dgl graphs
        Returns: True if file exists.
        """
        return all(
            os.path.exists(os.path.join(self.save_path, x))
            for x in [self.filename_graphs, self.filename_line_graphs, self.filename_state_attr, self.filename_labels]
        )

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)
        graphs, lattices, line_graphs, state_attrs = [], [], [], []

        for idx in trange(num_graphs):
            structure = self.structures[idx]
            graph, lattice, state_attr = self.converter.get_graph(structure)
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
            graph.ndata["pos"] = torch.tensor(structure.cart_coords)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            line_graph = create_line_graph(graph, self.threebody_cutoff, directed=True)
            line_graphs.append(line_graph)

        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()  # type: ignore
        else:
            state_attrs = torch.tensor(state_attrs)  # type: ignore

        self.graphs = graphs
        self.lattices = lattices
        self.line_graphs = line_graphs
        self.state_attr = state_attrs

    def save(self):
        """Save dgl graphs and labels."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        filepath_graphs = os.path.join(self.save_path, self.filename_graphs)
        torch.save(self.lattices, self.filename_lattice)
        filepath_line_graphs = os.path.join(self.save_path, self.filename_line_graphs)
        filepath_state_attr = os.path.join(self.save_path, self.filename_state_attr)
        filepath_labels = os.path.join(self.save_path, self.filename_labels)

        # save labels separately since save_graphs only supports tensors
        # and force/stress/magmom labels are of different shapes depending on the graph
        with open(filepath_labels, "w") as file:
            json.dump(self.labels, file)
        save_graphs(filepath_graphs, self.graphs)
        save_graphs(filepath_line_graphs, self.line_graphs)
        torch.save(self.state_attr, filepath_state_attr)

    def load(self):
        """Load CHGNet dataset from files."""
        filepath_graphs = os.path.join(self.save_path, self.filename_graphs)
        filepath_lattices = os.path.join(self.save_path, self.filename_lattice)
        filepath_line_graphs = os.path.join(self.save_path, self.filename_line_graphs)
        filepath_state_attr = os.path.join(self.save_path, self.filename_state_attr)
        filepath_labels = os.path.join(self.save_path, self.filename_labels)

        self.graphs, _ = load_graphs(filepath_graphs)
        self.line_graphs, _ = load_graphs(filepath_line_graphs)
        self.state_attr = torch.load(filepath_state_attr)

        if os.path.exists(filepath_lattices):
            self.lattices = torch.load(filepath_lattices)
        else:
            self.lattices = [g.edata['lattice'][0] for g in self.graphs]

        with open(filepath_labels) as f:
            self.labels = json.load(f)

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        labels = {
            k: torch.tensor(v[idx])
            if v[idx] is not None
            else torch.tensor(self.graphs[idx].num_nodes() * [torch.nan], dtype=matgl.float_th)[:, None]
            for k, v in self.labels.items()
            if k not in self.skip_label_keys
        }

        return (
            self.graphs[idx],
            self.lattices[idx],
            self.line_graphs[idx],
            self.state_attr[idx],
            labels,
        )

    def __len__(self):
        """Get size of dataset."""
        return len(self.state_attr)


class OOMCHGNetDataset(CHGNetDataset):
    def load(self):
        """Load CHGNet dataset from files."""
        os.path.join(self.save_path, self.filename_graphs)
        os.path.join(self.save_path, self.filename_line_graphs)
        filepath_state_attr = os.path.join(self.save_path, self.filename_state_attr)
        filepath_labels = os.path.join(self.save_path, self.filename_labels)

        self.state_attr = torch.load(filepath_state_attr)

        with open(filepath_labels) as f:
            self.labels = json.load(f)

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        idx = int(idx)
        graphs, _ = load_graphs(os.path.join(self.save_path, self.filename_graphs), [idx])
        line_graphs, _ = load_graphs(os.path.join(self.save_path, self.filename_line_graphs), [idx])

        graph = graphs[0]
        lattice = graph.edata['lattice'][0]
        line_graph = line_graphs[0]
        if 'frac_coords' not in graph.ndata.keys():
            graph.ndata['frac_coords'] = (
                torch.linalg.inv(lattice.T)
                @ graph.ndata['pos'].to(dtype=torch.float32).T
            ).T

        labels = {
            k: torch.tensor(v[idx])
            if v[idx] is not None
            else torch.tensor(graph.num_nodes() * [torch.nan], dtype=matgl.float_th)[:, None]
            for k, v in self.labels.items()
            if k not in self.skip_label_keys
        }

        return (
            graph,
            lattice,
            line_graph,
            self.state_attr[idx],
            labels,
        )

    def __len__(self):
        """Get size of dataset."""
        return len(self.labels["energy"])


class ChunkedCHGNetDataset(CHGNetDataset):
    def __init__(
        self,
        filename_graphs: str = "graphs_part%.bin",
        filename_line_graphs: str = "linegraphs_part%.bin",
        filename_labels: str = "labels_part%.json",
        num_chunks: int = 10,
        chunks_indices: list[int] | None = None,
        name="ChunkedCHGNETDataset",
        raw_dir: str | None = None,
        save_dir: str | None = None,
        skip_label_keys: tuple[str] | None = None,
    ):
        """
        Args:
            filename_graphs: filename of dgl graphs
            filename_line_graphs: filename of dgl line graphs
            filename_labels: filename of target labels file
            filename_state_attr: filename of state attributes
            num_chunks: number of chunks
            chunks_indices: indices of chunks to load.
        """
        if chunks_indices is None:
            chunks_indices = list(range(num_chunks))
        elif len(chunks_indices) != num_chunks:
            raise ValueError("Length of chunks_indices must be equal to num_chunks")

        self.chunks_indices = chunks_indices
        self.chunk_sizes = None

        super().__init__(
            filename_graphs=filename_graphs,
            filename_line_graphs=filename_line_graphs,
            filename_labels=filename_labels,
            name=name,
            raw_dir=raw_dir,
            save_dir=save_dir,
            skip_label_keys=skip_label_keys,
        )

    @property
    def num_chunks(self):
        return len(self.chunks_indices)

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not.

        Returns: True if file exists.
        """
        for ind in self.chunks_indices:
            if not all(
                os.path.exists(os.path.join(self.save_path, x.replace("%", str(ind))))
                for x in [self.filename_graphs, self.filename_line_graphs, self.filename_labels]
            ):
                return False
        return True

    def process(self) -> tuple:
        raise NotImplementedError("ChunkedCHGNetDataset does not support processing data.")

    def save(self):
        """Save dgl graphs and labels."""

    def load(self):
        """Load only CHGNet dataset labels."""
        self.labels = defaultdict(list)
        self.chunk_sizes = []

        for i in trange(self.num_chunks, desc="Loading labels"):
            ind = self.chunks_indices[i]
            filepath_labels = os.path.join(self.save_path, self.filename_labels.replace("%", str(ind)))

            with open(filepath_labels) as f:
                labels = json.load(f)

            for k, v in labels.items():
                self.labels[k].extend(v)

            self.chunk_sizes.append(len(labels[k]))

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        idx_ = idx
        for chunk_idx, chunk_size in zip(self.chunks_indices, self.chunk_sizes):
            if idx_ < chunk_size:
                break
            idx_ -= chunk_size

        filepath_graphs = os.path.join(self.save_path, self.filename_graphs.replace("%", str(chunk_idx)))
        filepath_line_graphs = os.path.join(self.save_path, self.filename_line_graphs.replace("%", str(chunk_idx)))

        idx_ = int(idx_)
        graphs, _ = load_graphs(filepath_graphs, [idx_])
        line_graphs, _ = load_graphs(filepath_line_graphs, [idx_])

        graph = graphs[0]
        line_graph = line_graphs[0]
        labels = {
            k: torch.tensor(v[idx])
            if v[idx] is not None
            else torch.tensor(graph.num_nodes() * [torch.nan], dtype=matgl.float_th)[:, None]
            for k, v in self.labels.items()
            if k not in self.skip_label_keys
        }

        return graph, line_graph, torch.tensor([0, 0]), labels

    def __len__(self):
        """Get size of dataset."""
        return sum(self.chunk_sizes)
