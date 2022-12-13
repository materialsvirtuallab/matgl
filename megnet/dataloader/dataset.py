"""
Tools to construct a dataloader for DGL grphs
"""
import os

import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate_fn(batch):
    """
    Merge a list of dgl graphs to form a batch
    """
    graphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return g, labels


def MEGNetDataLoader(train_data, val_data, test_data, collate_fn, batch_size, num_workers):
    """
    Dataloader for MEGNet training
    Args:
    train_data: training data
    val_data: validation data
    test_data: testing data
    collate_fn:
    """
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
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

    def __init__(self, structures, labels, label_name, crystal2graph, name="MEGNETDataset"):
        """
        Args:
        structures: Pymatgen strutcure
        labels: property values
        label: label name
        crystal2graph: Transformer for converting crystals to DGL graphs, e.g., Pmg2Graph.
        """
        self.crystal2graph = crystal2graph
        self.structures = structures
        self.labels = torch.FloatTensor(labels)
        self.label_name = label_name
        super().__init__(name=name)

    def has_cache(self, filename="dgl_graph.bin"):
        """
        Check if the dgl_graph.bin exists or not
        Args:
        :filename: Name of file storing dgl graphs
        """
        return os.path.exists(filename)

    def process(self):
        """
        Convert Pymatgen structure into dgl graphs
        """
        self.graphs = [
            self.crystal2graph.get_graph_from_structure(structure=structure)[0] for structure in tqdm(self.structures)
        ]
        return self.graphs

    def save(self, filename="dgl_graph.bin"):
        """
        Save dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        """
        labels_with_key = {self.label_name: self.labels}
        save_graphs(filename, self.graphs, labels_with_key)

    def load(self, filename="dgl_graph.bin"):
        """
        Load dgl graphs
        Args:
        :filename: Name of file storing dgl graphs
        """
        self.graphs, label_dict = load_graphs(filename)
        self.label = torch.stack([label_dict[key] for key in self.label_keys], dim=1)

    def __getitem__(self, idx):
        """
        Get graph and label with idx
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """
        Get size of dataset
        """
        return len(self.graphs)
