import os

import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from torch.utils.data import DataLoader
from tqdm import trange


def _collate_fn(batch):
    """
    Merge a list of samples to form a mini batch for dgl graphs
    """
    graphs, labels = map(list, zip(*batch))
    g = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return g, labels


def MEGNetDataLoader(train_data, val_data, test_data, collate_fn, batch_size, num_workers):
    """
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
    Args:
    structures: Pymatgen strutcure
    labels: property values
    label: label name
    cry_graph: Pmg2Graph
    """

    def __init__(self, structures, labels, label_name, cry_graph):
        self.cry_graph = cry_graph
        self.structures = structures
        self.labels = torch.FloatTensor(labels)
        self.label_name = label_name
        self.path = os.getcwd()
        super().__init__(name="MEGNetDataset")

    def has_cache(self):
        graph_path = str(self.path) + "/dgl_graph.bin"
        return os.path.exists(graph_path)

    def process(self):
        self.graphs = self._load_graph()

    def _load_graph(self):
        num_graphs = self.labels.shape[0]
        graphs = []

        for idx in trange(num_graphs):
            structure = self.structures[idx]
            graph, state_attr = self.cry_graph.get_graph_from_structure(structure=structure)
            graphs.append(graph)
        return graphs

    def save(self):
        graph_path = str(self.path) + "/dgl_graph.bin"
        labels_with_key = {self.label_name: self.labels}
        save_graphs(graph_path, self.graphs, labels_with_key)

    def load(self):
        graph_path = str(self.path) + "/dgl_graph.bin"
        self.graphs, label_dict = load_graphs(graph_path)
        self.label = torch.stack([label_dict[key] for key in self.label_keys], dim=1)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)
