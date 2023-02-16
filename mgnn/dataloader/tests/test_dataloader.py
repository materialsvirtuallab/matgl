import os
import unittest

import numpy as np
from dgl.data.utils import split_dataset
from pymatgen.core import Molecule
from pymatgen.util.testing import PymatgenTest

from mgnn.dataloader.dataset import MEGNetDataLoader, MEGNetDataset, _collate_fn
from mgnn.graph.converters import Pmg2Graph, get_element_list

module_dir = os.path.dirname(os.path.abspath(__file__))


class MEGNetDatasetTest(PymatgenTest):
    def test_MEGNetDataset(self):
        s1 = self.get_structure("LiFePO4")
        s2 = self.get_structure("BaNiO3")
        structures = [s1, s2]
        label = [-1.0, 2.0]
        element_types = get_element_list([s1, s2])
        cry_graph = Pmg2Graph(element_types=element_types, cutoff=4.0)
        dataset = MEGNetDataset(structures=structures, converter=cry_graph, labels=label, label_name="label")
        g1, label1, state1 = dataset[0]
        g2, label2, state2 = dataset[1]
        self.assertTrue(label1 == label[0])
        self.assertTrue(g1.num_edges() == cry_graph.get_graph_from_structure(s1)[0].num_edges())
        self.assertTrue(g1.num_nodes() == cry_graph.get_graph_from_structure(s1)[0].num_nodes())
        self.assertTrue(g2.num_edges() == cry_graph.get_graph_from_structure(s2)[0].num_edges())
        self.assertTrue(g2.num_nodes() == cry_graph.get_graph_from_structure(s2)[0].num_nodes())

    def test_MEGNetDataset_for_mol(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        methane = Molecule(["C", "H", "H", "H", "H"], coords)
        element_types = get_element_list([methane])
        mol_graph = Pmg2Graph(element_types=element_types, cutoff=1.5)
        label = [1.0, 2.0]
        structures = [methane, methane]
        dataset = MEGNetDataset(
            structures=structures, converter=mol_graph, labels=label, label_name="label", name="MolDataset"
        )
        g1, label1, state1 = dataset[0]
        g2, label2, state2 = dataset[1]
        self.assertTrue(label1 == label[0])
        self.assertTrue(g1.num_edges() == mol_graph.get_graph_from_molecule(methane)[0].num_edges())
        self.assertTrue(g1.num_nodes() == mol_graph.get_graph_from_molecule(methane)[0].num_nodes())
        self.assertTrue(g2.num_edges() == mol_graph.get_graph_from_molecule(methane)[0].num_edges())
        self.assertTrue(g2.num_nodes() == mol_graph.get_graph_from_molecule(methane)[0].num_nodes())

    def test_MEGNetDataLoader(self):
        s1 = self.get_structure("LiFePO4")
        s2 = self.get_structure("BaNiO3")
        structures = [s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s2, s2, s2, s2, s2, s2, s2, s2, s2, s2]
        label = np.zeros(20)
        element_types = get_element_list([s1, s2])
        cry_graph = Pmg2Graph(element_types=element_types, cutoff=4.0)
        dataset = MEGNetDataset(structures=structures, converter=cry_graph, labels=label, label_name="label")
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MEGNetDataLoader(train_data, val_data, test_data, _collate_fn, 2, 1)
        self.assertTrue(len(train_loader) == 8)
        self.assertTrue(len(val_loader) == 1)
        self.assertTrue(len(test_loader) == 1)

    def test_MEGNetDataLoader_for_mol(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        m1 = Molecule(["C", "H", "H", "H", "H"], coords)
        structures = [m1, m1, m1, m1, m1, m1, m1, m1, m1, m1]
        label = np.zeros(10)
        element_types = get_element_list([m1])
        mol_graph = Pmg2Graph(element_types=element_types, cutoff=1.5)
        dataset = MEGNetDataset(structures=structures, converter=mol_graph, labels=label, label_name="label")
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.6, 0.2, 0.2],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MEGNetDataLoader(train_data, val_data, test_data, _collate_fn, 2, 1)
        self.assertTrue(len(train_loader) == 3)
        self.assertTrue(len(val_loader) == 1)
        self.assertTrue(len(test_loader) == 1)


if __name__ == "__main__":
    unittest.main()
