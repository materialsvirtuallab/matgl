from __future__ import annotations

import os
import unittest

import numpy as np
from dgl.data.utils import split_dataset
from pymatgen.core import Molecule
from pymatgen.util.testing import PymatgenTest

from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list
from matgl.graph.data import (
    M3GNetDataset,
    MEGNetDataset,
    MGLDataLoader,
    _collate_fn,
)

module_dir = os.path.dirname(os.path.abspath(__file__))


class DatasetTest(PymatgenTest):
    def test_megnet_dataset(self):
        s1 = self.get_structure("LiFePO4")
        s2 = self.get_structure("BaNiO3")
        structures = [s1, s2]
        label = [-1.0, 2.0]
        element_types = get_element_list([s1, s2])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MEGNetDataset(structures=structures, converter=cry_graph, labels=label, label_name="label")
        g1, label1, state1 = dataset[0]
        g2, label2, state2 = dataset[1]
        self.assertTrue(label1 == label[0])
        self.assertTrue(g1.num_edges() == cry_graph.get_graph(s1)[0].num_edges())
        self.assertTrue(g1.num_nodes() == cry_graph.get_graph(s1)[0].num_nodes())
        self.assertTrue(g2.num_edges() == cry_graph.get_graph(s2)[0].num_edges())
        self.assertTrue(g2.num_nodes() == cry_graph.get_graph(s2)[0].num_nodes())

    def test_megnet_dataset_for_mol(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        methane = Molecule(["C", "H", "H", "H", "H"], coords)
        element_types = get_element_list([methane])
        mol_graph = Molecule2Graph(element_types=element_types, cutoff=1.5)
        label = [1.0, 2.0]
        structures = [methane, methane]
        dataset = MEGNetDataset(
            structures=structures, converter=mol_graph, labels=label, label_name="label", name="MolDataset"
        )
        g1, label1, state1 = dataset[0]
        g2, label2, state2 = dataset[1]
        self.assertTrue(label1 == label[0])
        self.assertTrue(g1.num_edges() == mol_graph.get_graph(methane)[0].num_edges())
        self.assertTrue(g1.num_nodes() == mol_graph.get_graph(methane)[0].num_nodes())
        self.assertTrue(g2.num_edges() == mol_graph.get_graph(methane)[0].num_edges())
        self.assertTrue(g2.num_nodes() == mol_graph.get_graph(methane)[0].num_nodes())

    def test_m3gnet_dataset(self):
        s1 = self.get_structure("LiFePO4")
        s2 = self.get_structure("BaNiO3")
        structures = [s1, s2]
        energies = [-1.0, 2.0]
        forces = [np.zeros((28, 3)).tolist(), np.zeros((10, 3)).tolist()]
        stresses = [np.zeros((3, 3)).tolist(), np.zeros((3, 3)).tolist()]
        element_types = get_element_list([s1, s2])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = M3GNetDataset(
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            energies=energies,
            forces=forces,
            stresses=stresses,
        )
        g1, l_g1, state1, energies_g1, forces_g1, stresses_g1 = dataset[0]
        g2, l_g2, state2, energies_g2, forces_g2, stresses_g2 = dataset[1]
        self.assertTrue(energies_g1 == energies[0])
        self.assertTrue(g1.num_edges() == cry_graph.get_graph(s1)[0].num_edges())
        self.assertTrue(g1.num_nodes() == cry_graph.get_graph(s1)[0].num_nodes())
        self.assertTrue(g2.num_edges() == cry_graph.get_graph(s2)[0].num_edges())
        self.assertTrue(g2.num_nodes() == cry_graph.get_graph(s2)[0].num_nodes())
        self.assertTrue(np.shape(forces_g1)[0], 28)
        self.assertTrue(np.shape(forces_g2)[0], 10)

    def test_megnet_dataloader(self):
        s1 = self.get_structure("LiFePO4")
        s2 = self.get_structure("BaNiO3")
        structures = [s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s2, s2, s2, s2, s2, s2, s2, s2, s2, s2]
        label = np.zeros(20)
        element_types = get_element_list([s1, s2])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MEGNetDataset(structures=structures, converter=cry_graph, labels=label, label_name="label")
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=_collate_fn,
            batch_size=2,
            num_workers=1,
        )
        self.assertTrue(len(train_loader) == 8)
        self.assertTrue(len(val_loader) == 1)
        self.assertTrue(len(test_loader) == 1)

    def test_megnet_dataloader_for_mol(self):
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
        mol_graph = Molecule2Graph(element_types=element_types, cutoff=1.5)
        dataset = MEGNetDataset(structures=structures, converter=mol_graph, labels=label, label_name="label")
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.6, 0.2, 0.2],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=_collate_fn,
            batch_size=2,
            num_workers=1,
        )
        self.assertTrue(len(train_loader) == 3)
        self.assertTrue(len(val_loader) == 1)
        self.assertTrue(len(test_loader) == 1)

    def test_m3gnet_dataloader(self):
        s1 = self.get_structure("LiFePO4")
        s2 = self.get_structure("BaNiO3")
        structures = [s1, s2, s1, s2, s1, s2, s1, s2, s1, s2, s1, s2, s1, s2, s1, s2, s1, s2, s1, s2]
        energies = np.zeros(20)
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        s = np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        stresses = [s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s]
        element_types = get_element_list([s1, s2])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = M3GNetDataset(
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            energies=energies,
            forces=forces,
            stresses=stresses,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=_collate_fn,
            batch_size=2,
            num_workers=1,
        )
        self.assertTrue(len(train_loader) == 8)
        self.assertTrue(len(val_loader) == 1)
        self.assertTrue(len(test_loader) == 1)

    @classmethod
    def tearDownClass(cls):
        for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
