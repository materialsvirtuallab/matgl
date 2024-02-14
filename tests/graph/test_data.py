from __future__ import annotations

import os

# This function is used for M3GNet property dataset
from functools import partial

import numpy as np
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn, collate_fn_efs
from pymatgen.core import Molecule

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestDataset:
    def test_megnet_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        label = [-1.0, 2.0]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(structures=structures, converter=cry_graph, labels={"label": label}, clear_processed=True)
        g1, lat1, state1, label1 = dataset[0]
        g2, lat2, state2, label2 = dataset[1]
        assert label1["label"] == label[0]
        assert g1.num_edges() == cry_graph.get_graph(LiFePO4)[0].num_edges()
        assert g1.num_nodes() == cry_graph.get_graph(LiFePO4)[0].num_nodes()
        assert g2.num_edges() == cry_graph.get_graph(BaNiO3)[0].num_edges()
        assert g2.num_nodes() == cry_graph.get_graph(BaNiO3)[0].num_nodes()
        assert np.allclose(lat1.detach().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().numpy(), structures[1].lattice.matrix)
        # Check that structures are indeed cleared.
        assert len(dataset.structures) == 0
        self.teardown_class()

    def test_megnet_dataset_with_graph_label(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        label = [-1.0, 2.0]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        graph_label = [1, 0]
        dataset_with_graph_label = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels={"label": label},
            clear_processed=True,
            graph_labels=graph_label,
        )
        g1, lat1, state1, label1 = dataset_with_graph_label[0]
        g2, lat2, state2, label2 = dataset_with_graph_label[1]

        assert state1.detach().numpy() == graph_label[0]
        assert state2.detach().numpy() == graph_label[1]
        self.teardown_class()

    def test_megnet_dataset_with_graph_label_float(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        label = [-1.0, 2.0]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        graph_label = [[1.0, 2.0], [0.0, 1.0]]
        dataset_with_graph_label = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels={"label": label},
            clear_processed=True,
            graph_labels=graph_label,
        )
        g1, lat1, state1, label1 = dataset_with_graph_label[0]
        g2, lat2, state2, label2 = dataset_with_graph_label[1]

        assert state1.detach().numpy().tolist() == graph_label[0]
        assert state2.detach().numpy().tolist() == graph_label[1]

    def test_load_megenet_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        label = [-1.0, 2.0]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset()
        g1, lat1, state1, label1 = dataset[0]
        assert label1["label"] == label[0]
        assert g1.num_edges() == cry_graph.get_graph(LiFePO4)[0].num_edges()
        assert g1.num_nodes() == cry_graph.get_graph(LiFePO4)[0].num_nodes()
        assert np.allclose(lat1.detach().numpy(), structures[0].lattice.matrix)
        self.teardown_class()

    def test_megnet_dataset_for_mol(self, CH4):
        element_types = get_element_list([CH4])
        mol_graph = Molecule2Graph(element_types=element_types, cutoff=1.5)
        label = [1.0, 2.0]
        structures = [CH4, CH4]
        dataset = MGLDataset(structures=structures, converter=mol_graph, labels={"label": label}, name="MolDataset")
        g1, lat1, state1, label1 = dataset[0]
        g2, lat2, state2, label2 = dataset[1]
        assert label1["label"] == label[0]
        assert g1.num_edges() == mol_graph.get_graph(CH4)[0].num_edges()
        assert g1.num_nodes() == mol_graph.get_graph(CH4)[0].num_nodes()
        assert g2.num_edges() == mol_graph.get_graph(CH4)[0].num_edges()
        assert g2.num_nodes() == mol_graph.get_graph(CH4)[0].num_nodes()
        assert np.allclose(lat1.detach().numpy(), np.expand_dims(np.identity(3), axis=0))
        self.teardown_class()

    def test_mgl_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        energies = [-1.0, 2.0]
        forces = [np.zeros((28, 3)).tolist(), np.zeros((10, 3)).tolist()]
        stresses = [np.zeros((3, 3)).tolist(), np.zeros((3, 3)).tolist()]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            include_line_graph=True,
            labels={"energies": energies, "forces": forces, "stresses": stresses},
            clear_processed=True,
        )
        g1, lat1, l_g1, state1, pes1 = dataset[0]
        g2, lat2, l_g2, state2, pes2 = dataset[1]
        assert pes1["energies"] == energies[0]
        assert g1.num_edges() == cry_graph.get_graph(LiFePO4)[0].num_edges()
        assert g1.num_nodes() == cry_graph.get_graph(LiFePO4)[0].num_nodes()
        assert g2.num_edges() == cry_graph.get_graph(BaNiO3)[0].num_edges()
        assert g2.num_nodes() == cry_graph.get_graph(BaNiO3)[0].num_nodes()
        assert np.shape(pes1["forces"])[0], 28
        assert np.shape(pes2["forces"])[0], 10
        assert np.allclose(lat1.detach().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().numpy(), structures[1].lattice.matrix)
        # Check that structures are indeed cleared.
        assert len(dataset.structures) == 0

    def test_load_mgl_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(include_line_graph=True)
        dataset.load()
        g1, lat1, l_g1, state1, pes1 = dataset[0]
        g2, lat2, l_g2, state2, pes2 = dataset[1]
        assert pes1["energies"] == -1.0
        assert g1.num_edges() == cry_graph.get_graph(LiFePO4)[0].num_edges()
        assert g1.num_nodes() == cry_graph.get_graph(LiFePO4)[0].num_nodes()
        assert g2.num_edges() == cry_graph.get_graph(BaNiO3)[0].num_edges()
        assert g2.num_nodes() == cry_graph.get_graph(BaNiO3)[0].num_nodes()
        assert np.shape(pes1["forces"])[0], 28
        assert np.shape(pes2["forces"])[0], 10
        assert np.allclose(lat1.detach().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().numpy(), structures[1].lattice.matrix)

    def test_mgl_property_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        labels = [1.0, -2.0]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            filename_labels="eform.json",
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            include_line_graph=True,
            labels={"Eform_per_atom": labels},
        )
        g1, lat1, l_g1, state1, label1 = dataset[0]
        g2, lat2, l_g2, state2, label2 = dataset[1]
        assert label1["Eform_per_atom"] == labels[0]
        assert g1.num_edges() == cry_graph.get_graph(LiFePO4)[0].num_edges()
        assert g1.num_nodes() == cry_graph.get_graph(LiFePO4)[0].num_nodes()
        assert g2.num_edges() == cry_graph.get_graph(BaNiO3)[0].num_edges()
        assert g2.num_nodes() == cry_graph.get_graph(BaNiO3)[0].num_nodes()
        assert np.allclose(lat1.detach().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().numpy(), structures[1].lattice.matrix)
        dataset.save()

        dataset = MGLDataset(
            filename_labels="eform.json",
            include_line_graph=True,
        )
        g1, lat1, l_g1, state1, label1 = dataset[0]
        g2, lat2, l_g2, state2, label2 = dataset[1]
        assert label1["Eform_per_atom"] == labels[0]
        assert g1.num_edges() == cry_graph.get_graph(LiFePO4)[0].num_edges()
        assert g1.num_nodes() == cry_graph.get_graph(LiFePO4)[0].num_nodes()
        assert g2.num_edges() == cry_graph.get_graph(BaNiO3)[0].num_edges()
        assert g2.num_nodes() == cry_graph.get_graph(BaNiO3)[0].num_nodes()

        self.teardown_class()

    def test_mgl_property_dataset_with_graph_label(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        labels = [1.0, -2.0]
        graph_label = [0, 1]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            filename_labels="eform.json",
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            include_line_graph=True,
            labels={"Eform_per_atom": labels},
            graph_labels=graph_label,
        )
        g1, lat1, l_g1, state1, label1 = dataset[0]
        g2, lat2, l_g2, state2, label2 = dataset[1]
        assert state1.detach().numpy() == graph_label[0]
        assert state2.detach().numpy() == graph_label[1]
        self.teardown_class()

    def test_megnet_dataloader(self, LiFePO4, BaNiO3):
        structures = [LiFePO4] * 10 + [BaNiO3] * 10
        label = np.zeros(20).tolist()
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(structures=structures, converter=cry_graph, labels={"label": label})
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
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1

        train_loader_new, val_loader_new = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader_new) == 8
        assert len(val_loader_new) == 1
        self.teardown_class()

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
        label = np.zeros(10).tolist()
        element_types = get_element_list([m1])
        mol_graph = Molecule2Graph(element_types=element_types, cutoff=1.5)
        dataset = MGLDataset(structures=structures, converter=mol_graph, labels={"label": label})
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
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) == 3
        assert len(val_loader) == 1
        assert len(test_loader) == 1
        self.teardown_class()

    def test_mgl_dataloader(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        energies = np.zeros(20).tolist()
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        s = np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        stresses = [s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s]
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            include_line_graph=True,
            labels={"energies": energies, "forces": forces, "stresses": stresses},
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
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1
        self.teardown_class()

    def test_mgl_dataloader_without_stresses(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        energies = np.zeros(20).tolist()
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            include_line_graph=True,
            labels={"energies": energies, "forces": forces},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        my_collate_fn = partial(collate_fn_efs, include_stress=False)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1
        self.teardown_class()

    def test_mgl_property_dataloader(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        e_form = np.zeros(20)
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            threebody_cutoff=4.0,
            include_line_graph=True,
            labels={"EForm": e_form},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        my_collate_fn = partial(collate_fn, include_line_graph=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=1,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1

    @classmethod
    def teardown_class(cls):
        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json", "eform.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
