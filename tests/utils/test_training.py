from __future__ import annotations

import os
import shutil

import numpy as np
import pytorch_lightning as pl
from dgl.data.utils import split_dataset

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import M3GNetDataset, MEGNetDataset, MGLDataLoader, collate_fn, collate_fn_efs
from matgl.models import M3GNet, MEGNet
from matgl.utils.training import ModelLightningModule, PotentialLightningModule

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestModelTrainer:
    def test_megnet_training(self, LiFePO4, BaNiO3):
        structures = [LiFePO4] * 10 + [BaNiO3] * 10
        label = np.zeros(20)
        element_types = get_element_list([LiFePO4, BaNiO3])
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
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=1,
        )

        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=2,
            nblocks=3,
            hidden_layer_sizes_input=(64, 32),
            hidden_layer_sizes_conv=(64, 64, 32),
            nlayers_set2set=1,
            niters_set2set=3,
            hidden_layer_sizes_outputput=[32, 16],
            is_classification=False,
        )

        lit_model = ModelLightningModule(model=model)
        trainer = pl.Trainer(max_epochs=2)
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def test_m3gnet_training(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        energies = np.zeros(20)
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        s = np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        stresses = [s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s]
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = M3GNetDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=cry_graph,
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
            collate_fn=collate_fn_efs,
            batch_size=2,
            num_workers=1,
        )
        model = M3GNet(
            element_types=element_types,
            is_intensive=False,
        )
        lit_model = PotentialLightningModule(model=model)
        trainer = pl.Trainer(max_epochs=2)
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    @classmethod
    def teardown_class(cls):
        for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

        shutil.rmtree("lightning_logs")
