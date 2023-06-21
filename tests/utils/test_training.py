from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.backends.mps
from dgl.data.utils import split_dataset

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import M3GNetDataset, MEGNetDataset, MGLDataLoader, collate_fn, collate_fn_efs
from matgl.models import M3GNet, MEGNet
from matgl.utils.training import ModelLightningModule, PotentialLightningModule

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestModelTrainer:
    def test_megnet_training(self, LiFePO4, BaNiO3):
        structures = [LiFePO4] * 5 + [BaNiO3] * 5
        label = [-2] * 5 + [-3] * 5  # Artificial dataset.
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
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=10, accelerator="cpu" if torch.backends.mps.is_available() else "auto")

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

    def test_m3gnet_training(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 5
        energies = [-2.0, -3.0] * 5
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist()] * len(structures)
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = M3GNetDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
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
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=10, accelerator="cpu" if torch.backends.mps.is_available() else "auto")
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

    @classmethod
    def teardown_class(cls):
        for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

        shutil.rmtree("lightning_logs")
