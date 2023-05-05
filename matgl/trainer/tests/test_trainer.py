from __future__ import annotations

import os
import shutil
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import split_dataset
from pymatgen.util.testing import PymatgenTest
from torch.optim.lr_scheduler import ExponentialLR

from matgl.apps.pes import Potential
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import (
    M3GNetDataset,
    MEGNetDataset,
    MGLDataLoader,
    _collate_fn,
    _collate_fn_efs,
)
from matgl.layers._core import MLP
from matgl.models._m3gnet import M3GNet
from matgl.models._megnet import MEGNet
from matgl.trainer.m3gnet import M3GNetTrainer
from matgl.trainer.megnet import MEGNetTrainer

module_dir = os.path.dirname(os.path.abspath(__file__))


class MEGNetTrainerTest(PymatgenTest):
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

        g_sample, label_sample, attr_sample = dataset[0]

        node_feat = g_sample.ndata["node_type"]
        edge_feat = g_sample.edata["edge_attr"]
        node_embed = nn.Embedding(node_feat.shape[-1], 16)
        edge_embed = MLP([edge_feat.shape[-1], 16], activation=None)

        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=16,
            dim_state_embedding=2,
            nblocks=3,
            hidden_layer_sizes_input=(64, 32),
            hidden_layer_sizes_conv=(64, 64, 32),
            nlayers_set2set=1,
            niters_set2set=3,
            hidden_layer_sizes_outputput=[32, 16],
            is_classification=False,
            layer_node_embedding=node_embed,
            layer_edge_embedding=edge_embed,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        train_loss_function = F.mse_loss
        validate_loss_function = F.l1_loss

        trainer = MEGNetTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        trainer.train(
            num_epochs=2,
            train_loss_func=train_loss_function,
            val_loss_func=validate_loss_function,
            data_std=torch.zeros(1),
            data_mean=torch.zeros(1),
            train_loader=train_loader,
            val_loader=val_loader,
            logger_name="test_trainer.json",
        )


class M3GNetTrainerTest(PymatgenTest):
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
            collate_fn=_collate_fn_efs,
            batch_size=2,
            num_workers=1,
        )
        model = M3GNet(
            element_types=element_types,
            is_intensive=False,
        )

        ff = Potential(model=model)
        optimizer = torch.optim.Adam(ff.model.parameters(), lr=1.0e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        train_loss_function = F.mse_loss
        validate_loss_function = F.l1_loss

        trainer = M3GNetTrainer(potential=ff, optimizer=optimizer, scheduler=scheduler)

        trainer.train(
            num_epochs=2,
            train_loss=train_loss_function,
            val_loss=validate_loss_function,
            energy_weight=1.0,
            force_weight=1.0,
            stress_weight=0.1,
            train_loader=train_loader,
            val_loader=val_loader,
            logger_name="test_trainer.json",
        )

    @classmethod
    def tearDownClass(cls):
        for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "graph_attr.pt", "labels.json", "test_trainer.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
        shutil.rmtree("BestModel")
        shutil.rmtree("CheckPoints")


if __name__ == "__main__":
    unittest.main()
