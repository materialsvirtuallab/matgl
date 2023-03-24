from __future__ import annotations

import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from dgl.data.utils import split_dataset
from pymatgen.util.testing import PymatgenTest
from torch.optim.lr_scheduler import ExponentialLR

from matgl.dataloader.dataset import (
    M3GNetDataset,
    MEGNetDataset,
    MGLDataLoader,
    _collate_fn,
    _collate_fn_efs,
)
from matgl.graph.converters import Pmg2Graph, get_element_list
from matgl.layers.core import MLP
from matgl.models.m3gnet import M3GNet
from matgl.models.megnet import MEGNet
from matgl.models.potential import Potential
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
        cry_graph = Pmg2Graph(element_types=element_types, cutoff=4.0)
        dataset = MEGNetDataset(structures=structures, converter=cry_graph, labels=label, label_name="label")
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MGLDataLoader(train_data, val_data, test_data, _collate_fn, 2, 1)

        g_sample, label_sample, attr_sample = dataset[0]

        node_feat = g_sample.ndata["attr"]
        edge_feat = g_sample.edata["edge_attr"]
        attrs = attr_sample
        node_embed = MLP([node_feat.shape[-1], 16], activation=None)
        edge_embed = MLP([edge_feat.shape[-1], 16], activation=None)
        attr_embed = MLP([attrs.shape[-1], 16], activation=None)

        model = MEGNet(
            in_dim=16,
            num_blocks=3,
            hiddens=[64, 32],
            conv_hiddens=[64, 64, 32],
            s2s_num_layers=1,
            s2s_num_iters=3,
            output_hiddens=[32, 16],
            is_classification=False,
            node_embed=node_embed,
            edge_embed=edge_embed,
            attr_embed=attr_embed,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

        train_loss_function = F.mse_loss
        validate_loss_function = F.l1_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        print(model)

        trainer = MEGNetTrainer(
            model=model,
            optimizer=optimizer,
        )

        trainer.train(
            device=device,
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
        cry_graph = Pmg2Graph(element_types=element_types, cutoff=5.0)
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
        train_loader, val_loader, test_loader = MGLDataLoader(train_data, val_data, test_data, _collate_fn_efs, 2, 0)
        model = M3GNet(
            element_types=element_types,
            is_intensive=False,
        )

        print(model)

        ff = Potential(model=model)
        optimizer = torch.optim.Adam(ff.model.parameters(), lr=1.0e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        train_loss_function = F.mse_loss
        validate_loss_function = F.l1_loss

        trainer = M3GNetTrainer(potential=ff, optimizer=optimizer, scheduler=scheduler)

        trainer.train(
            device=torch.device("cpu"),
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


if __name__ == "__main__":
    unittest.main()
