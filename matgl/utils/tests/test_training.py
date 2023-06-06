from __future__ import annotations

import os
import shutil
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from dgl.data.utils import split_dataset
from pymatgen.util.testing import PymatgenTest
from torch.optim.lr_scheduler import ExponentialLR

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import (
    MEGNetDataset,
    MGLDataLoader,
    collate_fn,
)
from matgl.models import MEGNet
from matgl.utils.training import ModelTrainer

module_dir = os.path.dirname(os.path.abspath(__file__))


class ModelTrainerTest(PymatgenTest):
    def test_megnet_training(self):
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

        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        train_loss_function = F.mse_loss
        validate_loss_function = F.l1_loss

        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        trainer.train(
            n_epochs=2,
            train_loss_func=train_loss_function,
            val_loss_func=validate_loss_function,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    @classmethod
    def tearDownClass(cls):
        for fn in ("dgl_graph.bin", "dgl_line_graph.bin", "state_attr.pt", "labels.json", "test_trainer.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
        shutil.rmtree("matgl_training")


if __name__ == "__main__":
    unittest.main()
