from __future__ import annotations

import os
import shutil

# This function is used for M3GNet property dataset
from functools import partial

import numpy as np
import pytest
import pytorch_lightning as pl
import torch.backends.mps
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn, collate_fn_efs
from matgl.models import M3GNet, MEGNet, SO3Net, TensorNet
from matgl.utils.training import ModelLightningModule, PotentialLightningModule, xavier_init
from pymatgen.core import Lattice, Structure

module_dir = os.path.dirname(os.path.abspath(__file__))

# The device can be chosen as "cpu" or "cuda". Note:"mps" is currently not available
device = "cpu"
torch.set_default_device(device)
torch.set_float32_matmul_precision("high")


class TestModelTrainer:
    def test_megnet_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
        label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
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
            generator=torch.Generator(device=device),
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
        xavier_init(model)
        lit_model = ModelLightningModule(model=model)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=10, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        results = trainer.test(dataloaders=test_loader)
        assert "test_Total_Loss" in results[0]
        model = model.to(torch.device(device))
        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        self.teardown_class()

    def test_m3gnet_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
        structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
        energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist()] * len(structures)
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            include_line_graph=True,
            labels={"energies": energies, "forces": forces, "stresses": stresses},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        my_collate_fn = partial(collate_fn_efs, include_line_graph=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = M3GNet(element_types=element_types, is_intensive=False)
        lit_model = PotentialLightningModule(model=model, stress_weight=0.0001, include_line_graph=True)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0
        # specify customize optimizer and scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
        lit_model = PotentialLightningModule(
            model=model,
            stress_weight=0.0001,
            include_line_graph=True,
            loss="l1_loss",
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        self.teardown_class()

    def test_so3net_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
        structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
        energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist()] * len(structures)
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            include_line_graph=False,
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
            collate_fn=collate_fn_efs,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = SO3Net(element_types=element_types, lmax=2, is_intensive=False)
        lit_model = PotentialLightningModule(model=model, stress_weight=0.0001)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0
        # specify customize optimizer and scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
        lit_model = PotentialLightningModule(
            model=model,
            stress_weight=0.0001,
            loss="l1_loss",
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        self.teardown_class()

    def test_tensornet_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
        structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
        energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist()] * len(structures)
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            include_line_graph=False,
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
            collate_fn=collate_fn_efs,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = TensorNet(element_types=element_types, is_intensive=False)
        lit_model = PotentialLightningModule(model=model, stress_weight=0.0001)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0
        # specify customize optimizer and scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
        lit_model = PotentialLightningModule(
            model=model,
            stress_weight=0.0001,
            loss="l1_loss",
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        self.teardown_class()

    def test_m3gnet_training_without_stress(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
        structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
        energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            include_line_graph=True,
            labels={"energies": energies, "forces": forces},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        my_collate_fn = partial(collate_fn_efs, include_stress=False, include_line_graph=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = M3GNet(element_types=element_types, is_intensive=False)
        lit_model = PotentialLightningModule(model=model, include_line_graph=True, stress_weight=0.0)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=5, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0
        self.teardown_class()

    def test_m3gnet_property_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
        label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            include_line_graph=True,
            labels={"eform": label},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        collate_fn_property = partial(collate_fn, include_line_graph=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn_property,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = M3GNet(
            element_types=element_types,
            is_intensive=True,
            readout_type="set2set",
        )
        lit_model = ModelLightningModule(model=model, include_line_graph=True)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]

        lit_model = ModelLightningModule(model=model, include_line_graph=True, loss="l1_loss")
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]
        self.teardown_class()

    def test_so3net_property_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
        label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            labels={"eform": label},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = SO3Net(
            element_types=element_types,
            is_intensive=True,
            lmax=2,
            target_property="graph",
            readout_type="set2set",
        )
        lit_model = ModelLightningModule(model=model)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]

        lit_model = ModelLightningModule(model=model, loss="l1_loss")
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]
        self.teardown_class()

    def test_tensornet_property_training(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
        label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            labels={"eform": label},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = TensorNet(
            element_types=element_types,
            is_intensive=True,
            readout_type="set2set",
        )
        lit_model = ModelLightningModule(model=model)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]

        lit_model = ModelLightningModule(model=model, loss="l1_loss")
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert pred_LFP_energy < 0
        assert pred_BNO_energy < 0

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]
        self.teardown_class()

    def test_m3gnet_property_trainin_multiple_values_per_target(self, LiFePO4, BaNiO3):
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
        label = np.full((11, 5), -1.0).tolist()  # Artificial dataset.
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            threebody_cutoff=4.0,
            structures=structures,
            converter=converter,
            include_line_graph=True,
            labels={"multiple_values": label},
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        collate_fn_property = partial(collate_fn, include_line_graph=True, multiple_values_per_target=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn_property,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = M3GNet(
            element_types=element_types,
            is_intensive=True,
            readout_type="set2set",
            ntargets=5,
        )
        lit_model = ModelLightningModule(model=model, include_line_graph=True)
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert torch.any(pred_LFP_energy < 0)
        assert torch.any(pred_BNO_energy < 0)

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]

        lit_model = ModelLightningModule(model=model, include_line_graph=True, loss="l1_loss")
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=2, accelerator=device)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
        assert torch.any(pred_LFP_energy < 0)
        assert torch.any(pred_BNO_energy < 0)

        results = trainer.predict(model=lit_model, dataloaders=test_loader)

        assert "MAE" in results[0][0]
        self.teardown_class()

    @classmethod
    def teardown_class(cls):
        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass
        try:
            shutil.rmtree("lightning_logs")
        except FileNotFoundError:
            pass


@pytest.mark.parametrize("distribution", ["normal", "uniform", "fake"])
def test_xavier_init(distribution):
    model = MEGNet()
    # get a parameter
    w = model.output_proj.layers[0].get_parameter("weight").clone()

    if distribution == "fake":
        with pytest.raises(ValueError, match=r"^Invalid distribution:."):
            xavier_init(model, distribution=distribution)
    else:
        xavier_init(model, distribution=distribution)
        print(w)
        assert not torch.allclose(w, model.output_proj.layers[0].get_parameter("weight"))
        assert torch.allclose(torch.tensor(0.0), model.output_proj.layers[0].get_parameter("bias"))
