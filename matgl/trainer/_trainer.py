from __future__ import annotations

import dgl
import numpy as np
import pytorch_lightning as pl

# import torch and dgl modules
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from matgl.apps.pes import Potential

# import matgl relevant modules
from matgl.models import M3GNet, MEGNet


class MatglTrainer(pl.LightningModule):
    """
    MatglTrainer for MEGNet and M3GNet including potential
    Params:
    model: Which type of the model for training
    element_refs: element offset for PES
    energy_weight: relative importance of energy
    force_weight: relative importance of force
    stress_weight: relative importance of stress
    data_mean: average of training data
    data_std: standard deviation of training data
    calc_stress: whether stress calculation is required
    loss: loss function used for training
    optimizer: optimizer for training
    scheduler: scheduler for training
    lr: learning rate for training
    decay_steps: number of steps for decaying learning rate
    decay_alpha: parameter determines the minimum learning rate
    is_potential: whether the potential is used for training
    """

    def __init__(
        self,
        model: MEGNet | M3GNet,
        element_refs: np.darray | None = None,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float | None = None,
        data_mean=None,
        data_std=None,
        calc_stress: bool = False,
        loss: str = "mse_loss",
        optimizer: Optimizer | None = None,
        scheduler: lr_scheduler | None = None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        is_potential: bool = False,
    ):
        super().__init__()

        # check if the model is M3GNet or MEGNet
        if isinstance(model, M3GNet):
            if is_potential is True:
                self.model = Potential(model=model, element_refs=element_refs, calc_stresses=calc_stress)
            else:
                self.model = model
        else:
            self.model = model

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        if data_mean is None:
            data_mean = torch.zeros(1)
        if data_std is None:
            data_std = torch.ones(1)
        self.data_mean = data_mean
        self.data_std = data_std
        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        if loss == "mse_loss":
            self.loss = F.mse_loss
        else:
            self.loss = F.l1_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_hyperparameters()

    def forward(self, g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.tensor | None = None):
        if isinstance(self.model, Potential):
            e, f, s, h = self.model(g=g, l_g=l_g, state_attr=state_attr)
            return e, f.float(), s, h
        else:
            if self.model is M3GNet:
                preds = self.model(g=g, l_g=l_g, state_attr=state_attr)
            else:
                node_feat = g.ndata["node_type"]
                edge_feat = g.edata["edge_attr"]
                preds = self.model(g, edge_feat.float(), node_feat.long(), state_attr)
            return preds

    def step(self, batch: tuple):
        if isinstance(self.model, Potential):
            torch.set_grad_enabled(True)
            g, l_g, state_attr, energies, forces, stresses = batch
            e, f, s, _ = self(g=g, state_attr=state_attr, l_g=l_g)
            f = f.to(torch.float)
            preds: tuple = (e, f, s)
            labels: tuple = (energies, forces, stresses)
            num_atoms = g.batch_num_nodes()
            results = self.loss_fn_efs(
                loss=self.loss,
                preds=preds,
                labels=labels,
                energy_weight=self.energy_weight,
                force_weight=self.force_weight,
                stress_weight=self.stress_weight,
                num_atoms=num_atoms,
            )
            batch_size = preds[0].numel()
        else:
            g, labels, state_attr = batch
            preds = self(g=g, state_attr=state_attr)
            results = self.loss_fn(loss=self.loss, preds=preds, labels=labels)
            batch_size = preds.numel()
        return results, batch_size

    def training_step(self, batch: tuple, batch_idx: int):
        results, batch_size = self.step(batch)
        self.log_dict(
            {f"train_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return results["Total_Loss"]

    def on_train_epoch_end(self):
        # Step scheduler every epoch
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch: tuple, batch_idx: int):
        results, batch_size = self.step(batch)
        self.log_dict(
            {f"val_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch: tuple, batch_idx: int):
        results, batch_size = self.step(batch)
        self.log_dict(
            {f"test_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                eps=1e-8,
            )
        else:
            optimizer = self.optimizer
        if self.scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay_steps,
                eta_min=self.lr * self.decay_alpha,
            )
        else:
            scheduler = self.scheduler
        return [
            optimizer,
        ], [
            scheduler,
        ]

    def on_test_model_eval(self, *args, **kwargs):
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.set_grad_enabled(True)
        return self(batch)

    def loss_fn_efs(
        self,
        loss: nn.Module,
        labels: tuple,
        preds: tuple,
        energy_weight: float | None = None,
        force_weight: float | None = None,
        stress_weight: float | None = None,
        num_atoms: int | None = None,
    ):
        e_target, f_target, s_target = labels
        pred_e, pred_f, pred_s = preds

        e_loss = self.loss(e_target / num_atoms, pred_e / num_atoms)
        f_loss = self.loss(f_target, pred_f)

        e_mae = self.mae(e_target / num_atoms, pred_e / num_atoms)
        f_mae = self.mae(f_target, pred_f)

        e_rmse = self.rmse(e_target / num_atoms, pred_e / num_atoms)
        f_rmse = self.rmse(f_target, pred_f)

        s_mae = torch.zeros(1)
        s_rmse = torch.zeros(1)

        if stress_weight is not None:
            s_loss = loss(s_target, pred_s)
            s_mae = self.mae(s_target, pred_s)
            s_rmse = self.rmse(s_target, pred_s)
            total_loss = energy_weight * e_loss + force_weight * f_loss + stress_weight * s_loss
        else:
            total_loss = energy_weight * e_loss + force_weight * f_loss

        results = {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
        }
        return results

    def loss_fn(self, loss: nn.Module, labels: tuple, preds: tuple):
        total_loss = loss(labels, torch.squeeze(preds * self.data_std + self.data_mean))
        mae = self.mae(labels, torch.squeeze(preds * self.data_std + self.data_mean))
        rmse = self.rmse(labels, torch.squeeze(preds * self.data_std + self.data_mean))
        results = {"Total_Loss": total_loss, "MAE": mae, "RMSE": rmse}
        return results
