"""
M3GNet Trainer
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from timeit import default_timer

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from matgl.apps.pes import Potential

logger = logging.getLogger("m3gnet_trainer")


def loss_fn(
    loss: nn.Module,
    energy_weight: float,
    force_weight: float,
    labels: tuple[torch.tensor, torch.tensor, torch.tensor],
    preds: tuple[torch.tensor, torch.tensor, torch.tensor],
    num_atoms: int,
    stress_weight: float | None,
):
    e_target = labels[0].clone() / num_atoms
    f_target = labels[1].clone()
    s_target = labels[2].clone()
    e_pred = preds[0] / num_atoms
    e_loss = loss(e_target, e_pred)
    f_loss = loss(f_target, preds[1])
    e_mae = F.l1_loss(e_target, e_pred)
    f_mae = F.l1_loss(f_target, preds[1])
    if stress_weight is not None:
        s_loss = loss(s_target, preds[2])
        s_mae = F.l1_loss(s_target, preds[2])
        total_loss = energy_weight * e_loss + force_weight * f_loss + stress_weight * s_loss
    else:
        s_mae = torch.zeros(1)
        total_loss = energy_weight * e_loss + force_weight * f_loss
    total_loss = total_loss.to(torch.float)
    results = {
        "total_loss": total_loss,
        "energy_MAE": e_mae,
        "force_MAE": f_mae,
        "stress_MAE": s_mae,
    }
    return results


def train_one_step(
    potential: Potential,
    optimizer: torch.optim.Optimizer,
    train_loss: nn.Module,
    energy_weight: float,
    force_weight: float,
    dataloader: tuple,
    stress_weight: float | None = None,
    max_norm: float | None = None,
):
    potential.model.train()

    mae_e = torch.zeros(1)
    mae_f = torch.zeros(1)
    mae_s = torch.zeros(1)
    avg_loss = torch.zeros(1)

    start = default_timer()
    for g, _l_g, attrs, energies, forces, stresses in tqdm(dataloader):
        optimizer.zero_grad()
        pred_e, pred_f, pred_s, pred_h = potential(g=g, state_attr=attrs)
        pred_f = pred_f.to(torch.float)
        preds: tuple = (pred_e, pred_f, pred_s)
        labels: tuple = (energies, forces, stresses)
        num_atoms = g.batch_num_nodes()
        results = loss_fn(
            loss=train_loss,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            labels=labels,  # type: ignore
            preds=preds,  # type: ignore
            num_atoms=num_atoms,
        )

        results["total_loss"].backward()
        if max_norm is not None:
            nn.utils.clip_grad_norm_(potential.model.parameters(), max_norm)
        optimizer.step()

        mae_e += results["energy_MAE"].detach()
        mae_f += results["force_MAE"].detach()
        mae_s += results["stress_MAE"].detach()
        avg_loss += results["total_loss"].detach()
        # free memory
        del g, energies, forces, stresses, attrs
        del pred_e, pred_f, pred_s, pred_h
        del preds, labels, results

    stop = default_timer()

    mae_e = mae_e.cpu().item() / len(dataloader)
    mae_f = mae_f.cpu().item() / len(dataloader)
    mae_s = mae_s.cpu().item() / len(dataloader)
    avg_loss = avg_loss.cpu().item() / len(dataloader)

    epoch_time = stop - start

    return avg_loss, mae_e, mae_f, mae_s, epoch_time


def validate_one_step(
    potential: Potential,
    val_loss: nn.Module,
    energy_weight: float,
    force_weight: float,
    stress_weight: float | None,
    dataloader: tuple,
):
    mae_e = torch.zeros(1)
    mae_f = torch.zeros(1)
    mae_s = torch.zeros(1)
    avg_loss = torch.zeros(1)

    start = default_timer()

    #    with torch.no_grad():
    for g, _l_g, attrs, energies, forces, stresses in dataloader:
        pred_e, pred_f, pred_s, pred_h = potential(g=g, state_attr=attrs)
        pred_f = pred_f.to(torch.float)

        preds: tuple = (pred_e, pred_f, pred_s)
        labels: tuple = (energies, forces, stresses)
        num_atoms = g.batch_num_nodes()
        results = loss_fn(
            loss=val_loss,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            labels=labels,  # type: ignore
            preds=preds,  # type: ignore
            num_atoms=num_atoms,
        )

        mae_e += results["energy_MAE"].detach()
        mae_f += results["force_MAE"].detach()
        mae_s += results["stress_MAE"].detach()
        avg_loss += results["total_loss"].detach()
        # free memory
        del g, energies, forces, stresses, attrs
        del pred_e, pred_f, pred_s, pred_h
        del preds, labels, results

    stop = default_timer()

    mae_e = mae_e.cpu().item() / len(dataloader)
    mae_f = mae_f.cpu().item() / len(dataloader)
    mae_s = mae_s.cpu().item() / len(dataloader)
    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, mae_e, mae_f, mae_s, epoch_time


class StreamingJSONWriter:
    """
    Serialize streaming data to JSON.

    This class holds onto an open file reference to which it carefully
    appends new JSON data. Individual entries are input in a list, and
    after every entry the list is closed so that it remains valid JSON.
    When a new item is added, the file cursor is moved backwards to overwrite
    the list closing bracket.
    """

    def __init__(self, filename, encoder=json.JSONEncoder):
        if os.path.exists(filename):
            self.file = open(filename, "r+")
            self.delimiter = ","
        else:
            self.file = open(filename, "w")
            self.delimiter = "["
        self.encoder = encoder

    def dump(self, obj):
        """
        Dump a JSON-serializable object to file.
        """
        data = json.dumps(obj, cls=self.encoder)
        close_str = "\n]\n"
        self.file.seek(max(self.file.seek(0, os.SEEK_END) - len(close_str), 0))
        self.file.write(f"{self.delimiter}\n    {data}{close_str}")
        self.file.flush()
        self.delimiter = ","

    def close(self):
        self.file.close()


class M3GNetTrainer:
    def __init__(
        self, potential: Potential, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler
    ) -> None:
        """
        Args:
        potential: M3GNet Potential
        optimizer: torch optimizer
        scheduler: torch scheduler
        """
        self.potential = potential
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(
        self,
        num_epochs: int,
        train_loss: nn.Module,
        val_loss: nn.Module,
        energy_weight: float,
        force_weight: float,
        train_loader: tuple,
        val_loader: tuple,
        logger_name: str,
        stress_weight: float | None = None,
        max_norm: float | None = None,
    ) -> None:
        path = os.getcwd()
        ## Set a path for best model and checkpoints
        outpath = os.path.join(path, "BestModel")
        checkpath = os.path.join(path, "CheckPoints")
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.mkdir(outpath)
        if os.path.exists(checkpath):
            shutil.rmtree(checkpath)
        os.mkdir(checkpath)
        jsonlogger = StreamingJSONWriter(filename=logger_name)
        logger.info("## Training started ##")
        best_val_loss = 1000.0
        for epoch in tqdm(range(num_epochs)):
            avg_loss_train, train_energies_mae, train_forces_mae, train_stresses_mae, train_time = train_one_step(
                potential=self.potential,
                optimizer=self.optimizer,
                train_loss=train_loss,
                energy_weight=energy_weight,
                force_weight=force_weight,
                stress_weight=stress_weight,
                dataloader=train_loader,
                max_norm=max_norm,
            )
            avg_loss_val, val_energies_mae, val_forces_mae, val_stresses_mae, val_time = validate_one_step(
                potential=self.potential,
                val_loss=val_loss,
                energy_weight=energy_weight,
                force_weight=force_weight,
                stress_weight=stress_weight,
                dataloader=val_loader,
            )
            self.scheduler.step()
            logger.info(
                f"Epoch: {epoch + 1:03} Train Loss: {avg_loss_train:.4f} "
                f"Val Loss: {avg_loss_val:.4f} Train Time: {train_time:.2f} s. "
                f"Val Time: {val_time:.2f} s."
            )
            if avg_loss_val < best_val_loss:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state": self.potential.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "loss": avg_loss_val,
                    },
                    checkpath + "/%05d" % (epoch + 1) + "-%6.5f" % (avg_loss_val) + ".pt",
                )

                log_dict = {
                    "Epoch": epoch + 1,
                    "train_energy_mae": train_energies_mae,
                    "train_force_mae": train_forces_mae,
                    "train_stress_mae": train_stresses_mae,
                    "val_energy_mae": val_energies_mae,
                    "val_force_mae": val_forces_mae,
                    "val_stress_mae": val_stresses_mae,
                    "train_time": train_time,
                    "val_time": val_time,
                }

                jsonlogger.dump(log_dict)
                best_val_loss = avg_loss_val
                self.potential.model.save(outpath)
        jsonlogger.close()
        logger.info("## Training finished ##")
