from __future__ import annotations

import codecs
import csv
import logging
import os
from pathlib import Path
from timeit import default_timer

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__file__)


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    dataloader,
):
    """

    Args:
        model:
        optimizer:
        loss_function:
        data_std:
        data_mean:
        dataloader:

    Returns:

    """
    model.train()

    avg_loss = torch.zeros(1)

    start = default_timer()

    for g, labels, attrs in tqdm(dataloader):
        optimizer.zero_grad()

        node_feat = g.ndata["node_type"]
        edge_feat = g.edata["edge_attr"]

        pred = model(g, edge_feat.float(), node_feat.long(), attrs)

        pred = torch.squeeze(pred)

        loss = loss_function(pred, (labels - model.data_mean) / model.data_std)

        loss.backward()
        optimizer.step()

        avg_loss += loss.detach()

    stop = default_timer()

    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, epoch_time


def validate_one_step(
    model: nn.Module,
    loss_function: nn.Module,
    dataloader: tuple,
):
    avg_loss = torch.zeros(1)

    start = default_timer()

    with torch.no_grad():
        for g, labels, attrs in dataloader:
            node_feat = g.ndata["node_type"]
            edge_feat = g.edata["edge_attr"]

            pred = model(g, edge_feat.float(), node_feat.long(), attrs)

            pred = torch.squeeze(pred)

            loss = loss_function(model.data_mean + pred * model.data_std, labels)

            avg_loss += loss

    stop = default_timer()

    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, epoch_time


class ModelTrainer:
    def __init__(self, model, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
        """

        Args:
            model: Model to be trained.
            optimizer: torch.Optimizer
            scheduler: torch Scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(
        self,
        nepochs: int,
        train_loss_func: nn.Module,
        val_loss_func: nn.Module,
        train_loader: tuple,
        val_loader: tuple,
        logpath: str | Path = "matgl_training",
    ) -> None:
        """

        Args:
            nepochs:
            train_loss_func:
            val_loss_func:
            train_loader:
            val_loader:
            training_logfile:
            logpath:
        """
        logpath = Path(logpath)
        outpath = logpath / "best_model"
        checkpath = logpath / "checkpoints"
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(checkpath, exist_ok=True)
        logger.info("## Training started ##")
        best_val_loss = 1000.0

        with codecs.open(logpath / "training_log.csv", "w", "utf-8") as fp:  # type: ignore
            csvlog = csv.writer(fp)
            row = ["epoch", "train_loss", "val_loss", "train_time", "val_time"]
            csvlog.writerow(row)
            for epoch in tqdm(range(nepochs)):
                train_loss, train_time = train_one_step(
                    self.model,
                    self.optimizer,
                    train_loss_func,
                    train_loader,
                )
                val_loss, val_time = validate_one_step(self.model, val_loss_func, val_loader)

                self.scheduler.step()
                logger.info(
                    f"Epoch: {epoch + 1:03} Train Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} Train Time: {train_time:.2f} s. "
                    f"Val Time: {val_time:.2f} s."
                )
                if val_loss < best_val_loss:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "loss": val_loss,
                        },
                        checkpath / f"{epoch + 1}-{val_loss}.pt",
                    )
                    csvlog.writerow([epoch + 1, train_loss, val_loss, train_time, val_time])
                    best_val_loss = val_loss
                    self.model.save(outpath)
        logger.info("## Training finished ##")
