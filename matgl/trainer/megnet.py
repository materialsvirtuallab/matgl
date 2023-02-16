"""
MEGNet Trainer
"""
import json
from collections import namedtuple
from timeit import default_timer
from typing import Callable

import os
import torch
import torch.nn as nn


from matgl.models.megnet import MEGNet

from tqdm import tqdm


def train_one_step(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_std: torch.Tensor,
    data_mean: torch.Tensor,
    dataloader,
):
    model.train()

    avg_loss = 0

    start = default_timer()

    for g, labels, attrs in tqdm(dataloader):
        optimizer.zero_grad()

        g = g.to(device)
        labels = labels.to(device)

        node_feat = g.ndata["attr"]
        edge_feat = g.edata["edge_attr"]
        attrs = attrs.to(device)

        pred = model(g, edge_feat.float(), node_feat.float(), attrs.float())

        pred = torch.squeeze(pred)

        loss = loss_function(pred, (labels - data_mean) / data_std)

        loss.backward()
        optimizer.step()

        avg_loss += loss.detach()

    stop = default_timer()

    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, epoch_time


def validate_one_step(
    model: nn.Module,
    device: torch.device,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_std: torch.Tensor,
    data_mean: torch.Tensor,
    dataloader: namedtuple,
):
    avg_loss = 0

    start = default_timer()

    with torch.no_grad():
        for g, labels, attrs in dataloader:
            g = g.to(device)
            labels = labels.to(device)

            node_feat = g.ndata["attr"]
            edge_feat = g.edata["edge_attr"]
            attrs = attrs.to(device)

            pred = model(g, edge_feat.float(), node_feat.float(), attrs.float())

            pred = torch.squeeze(pred)

            loss = loss_function(data_mean + pred * data_std, labels)

            avg_loss += loss

    stop = default_timer()

    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, epoch_time


class StreamingJSONWriter(object):
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
            self.delimeter = ","
        else:
            self.file = open(filename, "w")
            self.delimeter = "["
        self.encoder = encoder

    def dump(self, obj):
        """
        Dump a JSON-serializable object to file.
        """
        data = json.dumps(obj, cls=self.encoder)
        close_str = "\n]\n"
        self.file.seek(max(self.file.seek(0, os.SEEK_END) - len(close_str), 0))
        self.file.write("%s\n    %s%s" % (self.delimeter, data, close_str))
        self.file.flush()
        self.delimeter = ","

    def close(self):
        self.file.close()


class MEGNetTrainer:
    def __init__(self, model: MEGNet, optimizer: torch.optim.Optimizer) -> None:
        """
        Parameters:
        model: MEGNet model
        optimizer: torch Optimizer
        """
        self.model = model
        self.optimizer = optimizer

    def train(
        self,
        device: torch.device,
        num_epochs: int,
        train_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        val_loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data_std: torch.tensor,
        data_mean: torch.tensor,
        train_loader: namedtuple,
        val_loader: namedtuple,
        logger_name: str,
    ) -> None:
        path = os.getcwd()
        ## Set a path for best model and checkpoints
        outpath = os.path.join(path, "BestModel")
        checkpath = os.path.join(path, "CheckPoints")
        if os.path.exists(outpath):
            os.remove(outpath)
        os.mkdir(outpath)
        if os.path.exists(checkpath):
            os.remove(checkpath)
        os.mkdir(checkpath)
        logger = StreamingJSONWriter(filename=logger_name)
        print("## Training started ##")
        best_val_loss = 1000.0
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_time = train_one_step(
                self.model,
                device,
                self.optimizer,
                train_loss_func,
                data_std,
                data_mean,
                train_loader,
            )
            val_loss, val_time = validate_one_step(self.model, device, val_loss_func, data_std, data_mean, val_loader)

            print(
                f"Epoch: {epoch + 1:03} Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} Train Time: {train_time:.2f} s. "
                f"Val Time: {val_time:.2f} s."
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpath + "/%05d" % (epoch + 1) + "-%6.5f" % (val_loss) + ".pt",
            )

            log_dict = {
                "Epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_time": train_time,
                "val_time": val_time,
            }

            logger.dump(log_dict)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), outpath + "/best-model.pt")

            print("## Training finished ##")
