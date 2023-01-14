import argparse, json
from collections import namedtuple
from timeit import default_timer
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from munch import Munch

import megnet.utils
#from megnet.utils import create_dataloaders, StreamingJSONWriter, set_seed, prepare_config, prepare_data
from qm9_utils  import create_dataloaders, StreamingJSONWriter, set_seed, prepare_config, prepare_data #these funcitons are loaded from the qm9_utils.py in this example folder.
from megnet.models import MEGNet
from megnet.models.helper import MLP
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str("0,1,2,3")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def train(
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data: namedtuple,
        dataloader: namedtuple,
):
    model.train()

    avg_loss = 0

    start = default_timer()

    for g, labels in tqdm(dataloader):
        optimizer.zero_grad()

        g = g.to(device)
        labels = labels.to(device)

        node_feat = torch.hstack((g.ndata['attr'], g.ndata['pos']))
        edge_feat = g.edata['edge_attr']
        attrs = torch.ones(g.batch_size, 2).to(device) * torch.tensor([data.z_mean, data.num_bond_mean]).to(device)

        pred = model(g, edge_feat, node_feat, attrs)

        loss = loss_function(pred, (labels - data.mean) / data.std)

        loss.backward()
        optimizer.step()

        avg_loss += loss.detach()

    stop = default_timer()

    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, epoch_time


def validate(
        model: nn.Module,
        device: torch.device,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data: namedtuple,
        dataloader: namedtuple,
):
    avg_loss = 0

    start = default_timer()

    with torch.no_grad():
        for g, labels in dataloader:
            g = g.to(device)
            labels = labels.to(device)

            node_feat = torch.hstack((g.ndata['attr'], g.ndata['pos']))
            edge_feat = g.edata['edge_attr']
            attrs = torch.ones(g.batch_size, 2).to(device) * torch.tensor([data.z_mean, data.num_bond_mean]).to(device)

            pred = model(g, edge_feat, node_feat, attrs)

            loss = loss_function(data.mean + pred * data.std, labels)

            avg_loss += loss

    stop = default_timer()

    avg_loss = avg_loss.cpu().item() / len(dataloader)
    epoch_time = stop - start

    return avg_loss, epoch_time

def init_process_group(world_size, rank, backend="gloo"):
    dist.init_process_group(
        backend=backend,     # change to 'nccl' for multiple GPUs, 'gloo' for single GPU
        init_method='tcp://127.0.0.1:12345',
        world_size=world_size,
        rank=rank)

def run(
        rank: int,
        args: argparse.ArgumentParser,
    #    config: Munch,
    #    data: namedtuple,
):
    
    config = prepare_config(f'./configs/{args.config_name}.yaml')
    data = prepare_data(config)
    g_sample = data.train[0][0]

    node_feat = torch.hstack((g_sample.ndata['attr'], g_sample.ndata['pos']))
    edge_feat = g_sample.edata['edge_attr']
    attrs = torch.tensor([data.z_mean, data.num_bond_mean])

    node_embed = MLP([node_feat.shape[-1], config.model.DIM])
    edge_embed = MLP([edge_feat.shape[-1], config.model.DIM])
    attr_embed = MLP([attrs.shape[-1], config.model.DIM])

    backend = "gloo" if args.n_gpus <= 1 else "nccl"
    use_ddp = False if args.n_gpus <= 1 else True
    init_process_group(world_size=args.n_gpus, rank=rank, backend=backend)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    model = MEGNet(
        in_dim=config.model.DIM,
        num_blocks=config.model.num_blocks,
        hiddens=[config.model.N1, config.model.N2],
        conv_hiddens=[config.model.N1, config.model.N1, config.model.N2],
        s2s_num_layers=config.model.s2s_num_layers,
        s2s_num_iters=config.model.s2s_num_iters,
        output_hiddens=[config.model.N2, config.model.N3],
        is_classification=False,
        node_embed=node_embed,
        edge_embed=edge_embed,
        attr_embed=attr_embed
    )

    model = model.to(device)

    if device.type == 'cpu':
        model = DistributedDataParallel(model)
    else:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), config.optimizer.lr)

    train_loss_function = F.mse_loss
    validate_loss_function = F.l1_loss

    dataloaders = create_dataloaders(config, data, use_ddp=use_ddp)

    logger = StreamingJSONWriter(filename='./qm9_logs.json')

    print('## Training started ##')

    for epoch in tqdm(range(1000)):
#    for epoch in tqdm(range(config.optimizer.max_epochs)):
        train_loss, train_time = train(
            model, device, optimizer, train_loss_function, data, dataloaders.train)
        val_loss, val_time = validate(
            model, device, validate_loss_function, data, dataloaders.val)

        print(
            f'Epoch: {epoch + 1:03} Train Loss: {train_loss:.4f} '
            f'Val Loss: {val_loss:.4f} Train Time: {train_time:.2f} s. '
            f'Val Time: {val_time:.2f} s.'
        )

        log_dict = {'Epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                    'train_time': train_time, 'val_time': val_time}

        logger.dump(log_dict)


    print('## Training finished ##')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser('Agent Backbone Training')

    argparser.add_argument('--config-name', default='qm9_test', type=str)
    argparser.add_argument('--test-validation', dest='test_validation',
                           action='store_true')
    argparser.add_argument('--no-test-validation', dest='test_validation',
                           action='store_false')
    argparser.set_defaults(test_validation=True)
    argparser.add_argument('--seed', default=0, type=int)
    argparser.add_argument('--ngpus', dest='n_gpus',  default=1, type=int)

    args = argparser.parse_args()
    set_seed(args.seed)
    print(args)
    mp.spawn(run, args=[args], nprocs=args.n_gpus)
