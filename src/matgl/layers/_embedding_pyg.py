"""Embedding node, edge and optional state attributes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

import matgl
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import new_radial_tensor, scatter_add, tensor_norm, vector_to_skewtensor, vector_to_symtensor

if TYPE_CHECKING:
    from torch_geometric.data import Data


class TensorEmbeddingPYG(MessagePassing):
    """Embedding block for TensorNet to generate node, edge, and optional state features using PyG.
    Adapted from the DGL implementation in https://github.com/torchmd/torchmd-net.
    """

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        activation: nn.Module,
        ntypes_node: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
    ):
        """
        Args:
            units (int): Number of hidden neurons.
            degree_rbf (int): Number of radial basis functions.
            activation (nn.Module): Activation function.
            ntypes_node (int): Number of node labels.
            cutoff (float): Cutoff radius for graph construction.
            dtype (torch.dtype): Data type for all variables.
        """
        super().__init__(aggr="add")  # Use 'add' aggregation for summing messages
        self.units = units
        self.cutoff = cutoff

        # Initialize layers
        self.distance_proj1 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.distance_proj2 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.distance_proj3 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.emb = nn.Embedding(ntypes_node, units, dtype=dtype)
        self.emb2 = nn.Linear(2 * units, units, dtype=dtype)
        self.act = activation
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(3)])
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )
        self.init_norm = nn.LayerNorm(units, dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the parameters."""
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def message(self, x_i, x_j, edge_attr, edge_weight, Iij, Aij, Sij):
        """Message function for edge updates."""
        vi = x_i  # Source node features
        vj = x_j  # Destination node features
        # Concatenate node and state features
        zij = torch.cat([vi, vj], dim=-1)
        Zij = self.emb2(zij)
        scalars = Zij[..., None, None] * Iij
        skew_matrices = Zij[..., None, None] * Aij
        traceless_tensors = Zij[..., None, None] * Sij
        return {"I": scalars, "A": skew_matrices, "S": traceless_tensors}

    def aggregate(self, graph, index, dim_size=None):
        """Aggregate messages for node updates."""
        scalars = scatter_add(graph.I, index, dim_size=dim_size)
        skew_matrices = scatter_add(graph.A, index, dim_size=dim_size)
        traceless_tensors = scatter_add(graph.S, index, dim_size=dim_size)
        return scalars, skew_matrices, traceless_tensors

    def forward(self, graph: Data, state_attr=None):
        """
        Args:
            graph (torch_geometric.data.Data): Graph data with node features (x), edge indices (edge_index),
                                             edge attributes (edge_attr), and bond distances (bond_dist).
            state_attr (torch.Tensor, optional): Global state attributes.

        Returns:
            X (torch.Tensor): Embedded node tensor representation.
            state_feat (torch.Tensor): Embedded state features.
        """
        z, edge_index, edge_attr, edge_weight = graph.node_type, graph.edge_index, graph.edge_attr, graph.bond_dist
        edge_vec = graph.bond_vec

        # Node embedding
        x = self.emb(z)  # Assuming node_type is integer for embedding

        # set dummy state features for now and nothing to do for state_attr
        state_feat = None

        # Edge processing
        C = cosine_cutoff(edge_weight, self.cutoff)
        W1 = self.distance_proj1(edge_attr) * C.view(-1, 1)
        W2 = self.distance_proj2(edge_attr) * C.view(-1, 1)
        W3 = self.distance_proj3(edge_attr) * C.view(-1, 1)
        edge_vec = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)

        # Radial tensor components
        Iij, Aij, Sij = new_radial_tensor(
            torch.eye(3, 3, device=edge_vec.device, dtype=edge_vec.dtype).unsqueeze(0).unsqueeze(0),
            vector_to_skewtensor(edge_vec).unsqueeze(-3),
            vector_to_symtensor(edge_vec).unsqueeze(-3),
            W1,
            W2,
            W3,
        )

        # Perform message passing
        msg = self.message(
            x_i=x[edge_index[0]],
            x_j=x[edge_index[1]],
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            Iij=Iij,
            Aij=Aij,
            Sij=Sij,
        )

        graph.I, graph.A, graph.S = msg["I"], msg["A"], msg["S"]
        scalars, skew_matrices, traceless_tensors = self.aggregate(graph, edge_index[1], dim_size=x.size(0))

        # Node update
        norm = tensor_norm(scalars + skew_matrices + traceless_tensors)
        norm = self.init_norm(norm)
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_matrices = self.linears_tensor[1](skew_matrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(norm.shape[0], self.units, 3)
        scalars, skew_matrices, traceless_tensors = new_radial_tensor(
            scalars, skew_matrices, traceless_tensors, norm[..., 0], norm[..., 1], norm[..., 2]
        )
        X = scalars + skew_matrices + traceless_tensors

        return X, state_feat
