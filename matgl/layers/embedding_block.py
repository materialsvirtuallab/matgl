"""
Embedding node, edge and optional state attributes
"""
from __future__ import annotations

import torch
import torch.nn as nn

from matgl.layers.core import MLP


class EmbeddingBlock(nn.Module):
    """
    Embedding block for generating node, bond and state features
    """

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        num_node_feats: int,
        num_edge_feats: int,
        num_node_types: int | None = None,
        num_state_feats: int | None = None,
        include_states: bool = False,
        num_state_types: int | None = None,
        state_embedding_dim: int | None = None,
        device: str = "cpu",
    ):
        """
        Parameters:
        -----------
        num_node_feats (int): dimensionality of node features
        num_edge_feats (int): dimensionality of edge features
        num_state_feats (int): dimensionality of state features
        include_states (bool): whether including state for M3GNet or not
        state_embedding_dim (int): dimensionality of state embedding
        activation (str): activation function type
        """
        super().__init__()
        device = torch.device(device)
        self.include_states = include_states
        self.num_state_types = num_state_types
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.num_state_feats = num_state_feats
        self.num_node_types = num_node_types
        self.state_embedding_dim = state_embedding_dim
        self.activation = activation
        self.device = device
        if num_state_types and state_embedding_dim is not None:
            self.state_embedding = nn.Embedding(num_state_types, state_embedding_dim, device=device)  # type: ignore
        if num_node_types is not None:
            self.node_embedding = nn.Embedding(num_node_types, num_node_feats, device=device)
        self.edge_embedding = MLP(
            [degree_rbf, self.num_edge_feats],
            activation=activation,
            activate_last=True,
            device=device,
        )

    def forward(self, node_attr, edge_attr, state_attr):
        """
        Output embedded features

        Args:
        node_attr: node attribute
        edge_attr: edge attribute
        state_attr: state attribute

        Returns:
        node_feat: embedded node features
        edge_feat: embedded edge features
        state_feat: embedded state features
        """

        if self.num_node_types is not None:
            node_feat = self.node_embedding(node_attr)
        else:
            node_embed = MLP([node_attr.shape[-1], self.num_node_feats], activation=self.activation, device=self.device)
            node_feat = node_embed(node_attr.to(torch.float32))

        edge_feat = self.edge_embedding(edge_attr.to(torch.float32))
        if self.include_states is True:
            if self.num_state_types and self.state_embedding_dim is not None:
                state_feat = self.state_embedding(state_attr)
                state_feat = self.activation(state_feat)
            else:
                state_attr = torch.unsqueeze(state_attr, 0)
                state_embed = MLP(
                    [state_attr.shape[-1], self.num_state_feats], activation=self.activation, device=self.device
                )
                state_feat = state_embed(state_attr.to(torch.float32))
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat
