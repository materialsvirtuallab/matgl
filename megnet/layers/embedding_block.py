"""
Embedding node, edge and optional state attributes
"""
import torch
import torch.nn as nn

from megnet.layers.core import MLP


class EmbeddingBlock(nn.Module):
    """
    Embedding block for generating node, bond and state features
    """

    def __init__(
        self,
        num_node_feats: int = None,
        num_edge_feats: int = None,
        num_state_feats: int = None,
        include_states: bool = False,
        num_state_types: int = None,
        state_embedding_dim: int = None,
        activation: str = "swish",
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

        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        self.include_states = include_states
        self.num_state_types = num_state_types
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.num_state_feats = num_state_feats
        self.state_embedding_dim = state_embedding_dim

        if num_state_types and state_embedding_dim is not None:
            self.state_embedding = nn.Embedding(num_state_types, state_embedding_dim)

    def forward(self, node_attr, edge_attr, state_attr):
        """
        Output embedded features

        Args:
        g: dgl graph
        state_attr: state attribute

        Returns:
        g: dgl graph includeing node, edge embedded features
        state_feat: embedded state features
        """
        node_embed = MLP([node_attr.shape[-1], self.num_node_feats], activation=self.activation)
        edge_embed = MLP([edge_attr.shape[-1], self.num_edge_feats], activation=self.activation)
        node_feat = node_embed(node_attr.to(torch.float32))
        edge_feat = edge_embed(edge_attr.to(torch.float32))
        if self.include_states is True:
            if self.num_state_types and self.state_embedding_dim is not None:
                state_feat = self.state_embedding(state_attr)
                state_feat = self.activation(state_feat)
            else:
                state_attr = torch.unsqueeze(state_attr, 0)
                state_embed = MLP([state_attr.shape[-1], self.num_state_feats], activation=self.activation)
                state_feat = state_embed(state_attr.to(torch.float32))
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat
