from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


class Set2Set(nn.Module):
    """Set2Set readout for node features in PyTorch Geometric."""

    def __init__(self, in_feats: int, n_iters: int, n_layers: int):
        super().__init__()
        self.in_feats = in_feats
        self.n_iters = n_iters
        self.n_layers = n_layers
        # LSTM takes input of size 2 * in_feats (concatenation of features and query)
        self.lstm = nn.LSTM(2 * in_feats, in_feats, n_layers)
        # Initialize LSTM hidden state
        self.hidden_dim = in_feats
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for node feature aggregation.

        Args:
            x: Node features, shape (num_nodes, in_feats)
            batch: Batch vector, shape (num_nodes,), indicating graph assignment

        Returns:
            Pooled features, shape (num_graphs, 2 * in_feats)
        """
        # Get batch size (number of graphs)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        # Initialize LSTM hidden state
        h = (
            torch.zeros(self.n_layers, batch_size, self.in_feats).to(x.device),
            torch.zeros(self.n_layers, batch_size, self.in_feats).to(x.device),
        )

        # Initialize query vector q_star
        q_star = torch.zeros(batch_size, self.in_feats).to(x.device)

        for _ in range(self.n_iters):
            # Query vector for each graph
            q = q_star  # Shape: (batch_size, in_feats)
            # Expand query to all nodes
            q_expanded = q[batch]  # Shape: (num_nodes, in_feats)
            # Concatenate node features and query
            lstm_input = torch.cat([x, q_expanded], dim=-1)  # Shape: (num_nodes, 2 * in_feats)
            # Pool to graph level
            lstm_input_pooled = scatter(lstm_input, batch, dim=0, reduce="sum")  # Shape: (batch_size, 2 * in_feats)
            # Reshape for LSTM: (seq_len=1, batch_size, 2 * in_feats)
            lstm_input_pooled = lstm_input_pooled.unsqueeze(0)
            # LSTM step
            q_star, h = self.lstm(lstm_input_pooled, h)  # q_star: (1, batch_size, in_feats)
            q_star = q_star.squeeze(0)  # Shape: (batch_size, in_feats)

        # Final output: concatenate final query and last hidden state
        output = torch.cat([q_star, h[0][-1]], dim=-1)  # Shape: (batch_size, 2 * in_feats)
        return output
