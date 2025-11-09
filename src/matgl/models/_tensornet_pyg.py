"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
)
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import (
    decompose_tensor,
    new_radial_tensor,
    scatter_add,
    vector_to_skewtensor,
    vector_to_symtensor,
)

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


def compose_tensor(I_tensor: torch.Tensor, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Compose tensor from scalar (I_tensor), skew-symmetric (A), and traceless symmetric (S) components.

    Args:
        I_tensor: Scalar component, shape (num_nodes, 1, 1, units) or (num_nodes, 3, 3, units)
        A: Skew-symmetric component, shape (num_nodes, 3, 3, units)
        S: Traceless symmetric component, shape (num_nodes, 3, 3, units)

    Returns:
        Composed tensor, shape (num_nodes, 3, 3, units)
    """
    # I_tensor is scalar (1x1), A is skew (3x3), S is traceless symmetric (3x3)
    # For I_tensor, we need to expand it to 3x3 identity matrix
    if I_tensor.shape[1] == 1 and I_tensor.shape[2] == 1:
        # I_tensor has shape (num_nodes, 1, 1, units)
        # Expand scalar to 3x3 identity matrix: multiply I_tensor by identity
        eye = torch.eye(3, 3, device=I_tensor.device, dtype=I_tensor.dtype)  # (3, 3)
        # I_tensor: (num_nodes, 1, 1, units)
        # We need: I_expanded[i, :, :, u] = I_tensor[i, 0, 0, u] * eye
        # I_values: (num_nodes, units)
        I_values = I_tensor.squeeze(1).squeeze(1)  # (num_nodes, units)
        # eye_expanded: (1, 3, 3, 1) for broadcasting
        eye_expanded = eye.unsqueeze(0).unsqueeze(-1)  # (1, 3, 3, 1)
        # I_values.unsqueeze(1).unsqueeze(1): (num_nodes, 1, 1, units)
        # Multiply: (num_nodes, 1, 1, units) * (1, 3, 3, 1) -> (num_nodes, 3, 3, units)
        I_expanded = I_values.unsqueeze(1).unsqueeze(1) * eye_expanded  # (num_nodes, 3, 3, units)
    else:
        I_expanded = I_tensor

    # A is already 3x3 skew-symmetric, shape (num_nodes, 3, 3, units)
    # S is already 3x3 traceless symmetric, shape (num_nodes, 3, 3, units)
    # Verify shapes before addition
    assert I_expanded.shape[-1] == A.shape[-1] == S.shape[-1], (
        f"Shape mismatch: I_expanded {I_expanded.shape}, A {A.shape}, S {S.shape}"
    )
    return I_expanded + A + S


def compute_pair_vector_and_distance(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    pbc_offshift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate bond vectors and distances.

    Args:
        pos: Node positions, shape (num_nodes, 3)
        edge_index: Edge indices, shape (2, num_edges)
        pbc_offshift: Periodic boundary condition offsets, shape (num_edges, 3)

    Returns:
        bond_vec: Bond vectors, shape (num_edges, 3)
        bond_dist: Bond distances, shape (num_edges,)
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]
    src_pos = pos[src_idx]
    dst_pos = pos[dst_idx]

    if pbc_offshift is not None:
        dst_pos = dst_pos + pbc_offshift

    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def radial_message_passing(
    edge_vec_norm: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform radial message passing to aggregate edge information to nodes.

    Args:
        edge_vec_norm: Normalized edge vectors, shape (num_edges, 3)
        edge_attr: Edge attributes, shape (num_edges, 3, units)
        edge_index: Edge indices, shape (2, num_edges)
        num_nodes: Number of nodes

    Returns:
        I: Scalar components, shape (num_nodes, 1, 1, units)
        A: Skew-symmetric components, shape (num_nodes, 3, 3, units)
        S: Traceless symmetric components, shape (num_nodes, 3, 3, units)
    """
    dst = edge_index[1]

    # Create radial tensors from edge vectors
    # Following the original PyG implementation pattern
    # new_radial_tensor does: f_I[..., None, None] * scalars
    # So f_I (num_edges, units) -> (num_edges, units, 1, 1)
    # The original uses: eye (1, 1, 3, 3) which broadcasts with f_I (num_edges, units, 1, 1)
    # Result: (num_edges, units, 3, 3) - we transpose to (num_edges, 3, 3, units)

    # Get units dimension from edge_attr
    units = edge_attr.shape[-1]

    # For scalars: use (1, 1, 1, 1) which will broadcast with f_I
    eye_scalar_base = torch.ones(1, 1, 1, 1, device=edge_vec_norm.device, dtype=edge_vec_norm.dtype)
    A_skew_base = vector_to_skewtensor(edge_vec_norm).unsqueeze(-3)  # (num_edges, 1, 3, 3)
    S_sym_base = vector_to_symtensor(edge_vec_norm).unsqueeze(-3)  # (num_edges, 1, 3, 3)

    # Split edge_attr into three components
    edge_attr_I = edge_attr[:, 0, :]  # (num_edges, units)
    edge_attr_A = edge_attr[:, 1, :]  # (num_edges, units)
    edge_attr_S = edge_attr[:, 2, :]  # (num_edges, units)

    # Call new_radial_tensor with original pattern
    # new_radial_tensor will multiply f_I[..., None, None] * scalars
    # f_I: (num_edges, units) -> (num_edges, units, 1, 1)
    # scalars: (1, 1, 1, 1) -> broadcasts to (num_edges, units, 1, 1)
    # Result: (num_edges, units, 1, 1)
    I_ij, A_ij, S_ij = new_radial_tensor(
        eye_scalar_base,
        A_skew_base,
        S_sym_base,
        edge_attr_I,
        edge_attr_A,
        edge_attr_S,
    )

    # Debug: Check shapes after new_radial_tensor
    # Expected: I_ij (num_edges, units, 1, 1), A_ij (num_edges, units, 3, 3), S_ij (num_edges, units, 3, 3)
    if I_ij.shape[1] != units or A_ij.shape[1] != units or S_ij.shape[1] != units:
        # If units is not in position 1, something went wrong with new_radial_tensor
        # This might happen if the broadcasting didn't work as expected
        raise RuntimeError(
            f"new_radial_tensor returned unexpected shapes: "
            f"I_ij {I_ij.shape} (expected units={units} in pos 1), "
            f"A_ij {A_ij.shape} (expected units={units} in pos 1), "
            f"S_ij {S_ij.shape} (expected units={units} in pos 1)"
        )

    # new_radial_tensor returns shapes based on input shapes
    # f_I[..., None, None] * scalars where scalars is (1, 1, 1, 1) and f_I is (num_edges, units)
    # Result: (num_edges, units, 1, 1)
    # f_A[..., None, None] * skew where skew is (num_edges, 1, 3, 3) and f_A is (num_edges, units)
    # Result: (num_edges, units, 3, 3)
    # We need: (num_edges, 1, 1, units) for I, (num_edges, 3, 3, units) for A and S
    # Transpose: move units dimension from position 1 to position -1

    # After new_radial_tensor, units should be in position 1
    # Always transpose to move units from position 1 to position -1
    # Check actual shapes and transpose accordingly
    if I_ij.dim() == 4:
        if I_ij.shape[1] == units and I_ij.shape[-1] != units:
            # I_ij is (num_edges, units, 1, 1), transpose to (num_edges, 1, 1, units)
            I_ij = I_ij.permute(0, 2, 3, 1)
        elif I_ij.shape[-1] == units and I_ij.shape[1] != units:
            # Already in correct shape (num_edges, 1, 1, units)
            pass
        else:
            # Unexpected shape - try to fix it
            if I_ij.shape[1] == units:
                I_ij = I_ij.permute(0, 2, 3, 1)
    if A_ij.dim() == 4:
        if A_ij.shape[1] == units and A_ij.shape[-1] != units:
            # A_ij is (num_edges, units, 3, 3), transpose to (num_edges, 3, 3, units)
            A_ij = A_ij.permute(0, 2, 3, 1)
        elif A_ij.shape[-1] == units and A_ij.shape[1] != units:
            # Already in correct shape (num_edges, 3, 3, units)
            pass
        else:
            # Unexpected shape - try to fix it
            if A_ij.shape[1] == units:
                A_ij = A_ij.permute(0, 2, 3, 1)
    if S_ij.dim() == 4:
        if S_ij.shape[1] == units and S_ij.shape[-1] != units:
            # S_ij is (num_edges, units, 3, 3), transpose to (num_edges, 3, 3, units)
            S_ij = S_ij.permute(0, 2, 3, 1)
        elif S_ij.shape[-1] == units and S_ij.shape[1] != units:
            # Already in correct shape (num_edges, 3, 3, units)
            pass
        else:
            # Unexpected shape - try to fix it
            if S_ij.shape[1] == units:
                S_ij = S_ij.permute(0, 2, 3, 1)

    # Ensure final shapes are correct before aggregation
    # I_ij should be (num_edges, 1, 1, units)
    # A_ij should be (num_edges, 3, 3, units)
    # S_ij should be (num_edges, 3, 3, units)
    # Verify shapes have units in the last dimension
    assert I_ij.shape[-1] == units, f"I_ij shape {I_ij.shape} should have units={units} in last dim"
    assert A_ij.shape[-1] == units, f"A_ij shape {A_ij.shape} should have units={units} in last dim"
    assert S_ij.shape[-1] == units, f"S_ij shape {S_ij.shape} should have units={units} in last dim"

    # Aggregate to nodes
    I_tensor = scatter_add(I_ij, dst, dim_size=num_nodes, dim=0)
    A = scatter_add(A_ij, dst, dim_size=num_nodes, dim=0)
    S = scatter_add(S_ij, dst, dim_size=num_nodes, dim=0)

    return I_tensor, A, S


def message_passing(
    I_tensor: torch.Tensor,
    A: torch.Tensor,
    S: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform message passing for tensor components.

    Args:
        I_tensor: Scalar components, shape (num_nodes, 1, 1, units)
        A: Skew-symmetric components, shape (num_nodes, 3, 3, units)
        S: Traceless symmetric components, shape (num_nodes, 3, 3, units)
        edge_attr: Edge attributes, shape (num_edges, 3, units)
        edge_index: Edge indices, shape (2, num_edges)
        num_nodes: Number of nodes

    Returns:
        Im: Aggregated scalar messages, shape (num_nodes, 1, 1, units)
        Am: Aggregated skew messages, shape (num_nodes, 3, 3, units)
        Sm: Aggregated traceless messages, shape (num_nodes, 3, 3, units)
    """
    dst = edge_index[1]

    # Get node features for destination nodes
    I_j = I_tensor[dst]
    A_j = A[dst]
    S_j = S[dst]

    # Create radial tensors from edge attributes
    # edge_attr has shape (num_edges, units, 3) where the last dim is (I, A, S) components
    # We need to extract each component and expand to match node feature shapes
    edge_attr_I = edge_attr[..., 0]  # (num_edges, units)
    edge_attr_A = edge_attr[..., 1]  # (num_edges, units)
    edge_attr_S = edge_attr[..., 2]  # (num_edges, units)

    # After linear transformations, I, A, S all have shape (num_nodes, 3, 3, units)
    # So I_j, A_j, S_j have shape (num_edges, 3, 3, units)
    # Apply edge attributes directly: multiply edge_attr with the last dimension (units)
    # edge_attr_I: (num_edges, units) -> (num_edges, 1, 1, units) for broadcasting
    # edge_attr_A: (num_edges, units) -> (num_edges, 1, 1, units) for broadcasting
    # edge_attr_S: (num_edges, units) -> (num_edges, 1, 1, units) for broadcasting
    edge_attr_I_expanded = edge_attr_I.unsqueeze(1).unsqueeze(1)  # (num_edges, 1, 1, units)
    edge_attr_A_expanded = edge_attr_A.unsqueeze(1).unsqueeze(1)  # (num_edges, 1, 1, units)
    edge_attr_S_expanded = edge_attr_S.unsqueeze(1).unsqueeze(1)  # (num_edges, 1, 1, units)

    # Apply edge attributes to node features
    I_m = I_j * edge_attr_I_expanded  # (num_edges, 3, 3, units)
    A_m = A_j * edge_attr_A_expanded  # (num_edges, 3, 3, units)
    S_m = S_j * edge_attr_S_expanded  # (num_edges, 3, 3, units)

    # Aggregate messages
    Im = scatter_add(I_m, dst, dim_size=num_nodes, dim=0)
    Am = scatter_add(A_m, dst, dim_size=num_nodes, dim=0)
    Sm = scatter_add(S_m, dst, dim_size=num_nodes, dim=0)

    return Im, Am, Sm


def tensor_matmul_o3(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """O(3) equivariant tensor multiplication.

    Args:
        X: First tensor, shape (num_nodes, 3, 3, units)
        Y: Second tensor, shape (num_nodes, 3, 3, units)

    Returns:
        Result tensor, shape (num_nodes, 3, 3, units)
    """
    # O(3) equivariant: A + B where A = X @ Y, B = Y @ X
    A = torch.einsum("nijk,njlk->nilk", X, Y)
    B = torch.einsum("nijk,njlk->nilk", Y, X)
    return A + B


def tensor_matmul_so3(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """SO(3) equivariant tensor multiplication.

    Args:
        X: First tensor, shape (num_nodes, 3, 3, units)
        Y: Second tensor, shape (num_nodes, 3, 3, units)

    Returns:
        Result tensor, shape (num_nodes, 3, 3, units)
    """
    # SO(3) equivariant: 2 * (X @ Y)
    return 2 * torch.einsum("nijk,njlk->nilk", X, Y)


class TensorEmbedding(nn.Module):
    """Pure PyTorch TensorNet embedding layer."""

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        activation: nn.Module,
        ntypes_node: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
    ):
        super().__init__()
        self.units = units
        self.cutoff = cutoff

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

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Node types, shape (num_nodes,)
            edge_index: Edge indices, shape (2, num_edges)
            edge_weight: Edge weights (distances), shape (num_edges,)
            edge_vec: Edge vectors, shape (num_edges, 3)
            edge_attr: Edge attributes (RBF), shape (num_edges, num_rbf)

        Returns:
            X: Tensor representation, shape (num_nodes, 3, 3, units)
        """
        num_nodes = z.shape[0]

        # Node embedding
        x = self.emb(z)  # (num_nodes, units)

        # Edge processing
        C = cosine_cutoff(edge_weight, self.cutoff)
        W1 = self.distance_proj1(edge_attr) * C.view(-1, 1)  # (num_edges, units)
        W2 = self.distance_proj2(edge_attr) * C.view(-1, 1)
        W3 = self.distance_proj3(edge_attr) * C.view(-1, 1)

        edge_vec_norm = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)

        # Get atomic number messages
        src, dst = edge_index[0], edge_index[1]
        vi = x[src]
        vj = x[dst]
        zij = torch.cat([vi, vj], dim=-1)
        Zij = self.emb2(zij)  # (num_edges, units)

        # Create edge attributes with Zij
        edge_attr_processed = torch.stack([W1, W2, W3], dim=1)  # (num_edges, 3, units)
        edge_attr_processed = edge_attr_processed * Zij.unsqueeze(1)  # (num_edges, 3, units)

        # Radial message passing
        I_tensor, A, S = radial_message_passing(edge_vec_norm, edge_attr_processed, edge_index, num_nodes)

        # Compose initial tensor to get proper shape for norm computation
        X = compose_tensor(I_tensor, A, S)  # (num_nodes, 3, 3, units)

        # Verify X has correct shape before tensor_norm
        assert X.shape[-1] == self.units, (
            f"X shape {X.shape} should have units={self.units} in last dim, got {X.shape[-1]}"
        )

        # Normalize and process
        # Following original: norm = tensor_norm(scalars + skew_matrices + traceless_tensors)
        # For X with shape (num_nodes, 3, 3, units), we need to sum over (-3, -2)
        # which are the (3, 3) spatial dimensions
        # tensor_norm sums over (-2, -1), but we need (-3, -2) for our tensor shape
        # So we compute the norm manually: sum over the spatial (3, 3) dimensions
        norm = (X**2).sum((-3, -2))  # (num_nodes, units)

        # Verify norm has correct shape before LayerNorm
        assert norm.shape[-1] == self.units, (
            f"norm shape {norm.shape} should have units={self.units} in last dim, got {norm.shape[-1]}"
        )

        norm = self.init_norm(norm)  # (num_nodes, units)

        # Apply tensor linear transformations
        # I_tensor has shape (num_nodes, 1, 1, units), A and S have (num_nodes, 3, 3, units)
        # The linear layer expects (..., units) as the last dimension
        # Original code: permute(0, 2, 3, 1) puts units in position -2, then linear, then permute back
        # For (num_nodes, 3, 3, units): permute(0, 2, 3, 1) -> (num_nodes, 3, units, 3)
        # But linear expects (..., units), so we need to reshape or use a different approach
        # Actually, the linear is applied to each spatial position independently
        # So we reshape to (num_nodes * 3 * 3, units), apply linear, reshape back
        if I_tensor.shape[1] == 1 and I_tensor.shape[2] == 1:
            # Expand I_tensor from (num_nodes, 1, 1, units) to (num_nodes, 3, 3, units)
            eye = torch.eye(3, 3, device=I_tensor.device, dtype=I_tensor.dtype)  # (3, 3)
            I_values = I_tensor.squeeze(1).squeeze(1)  # (num_nodes, units)
            I_expanded = I_values.unsqueeze(1).unsqueeze(1) * eye.unsqueeze(0).unsqueeze(-1)  # (num_nodes, 3, 3, units)
            # Reshape to (num_nodes * 3 * 3, units), apply linear, reshape back
            I_reshaped = I_expanded.reshape(-1, self.units)  # (num_nodes * 9, units)
            I_reshaped = self.linears_tensor[0](I_reshaped)  # (num_nodes * 9, units)
            I_tensor = I_reshaped.reshape(I_expanded.shape)  # (num_nodes, 3, 3, units)
        else:
            # Reshape to (num_nodes * 3 * 3, units), apply linear, reshape back
            I_reshaped = I_tensor.reshape(-1, self.units)  # (num_nodes * 9, units)
            I_reshaped = self.linears_tensor[0](I_reshaped)  # (num_nodes * 9, units)
            I_tensor = I_reshaped.reshape(I_tensor.shape)  # (num_nodes, 3, 3, units)

        # Same for A and S
        A_reshaped = A.reshape(-1, self.units)  # (num_nodes * 9, units)
        A_reshaped = self.linears_tensor[1](A_reshaped)  # (num_nodes * 9, units)
        A = A_reshaped.reshape(A.shape)  # (num_nodes, 3, 3, units)

        S_reshaped = S.reshape(-1, self.units)  # (num_nodes * 9, units)
        S_reshaped = self.linears_tensor[2](S_reshaped)  # (num_nodes * 9, units)
        S = S_reshaped.reshape(S.shape)  # (num_nodes, 3, 3, units)

        # Process norm through scalar layers
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))

        norm = norm.reshape(norm.shape[0], self.units, 3)
        norm_I, norm_A, norm_S = norm[..., 0], norm[..., 1], norm[..., 2]

        # Apply norm to tensors
        I_tensor = I_tensor * norm_I.unsqueeze(1).unsqueeze(1)
        A = A * norm_A.unsqueeze(1).unsqueeze(1)
        S = S * norm_S.unsqueeze(1).unsqueeze(1)

        X = compose_tensor(I_tensor, A, S)

        return X


class TensorNetInteraction(nn.Module):
    """Pure PyTorch TensorNet interaction layer."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff
        self.equivariance_invariance_group = equivariance_invariance_group

        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(num_rbf, units, bias=True, dtype=dtype),
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )

        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(6)])

        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.linears_tensor:
            nn.init.xavier_uniform_(linear.weight)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X: Node tensor representations, shape (num_nodes, 3, 3, units)
            edge_index: Edge indices, shape (2, num_edges)
            edge_weight: Edge weights (distances), shape (num_edges,)
            edge_attr: Edge attributes (RBF), shape (num_edges, num_rbf)

        Returns:
            X: Updated tensor representations, shape (num_nodes, 3, 3, units)
        """
        num_nodes = X.shape[0]

        # Process edge attributes
        C = cosine_cutoff(edge_weight, self.cutoff)
        edge_attr_processed = edge_attr
        for linear_scalar in self.linears_scalar:
            edge_attr_processed = self.act(linear_scalar(edge_attr_processed))
        edge_attr_processed = (edge_attr_processed * C.view(-1, 1)).reshape(
            edge_attr.shape[0], self.units, 3
        )  # (num_edges, units, 3)

        # Normalize input tensor
        # For X with shape (num_nodes, 3, 3, units), we need to sum over (-3, -2)
        # which are the (3, 3) spatial dimensions to get (num_nodes, units)
        norm_X = (X**2).sum((-3, -2)) + 1  # (num_nodes, units)
        X = X / norm_X.reshape(-1, 1, 1, X.shape[-1])

        # Decompose input tensor
        # X has shape (num_nodes, 3, 3, units)
        # decompose_tensor expects (..., 3, 3), so we permute to (num_nodes, units, 3, 3)
        # then apply decompose_tensor which works on the last two dimensions (3, 3)
        X_permuted = X.permute(0, 3, 1, 2)  # (num_nodes, units, 3, 3)
        # decompose_tensor works on last two dims, so this will work for each (num_nodes, units) slice
        I_permuted, A_permuted, S_permuted = decompose_tensor(X_permuted)  # Each: (num_nodes, units, 3, 3)
        # Permute back: (num_nodes, units, 3, 3) -> (num_nodes, 3, 3, units)
        I_tensor = I_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)
        A = A_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)
        S = S_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)

        # Apply tensor linear transformations
        # Reshape to (num_nodes * 9, units), apply linear, reshape back
        I_reshaped = I_tensor.reshape(-1, self.units)  # (num_nodes * 9, units)
        I_reshaped = self.linears_tensor[0](I_reshaped)  # (num_nodes * 9, units)
        I_tensor = I_reshaped.reshape(I_tensor.shape)  # (num_nodes, 3, 3, units)

        A_reshaped = A.reshape(-1, self.units)  # (num_nodes * 9, units)
        A_reshaped = self.linears_tensor[1](A_reshaped)  # (num_nodes * 9, units)
        A = A_reshaped.reshape(A.shape)  # (num_nodes, 3, 3, units)

        S_reshaped = S.reshape(-1, self.units)  # (num_nodes * 9, units)
        S_reshaped = self.linears_tensor[2](S_reshaped)  # (num_nodes * 9, units)
        S = S_reshaped.reshape(S.shape)  # (num_nodes, 3, 3, units)
        Y = compose_tensor(I_tensor, A, S)

        # Message passing
        Im, Am, Sm = message_passing(I_tensor, A, S, edge_attr_processed, edge_index, num_nodes)
        msg = compose_tensor(Im, Am, Sm)

        # Apply group action
        if self.equivariance_invariance_group == "O(3)":
            C = tensor_matmul_o3(Y, msg)  # (num_nodes, 3, 3, units)
        elif self.equivariance_invariance_group == "SO(3)":
            C = tensor_matmul_so3(Y, msg)  # (num_nodes, 3, 3, units)
            C = 2 * C
        else:
            raise ValueError("equivariance_invariance_group must be 'O(3)' or 'SO(3)'")

        # decompose_tensor expects (..., 3, 3), so permute to (num_nodes, units, 3, 3)
        C_permuted = C.permute(0, 3, 1, 2)  # (num_nodes, units, 3, 3)
        I_permuted, A_permuted, S_permuted = decompose_tensor(C_permuted)  # Each: (num_nodes, units, 3, 3)
        # Permute back: (num_nodes, units, 3, 3) -> (num_nodes, 3, 3, units)
        I_tensor = I_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)
        A = A_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)
        S = S_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)

        # Normalize
        # For compose_tensor(I_tensor, A, S) with shape (num_nodes, 3, 3, units),
        # we need to sum over (-3, -2) to get (num_nodes, units)
        X_composed = compose_tensor(I_tensor, A, S)  # (num_nodes, 3, 3, units)
        normp1 = ((X_composed**2).sum((-3, -2)) + 1).reshape(-1, 1, 1, X_composed.shape[-1])
        I_tensor, A, S = I_tensor / normp1, A / normp1, S / normp1

        # Final tensor transformations
        # Reshape to (num_nodes * 9, units), apply linear, reshape back
        I_reshaped = I_tensor.reshape(-1, self.units)  # (num_nodes * 9, units)
        I_reshaped = self.linears_tensor[3](I_reshaped)  # (num_nodes * 9, units)
        I_tensor = I_reshaped.reshape(I_tensor.shape)  # (num_nodes, 3, 3, units)

        A_reshaped = A.reshape(-1, self.units)  # (num_nodes * 9, units)
        A_reshaped = self.linears_tensor[4](A_reshaped)  # (num_nodes * 9, units)
        A = A_reshaped.reshape(A.shape)  # (num_nodes, 3, 3, units)

        S_reshaped = S.reshape(-1, self.units)  # (num_nodes * 9, units)
        S_reshaped = self.linears_tensor[5](S_reshaped)  # (num_nodes * 9, units)
        S = S_reshaped.reshape(S.shape)  # (num_nodes, 3, 3, units)
        dX = compose_tensor(I_tensor, A, S)

        # Update
        X = X + dX + torch.einsum("nijk,njlk->nilk", dX, dX)

        return X


class WeightedAtomReadOut(nn.Module):
    """Pure PyTorch weighted atom readout."""

    def __init__(self, in_feats: int, dims: list[int], activation: nn.Module):
        super().__init__()
        self.dims = [in_feats, *dims]
        self.activation = activation
        self.mlp = MLP(dims=self.dims, activation=self.activation, activate_last=True)
        self.weight = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, node_feat: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            node_feat: Node features, shape (num_nodes, in_feats)
            batch: Batch indices, shape (num_nodes,). If None, assumes single graph.

        Returns:
            Graph-level features, shape (num_graphs, output_dim)
        """
        h = self.mlp(node_feat)
        w = self.weight(node_feat)

        weighted_h = h * w

        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            out = torch.zeros(num_graphs, weighted_h.size(1), device=weighted_h.device, dtype=weighted_h.dtype)
            out.index_add_(0, batch.to(torch.long), weighted_h)
        else:
            out = weighted_h.sum(dim=0, keepdim=True)

        return out


class ReduceReadOut(nn.Module):
    """Pure PyTorch reduce readout."""

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        super().__init__()
        self.op = op
        self.field = field
        if op not in ["mean", "sum", "max"]:
            raise ValueError("op must be 'mean', 'sum', or 'max'")

    def forward(self, node_feat: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            node_feat: Node features, shape (num_nodes, feat_dim)
            batch: Batch indices, shape (num_nodes,). If None, assumes single graph.

        Returns:
            Graph-level features, shape (num_graphs, feat_dim)
        """
        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            if self.op == "sum":
                out = torch.zeros(num_graphs, node_feat.size(1), device=node_feat.device, dtype=node_feat.dtype)
                out.index_add_(0, batch.to(torch.long), node_feat)
            elif self.op == "mean":
                out = torch.zeros(num_graphs, node_feat.size(1), device=node_feat.device, dtype=node_feat.dtype)
                out.index_add_(0, batch.to(torch.long), node_feat)
                counts = torch.zeros(num_graphs, device=node_feat.device, dtype=torch.long)
                counts.index_add_(0, batch.to(torch.long), torch.ones_like(batch, dtype=torch.long))
                out = out / counts.unsqueeze(1).clamp(min=1)
            else:  # max
                out = torch.full(
                    (num_graphs, node_feat.size(1)),
                    float("-inf"),
                    device=node_feat.device,
                    dtype=node_feat.dtype,
                )
                out.index_reduce_(0, batch.to(torch.long), node_feat, "amax", include_self=False)
        else:
            if self.op == "sum":
                out = node_feat.sum(dim=0, keepdim=True)
            elif self.op == "mean":
                out = node_feat.mean(dim=0, keepdim=True)
            else:  # max
                out = node_feat.max(dim=0, keepdim=True)[0]

        return out


class WeightedReadOut(nn.Module):
    """Pure PyTorch weighted readout for atomic properties."""

    def __init__(self, in_feats: int, dims: list[int], num_targets: int):
        super().__init__()
        from matgl.layers._core import GatedMLP

        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            node_feat: Node features, shape (num_nodes, in_feats)

        Returns:
            Atomic properties, shape (num_nodes, num_targets)
        """
        return self.gated(node_feat)


class TensorNet(MatGLModel):
    """The main TensorNet model. The official implementation can be found in https://github.com/torchmd/torchmd-net."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        units: int = 64,
        ntypes_state: int | None = None,
        dim_state_embedding: int = 0,
        dim_state_feats: int | None = None,
        include_state: bool = False,
        nblocks: int = 2,
        num_rbf: int = 32,
        max_n: int = 3,
        max_l: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        use_smooth: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff: float = 5.0,
        equivariance_invariance_group: str = "O(3)",
        dtype: torch.dtype = matgl.float_th,
        width: float = 0.5,
        readout_type: Literal["weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        **kwargs,
    ):
        r"""

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            units (int, optional): Hidden embedding size.
                (default: :obj:`64`)
            ntypes_state (int): Number of state labels
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            include_state (bool): Whether to include states features
            nblocks (int, optional): The number of interaction layers.
                (default: :obj:`2`)
            num_rbf (int, optional): The number of radial basis Gaussian functions :math:`\mu`.
                (default: :obj:`32`)
            max_n (int): maximum of n in spherical Bessel functions
            max_l (int): maximum of l in spherical Bessel functions
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            use_smooth (bool): Whether to use the smooth version of SphericalBessel functions.
                This is particularly important for the smoothness of PES.
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            cutoff (float): cutoff distance for interatomic interactions.
            equivariance_invariance_group (string, optional): Group under whose action on input
                positions internal tensor features will be equivariant and scalar predictions
                will be invariant. O(3) or SO(3).
               (default :obj:`"O(3)"`)
            dtype (torch.dtype): data type for all variables
            width (float): the width of Gaussian radial basis functions
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            is_intensive (bool): Whether the prediction is intensive
            ntargets (int): Number of target properties
            **kwargs: For future flexibility. Not used at the moment.

        """
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types  # type: ignore

        self.bond_expansion = BondExpansion(
            cutoff=cutoff,
            rbf_type=rbf_type,
            final=cutoff + 1.0,
            num_centers=num_rbf,
            width=width,
            smooth=use_smooth,
            max_n=max_n,
            max_l=max_l,
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], "Unknown group representation. Choose O(3) or SO(3)."

        self.units = units
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = nblocks
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff = cutoff
        self.dim_state_embedding = dim_state_embedding
        self.dim_state_feats = dim_state_feats
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.task_type = task_type

        # make sure the number of radial basis functions correct for tensor embedding
        if rbf_type == "SphericalBessel":
            num_rbf = max_n

        self.tensor_embedding = TensorEmbedding(
            units=units,
            degree_rbf=num_rbf,
            activation=activation,
            ntypes_node=len(element_types),
            cutoff=cutoff,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TensorNetInteraction(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
                for _ in range(nblocks)
                if nblocks != 0
            ]
        )

        self.out_norm = nn.LayerNorm(3 * units, dtype=dtype)
        self.linear = nn.Linear(3 * units, units, dtype=dtype)
        if is_intensive:
            input_feats = units
            if readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type:ignore[assignment]
                readout_feats = units
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=units,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.is_intensive = is_intensive
        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        g: torch.Tensor | dict[str, torch.Tensor] | object,
        state_attr: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Args:
            g: Either a PyG Data object or a dict with keys:
                - 'node_type' or 'z': Node types, shape (num_nodes,)
                - 'pos': Node positions, shape (num_nodes, 3)
                - 'edge_index': Edge indices, shape (2, num_edges)
                - 'pbc_offshift': Optional PBC offsets, shape (num_edges, 3)
                - 'batch': Optional batch indices, shape (num_nodes,)
            state_attr: State attrs for a batch of graphs (not used currently).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs
        """
        # Handle both dict and PyG Data object
        if isinstance(g, dict):
            z = g.get("node_type", g.get("z"))
            pos = g["pos"]
            edge_index = g["edge_index"]
            pbc_offshift = g.get("pbc_offshift", None)
            batch = g.get("batch", None)
            num_graphs = g.get("num_graphs", None)
        else:
            # PyG Data object - extract tensors
            # Type narrowing: check that g has required attributes
            if not hasattr(g, "pos") or not hasattr(g, "edge_index"):
                raise ValueError("Graph object must have 'pos' and 'edge_index' attributes")
            z = getattr(g, "node_type", getattr(g, "z", None))
            if z is None:
                raise ValueError("Graph must have 'node_type' or 'z' attribute")
            pos = g.pos  # type: ignore[union-attr]
            edge_index = g.edge_index  # type: ignore[union-attr]
            pbc_offshift = getattr(g, "pbc_offshift", None)
            batch = getattr(g, "batch", None)
            num_graphs = getattr(g, "num_graphs", None)

        # Obtain graph, with distances and relative position vectors
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)

        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(bond_dist)

        # Embedding layer
        X = self.tensor_embedding(z, edge_index, bond_dist, bond_vec, edge_attr)

        # Interaction layers
        for layer in self.layers:
            X = layer(X, edge_index, bond_dist, edge_attr)

        # decompose_tensor expects (..., 3, 3), so permute to (num_nodes, units, 3, 3)
        # X has shape (num_nodes, 3, 3, units)
        X_permuted = X.permute(0, 3, 1, 2)  # (num_nodes, units, 3, 3)
        scalars_permuted, skew_metrices_permuted, traceless_tensors_permuted = decompose_tensor(X_permuted)
        # Permute back: (num_nodes, units, 3, 3) -> (num_nodes, 3, 3, units)
        scalars = scalars_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)
        skew_metrices = skew_metrices_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)
        traceless_tensors = traceless_tensors_permuted.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)

        # tensor_norm sums over (-2, -1), but for (num_nodes, 3, 3, units) we need to sum over (-3, -2)
        # to get (num_nodes, units)
        scalars_norm = (scalars**2).sum((-3, -2))  # (num_nodes, units)
        skew_norm = (skew_metrices**2).sum((-3, -2))  # (num_nodes, units)
        traceless_norm = (traceless_tensors**2).sum((-3, -2))  # (num_nodes, units)
        x = torch.cat((scalars_norm, skew_norm, traceless_norm), dim=-1)  # (num_nodes, 3 * units)
        x = self.out_norm(x)
        x = self.linear(x)

        if self.is_intensive:
            node_vec = self.readout(x, batch)
            vec = node_vec  # type: ignore
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
            return torch.squeeze(output)

        atomic_energies = self.final_layer(x)
        if batch is not None:
            # edge case, if we do squeeze() directly, we will get torch.size([]) and it will crash in the training.
            if atomic_energies.shape == (1, 1):
                atomic_energies = atomic_energies.squeeze(-1)
            else:
                atomic_energies = atomic_energies.squeeze()
            # Batch case: Use scatter_add with batch tensor
            batch_long = batch.to(torch.long)
            if num_graphs is None:
                num_graphs = int(batch_long.max().item()) + 1
            return scatter_add(atomic_energies, batch_long, dim_size=num_graphs)
        # Single graph case: Sum all energies (equivalent to scatter_add with all nodes in one graph)
        return torch.sum(atomic_energies, dim=0, keepdim=True).squeeze()

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
