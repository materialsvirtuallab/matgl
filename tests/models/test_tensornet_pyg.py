from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from torch_geometric.data import Data

from matgl.layers import ActivationFunction, BondExpansion
from matgl.layers._embedding_pyg import TensorEmbedding as TensorEmbeddingPyG
from matgl.layers._readout_pyg import ReduceReadOut as ReduceReadOutPyG
from matgl.layers._readout_pyg import WeightedReadOut as WeightedReadOutPyG
from matgl.models._tensornet_pyg import ReduceReadOut as ReduceReadOutPure
from matgl.models._tensornet_pyg import TensorEmbedding, TensorNet
from matgl.models._tensornet_pyg import WeightedReadOut as WeightedReadOutPure


class TestTensorNet:
    def test_model(self, graph_MoS_pyg):
        _, graph, _ = graph_MoS_pyg
        for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
            model = TensorNet(is_intensive=False, activation_type=act)
            output = model(g=graph)
            assert torch.numel(output) == 1
        model.save(".")
        TensorNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")
        model = TensorNet(is_intensive=False, equivariance_invariance_group="SO(3)")
        assert torch.numel(output) == 1

    def test_exceptions(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            _ = TensorNet(element_types=None, is_intensive=False, activation_type="whatever")
        with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
            _ = TensorNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")

    def test_model_intensive(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True)
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_weighted_atom(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_ReduceReadOut(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(is_intensive=True, readout_type="reduce_atom")
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_classification(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(
            element_types=["Mo", "S"],
            is_intensive=True,
            task_type="classification",
        )
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_tensor_embedding_comparison(self, graph_MoS_pyg):
        """Test that pure PyTorch TensorEmbedding produces same output as PyG version."""
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]

        # Extract graph attributes
        z = graph.node_type
        edge_index = graph.edge_index
        edge_weight = graph.bond_dist
        edge_vec = graph.bond_vec

        # Model parameters
        units = 64
        num_rbf = 32  # Standard number of RBF features
        ntypes_node = len(torch.unique(z))
        cutoff = 5.0
        activation = ActivationFunction["swish"].value()

        # Compute edge_attr using BondExpansion (same as TensorNet does)
        bond_expansion = BondExpansion(
            rbf_type="Gaussian",
            num_centers=num_rbf,
            cutoff=cutoff,
            final=cutoff + 1.0,
        )
        edge_attr = bond_expansion(edge_weight)
        degree_rbf = edge_attr.shape[1]

        # Set edge_attr on graph for PyG version
        graph.edge_attr = edge_attr

        # Create both embedding layers
        embedding_pyg = TensorEmbeddingPyG(
            units=units,
            degree_rbf=degree_rbf,
            activation=activation,
            ntypes_node=ntypes_node,
            cutoff=cutoff,
            dtype=matgl.float_th,
        )

        embedding_pure = TensorEmbedding(
            units=units,
            degree_rbf=degree_rbf,
            activation=activation,
            ntypes_node=ntypes_node,
            cutoff=cutoff,
            dtype=matgl.float_th,
        )

        # Copy weights from PyG version to pure version to ensure equivalence
        embedding_pure.distance_proj1.load_state_dict(embedding_pyg.distance_proj1.state_dict())
        embedding_pure.distance_proj2.load_state_dict(embedding_pyg.distance_proj2.state_dict())
        embedding_pure.distance_proj3.load_state_dict(embedding_pyg.distance_proj3.state_dict())
        embedding_pure.emb.load_state_dict(embedding_pyg.emb.state_dict())
        embedding_pure.emb2.load_state_dict(embedding_pyg.emb2.state_dict())
        for i in range(3):
            embedding_pure.linears_tensor[i].load_state_dict(embedding_pyg.linears_tensor[i].state_dict())
        for i in range(2):
            embedding_pure.linears_scalar[i].load_state_dict(embedding_pyg.linears_scalar[i].state_dict())
        embedding_pure.init_norm.load_state_dict(embedding_pyg.init_norm.state_dict())

        # Set both to eval mode for consistent behavior
        embedding_pyg.eval()
        embedding_pure.eval()

        # Forward pass through PyG version
        with torch.no_grad():
            X_pyg, state_feat_pyg = embedding_pyg(graph, state_attr=None)

        # Forward pass through pure PyTorch version
        with torch.no_grad():
            X_pure = embedding_pure(z, edge_index, edge_weight, edge_vec, edge_attr)

        # Compare outputs
        # Note: PyG version returns (num_nodes, units, 3, 3) while pure version returns (num_nodes, 3, 3, units)
        # Permute PyG output to match pure version: (num_nodes, 3, 3, units) -> (num_nodes, 3, 3, units)
        X_pyg_permuted = X_pyg.permute(0, 2, 3, 1)  # (num_nodes, 3, 3, units)

        # Note: There may be small numerical differences due to implementation details,
        # so we use a reasonable tolerance
        assert X_pyg_permuted.shape == X_pure.shape, (
            f"Shape mismatch: PyG {X_pyg_permuted.shape} vs Pure {X_pure.shape}"
        )
        assert torch.allclose(X_pyg_permuted, X_pure, rtol=1e-4, atol=1e-5), (
            f"Output mismatch: max diff = {torch.max(torch.abs(X_pyg_permuted - X_pure)).item()}"
        )
        assert state_feat_pyg is None, "State features should be None"

    def test_readout_consistency(self):
        torch.manual_seed(0)
        in_feats = 16
        hidden_dims = [32]
        num_targets = 3
        num_nodes = 10
        node_feat = torch.randn(num_nodes, in_feats, dtype=matgl.float_th)
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        data = Data(node_feat=node_feat.clone(), batch=batch)

        readout_pyg = WeightedReadOutPyG(in_feats=in_feats, dims=hidden_dims, num_targets=num_targets)
        readout_pure = WeightedReadOutPure(in_feats=in_feats, dims=hidden_dims, num_targets=num_targets)
        readout_pure.gated.load_state_dict(readout_pyg.gated.state_dict())

        with torch.no_grad():
            out_pyg_weighted = readout_pyg(data)
            out_pure_weighted = readout_pure(node_feat)
        assert torch.allclose(out_pyg_weighted, out_pure_weighted, rtol=1e-5, atol=1e-6)

        for op in ["sum", "mean", "max"]:
            reduce_pyg = ReduceReadOutPyG(op=op)
            reduce_pure = ReduceReadOutPure(op=op)
            with torch.no_grad():
                out_pyg_reduce = reduce_pyg(data)
                out_pure_reduce = reduce_pure(node_feat, batch)
            assert torch.allclose(out_pyg_reduce, out_pure_reduce, rtol=1e-5, atol=1e-6)
