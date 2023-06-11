from __future__ import annotations

import unittest
from collections import namedtuple

import dgl
import numpy as np
import torch
from pymatgen.core.structure import Lattice, Structure
from torch import nn

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._graph_convolution import (
    MLP,
    M3GNetBlock,
    M3GNetGraphConv,
    MEGNetBlock,
    MEGNetGraphConv,
)
from matgl.layers._three_body import SphericalBesselWithHarmonics
from matgl.utils.cutoff import polynomial_cutoff

Graph = namedtuple("Graph", "graph, state_attr")


def build_graph(N, E, NDIM=5, EDIM=3, GDIM=10):
    graph = dgl.rand_graph(N, E)
    graph.ndata["node_feat"] = torch.rand(N, NDIM)
    graph.edata["edge_feat"] = torch.rand(E, EDIM)
    state_attr = torch.rand(1, GDIM)
    return Graph(graph, state_attr)


def get_graphs(num_graphs, NDIM=5, EDIM=3, GDIM=10):
    Ns = torch.randint(10, 30, (num_graphs,)).tolist()
    Es = torch.randint(35, 100, (num_graphs,)).tolist()
    graphs = [build_graph(*gspec, NDIM, EDIM, GDIM) for gspec in zip(Ns, Es)]
    return graphs


def batch(state_attrs_lists):
    graphs, attrs = list(zip(*state_attrs_lists))
    batched_graph = dgl.batch(graphs)
    batched_attrs = torch.vstack(attrs)
    return batched_graph, batched_attrs


def test_megnet_layer():
    graphs = get_graphs(5)
    batched_graph, attrs = batch(graphs)

    NDIM, EDIM, GDIM = 5, 3, 10
    edge_func = MLP(dims=[2 * NDIM + EDIM + GDIM, EDIM])
    node_func = MLP(dims=[EDIM + NDIM + GDIM, NDIM])
    state_func = MLP(dims=[EDIM + NDIM + GDIM, GDIM])
    layer = MEGNetGraphConv(edge_func, node_func, state_func)

    # one pass
    edge_feat = batched_graph.edata.pop("edge_feat")
    node_feat = batched_graph.ndata.pop("node_feat")
    out = layer(batched_graph, edge_feat, node_feat, attrs)
    return out


def test_megnet_block():
    DIM = 5
    N1, N2 = 64, 32
    block = MEGNetBlock(
        dims=[5, 10, 13],
        conv_hiddens=[N1, N1, N2],
        act=nn.SiLU(),
        skip=False,
    )
    graphs = get_graphs(5, NDIM=DIM, EDIM=DIM, GDIM=DIM)
    batched_graph, attrs = batch(graphs)

    # one pass
    edge_feat = batched_graph.edata.pop("edge_feat")
    node_feat = batched_graph.ndata.pop("node_feat")
    out = block(batched_graph, edge_feat, node_feat, attrs)
    return out


class TestGraphConv(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        cls.s.states = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        element_types = get_element_list([cls.s])
        p2g = Structure2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph(cls.s)
        cls.g1 = graph
        cls.state1 = state
        bond_vec, bond_dist = compute_pair_vector_and_distance(cls.g1)
        cls.g1.edata["bond_dist"] = bond_dist
        cls.g1.edata["bond_vec"] = bond_vec

    def test_m3gnet_graph_conv(self):
        bond_dist = self.g1.edata["bond_dist"]
        polynomial_cutoff(bond_dist, 4.0)
        bond_expansion = BondExpansion(max_l=3, max_n=3, cutoff=5.0, rbf_type="SphericalBessel", smooth=False)
        bond_basis = bond_expansion(bond_dist)
        self.g1.edata["rbf"] = bond_basis
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        sb_and_sh(l_g1)
        max_n = 3
        max_l = 3
        num_node_feats = 16
        num_edge_feats = 24
        num_state_feats = 32
        node_attr = self.g1.ndata["attr"]
        state_attr = torch.tensor([0.0, 0.0])
        embedding = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=num_node_feats,
            dim_edge_embedding=num_edge_feats,
            dim_state_feats=num_state_feats,
            include_state=True,
            activation=nn.SiLU(),
        )
        node_feat, edge_feat, state_feat = embedding(node_attr, bond_basis, state_attr)
        node_dim = node_feat.shape[-1]
        edge_dim = edge_feat.shape[-1]
        state_dim = state_feat.shape[-1]

        conv_hiddens = [32, 32]
        edge_in = 2 * node_dim + edge_dim + state_dim
        node_in = 2 * node_dim + edge_dim + state_dim
        state_in = node_dim + state_dim
        degree = max_n * max_l
        conv = M3GNetGraphConv.from_dims(
            degree=degree,
            include_states=True,
            edge_dims=[edge_in, *conv_hiddens, num_edge_feats],
            node_dims=[node_in, *conv_hiddens, num_node_feats],
            state_dims=[state_in, *conv_hiddens, num_state_feats],
            activation=nn.SiLU(),
        )
        edge_feat_new, node_feat_new, state_feat_new = conv(self.g1, edge_feat, node_feat, state_feat)
        assert [edge_feat_new.size(dim=0), edge_feat_new.size(dim=1)] == [28, 24]
        assert [node_feat_new.size(dim=0), node_feat_new.size(dim=1)] == [2, 16]
        assert [state_feat_new.size(dim=0), state_feat_new.size(dim=1)] == [1, 32]

    def test_m3gnet_block(self):
        polynomial_cutoff(self.g1.edata["bond_dist"], 4.0)
        bond_expansion = BondExpansion(max_l=3, max_n=3, cutoff=5.0, rbf_type="SphericalBessel", smooth=False)
        bond_basis = bond_expansion(self.g1.edata["bond_dist"])
        self.g1.edata["rbf"] = bond_basis
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        sb_and_sh(l_g1)
        num_node_feats = 16
        num_edge_feats = 32
        num_state_feats = 64
        state_attr = torch.tensor([0.0, 0.0])
        embedding = EmbeddingBlock(
            degree_rbf=9,
            dim_node_embedding=num_node_feats,
            dim_edge_embedding=num_edge_feats,
            dim_state_feats=num_state_feats,
            include_state=True,
            activation=nn.SiLU(),
        )
        node_attr = self.g1.ndata["attr"]
        edge_attr = bond_basis
        node_feat, edge_feat, state_feat = embedding(node_attr, edge_attr, state_attr)
        self.g1.ndata["node_feat"] = node_feat
        self.g1.edata["edge_feat"] = edge_feat
        graph_conv = M3GNetBlock(
            degree=3 * 3,
            activation=nn.SiLU(),
            conv_hiddens=[32, 16],
            num_node_feats=num_node_feats,
            num_edge_feats=num_edge_feats,
            num_state_feats=num_state_feats,
            include_state=True,
        )
        edge_feat_new, node_feat_new, state_feat_new = graph_conv(self.g1, edge_feat, node_feat, state_feat)
        assert [edge_feat_new.size(dim=0), edge_feat_new.size(dim=1)] == [28, 32]
        assert [node_feat_new.size(dim=0), node_feat_new.size(dim=1)] == [2, 16]
        assert [state_feat_new.size(dim=0), state_feat_new.size(dim=1)] == [1, 64]

        # without state features
        state_feat = None
        graph_conv = M3GNetBlock(
            degree=3 * 3,
            num_node_feats=num_node_feats,
            num_edge_feats=num_edge_feats,
            num_state_feats=num_state_feats,
            conv_hiddens=[32, 16],
            activation=nn.SiLU(),
            include_state=False,
        )
        edge_feat_new, node_feat_new, state_feat_new = graph_conv(self.g1, edge_feat, node_feat, state_feat)
        assert [edge_feat_new.size(dim=0), edge_feat_new.size(dim=1)] == [28, 32]
        assert [node_feat_new.size(dim=0), node_feat_new.size(dim=1)] == [2, 16]


if __name__ == "__main__":
    test_megnet_layer()
    test_megnet_block()
    unittest.main()
