from __future__ import annotations

import unittest

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
from matgl.layers._core import MLP, GatedMLP
from matgl.layers._three_body import SphericalBesselWithHarmonics, ThreeBodyInteractions
from matgl.utils.cutoff import polynomial_cutoff


class TestThreeBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        element_types = get_element_list([cls.s])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, state = p2g.get_graph(cls.s)
        cls.g1 = graph
        cls.state1 = state
        bond_vec, bond_dist = compute_pair_vector_and_distance(cls.g1)
        cls.g1.edata["bond_dist"] = bond_dist
        cls.g1.edata["bond_vec"] = bond_vec

    def test_spherical_bessel_with_harmonics(self):
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 9]

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=2, cutoff=5.0, use_smooth=False, use_phi=True)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 12]

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=True, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 9]

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=True, use_phi=True)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(l_g1)
        assert [three_body_basis.size(dim=0), three_body_basis.size(dim=1)] == [364, 27]

    def test_three_body_interactions(self):
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        bond_expansion = BondExpansion(max_l=3, max_n=3, cutoff=5.0, rbf_type="SphericalBessel", smooth=False)
        bond_basis = bond_expansion(self.g1.edata["bond_dist"])
        self.g1.edata["rbf"] = bond_basis
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        three_body_basis = sb_and_sh(l_g1)
        three_body_cutoff = polynomial_cutoff(self.g1.edata["bond_dist"], 4.0)
        max_n = 3
        max_l = 3
        num_node_feats = 16
        num_edge_feats = 16
        state_attr = torch.tensor([0.0, 0.0])
        embedding = EmbeddingBlock(
            degree_rbf=9, dim_node_embedding=num_node_feats, dim_edge_embedding=num_edge_feats, activation=nn.SiLU()
        )

        node_attr = self.g1.ndata["attr"]
        edge_attr = self.g1.edata["rbf"]
        node_feat, edge_feat, state_feat = embedding(node_attr, edge_attr, state_attr)
        degree = max_n * max_l
        three_body_interactions = ThreeBodyInteractions(
            update_network_atom=MLP(dims=[num_node_feats, degree], activation=nn.Sigmoid(), activate_last=True),
            update_network_bond=GatedMLP(in_feats=degree, dims=[num_edge_feats], use_bias=False),
        )
        edge_feat_updated = three_body_interactions(
            self.g1, l_g1, three_body_basis, three_body_cutoff, node_feat, edge_feat
        )
        assert [edge_feat_updated.size(dim=0), edge_feat_updated.size(dim=1)] == [28, 16]


if __name__ == "__main__":
    unittest.main()
