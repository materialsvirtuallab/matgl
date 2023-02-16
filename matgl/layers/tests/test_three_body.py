import unittest

import torch
import torch.nn as nn
from pymatgen.core.structure import Lattice, Molecule, Structure

from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)
from matgl.graph.converters import Pmg2Graph, get_element_list
from matgl.layers.bond_expansion import BondExpansion
from matgl.layers.core import MLP, GatedMLP
from matgl.layers.cutoff_functions import polynomial_cutoff
from matgl.layers.embedding_block import EmbeddingBlock
from matgl.layers.three_body import SphericalBesselWithHarmonics, ThreeBodyInteractions


class TestThreeBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.01, 0.0, 0.0], [0.5, 0.5, 0.5]])
        mol = Molecule(["C", "O"], [[0, 0, 0], [1.1, 0, 0]])

        element_types = get_element_list([cls.s])
        p2g = Pmg2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph_from_structure(cls.s)
        cls.g1 = graph
        cls.state1 = state
        bond_vec, bond_dist = compute_pair_vector_and_distance(cls.g1)
        cls.g1.edata["bond_dist"] = bond_dist
        cls.g1.edata["bond_vec"] = bond_vec

        element_types = get_element_list([mol])
        p2g = Pmg2Graph(element_types=element_types, cutoff=4.0)
        graph, state = p2g.get_graph_from_molecule(mol)
        cls.g2 = graph
        cls.state2 = state

    def test_spherical_bessel_with_harmonics(self):
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(self.g1, l_g1)
        self.assertListEqual([three_body_basis.size(dim=0), three_body_basis.size(dim=1)], [364, 9])

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=2, cutoff=5.0, use_smooth=False, use_phi=True)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(self.g1, l_g1)
        self.assertListEqual([three_body_basis.size(dim=0), three_body_basis.size(dim=1)], [364, 12])

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=True, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(self.g1, l_g1)
        self.assertListEqual([three_body_basis.size(dim=0), three_body_basis.size(dim=1)], [364, 9])

        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=True, use_phi=True)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(self.g1, l_g1)
        self.assertListEqual([three_body_basis.size(dim=0), three_body_basis.size(dim=1)], [364, 27])

    def test_three_body_interactions(self):
        bond_dist = self.g1.edata["bond_dist"]
        bond_dist.requires_grad = True
        three_body_cutoff = polynomial_cutoff(bond_dist, 4.0)
        bond_expansion = BondExpansion(max_l=3, max_n=3, cutoff=5.0, rbf_type="SphericalBessel", smooth=False)
        bond_basis = bond_expansion(bond_dist)
        sb_and_sh = SphericalBesselWithHarmonics(max_n=3, max_l=3, cutoff=5.0, use_smooth=False, use_phi=False)
        l_g1 = create_line_graph(self.g1, threebody_cutoff=4.0)
        l_g1.apply_edges(compute_theta_and_phi)
        three_body_basis = sb_and_sh(self.g1, l_g1)
        max_n = 3
        max_l = 3
        num_node_feats = 16
        num_edge_feats = 16
        state_attr = torch.tensor([0.0, 0.0])
        embedding = EmbeddingBlock(num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
        node_attr = self.g1.ndata["attr"]
        node_attr.requires_grad = True
        node_feat, edge_feat, state_feat = embedding(node_attr, bond_basis, state_attr)
        degree = max_n * max_l
        three_body_interactions = ThreeBodyInteractions(
            update_network_atom=MLP(dims=[num_node_feats, degree], activation=nn.Sigmoid()),
            update_network_bond=GatedMLP(in_feats=degree, dims=[num_edge_feats], use_bias=False),
        )
        edge_feat_updated = three_body_interactions(
            self.g1, l_g1, three_body_basis, three_body_cutoff, node_feat, edge_feat
        )
        self.assertListEqual([edge_feat_updated.size(dim=0), edge_feat_updated.size(dim=1)], [28, 16])


if __name__ == "__main__":
    unittest.main()
