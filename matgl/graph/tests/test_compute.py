from __future__ import annotations

import unittest

import numpy as np
import torch
from pymatgen.core.structure import Lattice, Molecule, Structure

from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
)


def _loop_indices(bond_atom_indices, pair_dist, cutoff=4.0):
    bin_count = np.bincount(bond_atom_indices[:, 0], minlength=bond_atom_indices[-1, 0] + 1)
    indices = []
    start = 0
    for bcont in bin_count:
        for i in range(bcont):
            for j in range(bcont):
                if start + i == start + j:
                    continue
                if pair_dist[start + i] > cutoff or pair_dist[start + j] > cutoff:
                    continue
                indices.append([start + i, start + j])
        start += bcont
    return np.array(indices)


def _calculate_cos_loop(graph, threebody_cutoff=4.0):
    """
    Calculate the cosine theta of triplets using loops
    Args:
        graph: List
    Returns: a list of cosine theta values
    """

    _, _, n_sites = torch.unique(graph.edges()[0], return_inverse=True, return_counts=True)
    start_index = 0
    cos = []
    for n_site in n_sites:
        for i in range(n_site):
            for j in range(n_site):
                if i == j:
                    continue
                vi = graph.edata["bond_vec"][i + start_index].numpy()
                vj = graph.edata["bond_vec"][j + start_index].numpy()
                di = np.linalg.norm(vi)
                dj = np.linalg.norm(vj)
                if (di <= threebody_cutoff) and (dj <= threebody_cutoff):
                    cos.append(vi.dot(vj) / np.linalg.norm(vi) / np.linalg.norm(vj))
        start_index += n_site
    return cos


class TestCompute(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = Structure(Lattice.cubic(3.17), ["Mo", "Mo"], [[0.01, 0, 0], [0.5, 0.5, 0.5]])
        element_types = get_element_list([cls.s1])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, state = p2g.get_graph(cls.s1)
        cls.g1 = graph
        cls.state1 = state

        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        methane = Molecule(["C", "H", "H", "H", "H"], coords)
        element_types = get_element_list([methane])
        p2g = Molecule2Graph(element_types=element_types, cutoff=2.0)
        graph, state = p2g.get_graph(methane)
        cls.g2 = graph
        cls.state2 = state

    def test_compute_pair_vector(self):
        bv, bd = compute_pair_vector_and_distance(self.g1)
        self.g1.edata["bond_vec"] = bv
        d = torch.linalg.norm(self.g1.edata["bond_vec"], axis=1)

        _, _, _, d2 = self.s1.get_neighbor_list(r=5.0)

        np.testing.assert_array_almost_equal(np.sort(d), np.sort(d2))

    def test_compute_pair_vector_for_molecule(self):
        bv, bd = compute_pair_vector_and_distance(self.g2)
        self.g2.edata["bond_vec"] = bv
        d = torch.linalg.norm(self.g2.edata["bond_vec"], axis=1)

        d2 = np.array(
            [
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
            ]
        )

        np.testing.assert_array_almost_equal(np.sort(d), np.sort(d2))

    def test_compute_angle(self):
        bv, bd = compute_pair_vector_and_distance(self.g1)
        self.g1.edata["bond_vec"] = bv
        self.g1.edata["bond_dist"] = bd
        cos_loop = _calculate_cos_loop(self.g1, 4.0)

        line_graph = create_line_graph(self.g1, 4.0)
        line_graph.apply_edges(compute_theta_and_phi)
        np.testing.assert_array_almost_equal(
            np.sort(np.array(cos_loop)), np.sort(np.array(line_graph.edata["cos_theta"]))
        )

        bv, bd = compute_pair_vector_and_distance(self.g2)
        self.g2.edata["bond_vec"] = bv
        self.g2.edata["bond_dist"] = bd
        cos_loop = _calculate_cos_loop(self.g2, 2.0)

        line_graph = create_line_graph(self.g2, 2.0)
        line_graph.apply_edges(compute_theta_and_phi)
        np.testing.assert_array_almost_equal(
            np.sort(np.array(cos_loop)), np.sort(np.array(line_graph.edata["cos_theta"]))
        )


if __name__ == "__main__":
    unittest.main()
