import os
import unittest

import numpy as np
from pymatgen.core import Molecule
from megnet.graph.converters import GaussianExpansion, Molecule2Graph

module_dir = os.path.dirname(os.path.abspath(__file__))


class GaussianExpansionTest(unittest.TestCase):
    def test_call(self):
        bond_dist = 1.0
        dist_converter = GaussianExpansion()
        expanded_dist = dist_converter(bond_dist)
        # check the shape of a vector
        self.assertTrue(np.allclose(expanded_dist.shape, [20]))
        # check the first value of expanded distance
        self.assertTrue(
            np.allclose(expanded_dist[0], np.exp(-0.5 * np.power(1.0 - 0.0, 2.0)))
        )
        # check the last value of expanded distance
        self.assertTrue(
            np.allclose(expanded_dist[-1], np.exp(-0.5 * np.power(1.0 - 4.0, 2.0)))
        )

class Molecule2GraphTest(unittest.TestCase):
    def test_process_convert(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        methane = Molecule(["C", "H", "H", "H", "H"], coords)
        mol_graph = Molecule2Graph(cutoff=1.5)
        a, b, c, d, e = mol_graph.process([methane], types={"H": 0, "C": 1})
        graph, state = mol_graph.convert(a, b, c, d, e, 0)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), 5))
        # check the number of edges
        self.assertTrue(np.allclose(graph.num_edges(), 8))
        # check the src_ids
        self.assertTrue(np.allclose(graph.edges()[0].numpy(), [0, 0, 0, 0, 1, 2, 3, 4]))
        # check the dst_ids
        self.assertTrue(np.allclose(graph.edges()[1].numpy(), [1, 2, 3, 4, 0, 0, 0, 0]))
        # check the shape of state features
        self.assertTrue(np.allclose(len(state), 2))
        # check the value of state features
        self.assertTrue(np.allclose(state, [3.208492, 0.8]))


if __name__ == "__main__":
    unittest.main()
