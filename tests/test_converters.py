import os
import unittest

import numpy as np
from pymatgen.core import Molecule, Structure

from megnet.graph.converters import (
    Crystal2Graph,
    GaussianExpansion,
    GetElementDictionary,
    Molecule2Graph,
)

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
        # a, b, c, d = mol_graph.process(methane, types={"H": 0, "C": 1})
        graph, state = mol_graph.get_graph(methane)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), 5))
        # check the number of edges
        self.assertTrue(np.allclose(graph.num_edges(), 8))
        # check the src_ids
        self.assertTrue(np.allclose(graph.edges()[0].numpy(), [0, 0, 0, 0, 1, 2, 3, 4]))
        # check the dst_ids
        self.assertTrue(np.allclose(graph.edges()[1].numpy(), [1, 2, 3, 4, 0, 0, 0, 0]))
        # check the atomic features of atom C
        self.assertTrue(np.allclose(graph.ndata["attr"][0], [1, 0]))
        # check the atomic features of atom H
        self.assertTrue(np.allclose(graph.ndata["attr"][1], [0, 1]))
        # check the edge features of atom 0 and atom 1
        dist_converter = GaussianExpansion()
        self.assertTrue(
            np.allclose(
                graph.edata["edge_attr"][0].numpy(), dist_converter(1.089).numpy()
            )
        )
        # check the shape of state features
        self.assertTrue(np.allclose(len(state), 2))
        # check the value of state features
        print(state)
        self.assertTrue(np.allclose(state, [3.208492, 0.8]))


class Crystal2GraphTest(unittest.TestCase):
    def test_process_convert(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        structure_LiFePO4 = Structure.from_file(
            os.path.join(module_dir, "cifs", "LiFePO4_mp-19017_computed.cif")
        )
        cry_graph = Crystal2Graph(cutoff=4.0)
        a, b, c, d, e = cry_graph.process(
            structure_LiFePO4, {"Li": 0, "O": 1, "P": 2, "Fe": 3}
        )
        graph, state = cry_graph.get_graph(a, b, c, d, e)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), structure_LiFePO4.num_sites))
        # check the atomic feature of atom 0
        self.assertTrue(np.allclose(graph.ndata["attr"][0].numpy(), [1, 0, 0, 0]))
        # check the atomic feature of atom 4
        self.assertTrue(np.allclose(graph.ndata["attr"][4].numpy(), [0, 0, 0, 1]))
        # check the number of bonds
        self.assertTrue(np.allclose(graph.num_edges(), 704))
        # check the edge features of bond between atom 0 and 1
        self.assertTrue(
            np.allclose(
                graph.edata["edge_attr"][0].numpy(),
                [
                    3.73113056e-04,
                    8.42360663e-04,
                    1.81931420e-03,
                    3.75896739e-03,
                    7.42986286e-03,
                    1.40489815e-02,
                    2.54132431e-02,
                    4.39771265e-02,
                    7.28023350e-02,
                    1.15296274e-01,
                    1.74677297e-01,
                    2.53168255e-01,
                    3.51021290e-01,
                    4.65596139e-01,
                    5.90794742e-01,
                    7.17158973e-01,
                    8.32809508e-01,
                    9.25182521e-01,
                    9.83242571e-01,
                    9.99644101e-01,
                ],
            )
        )
        # check the state features
        self.assertTrue(np.allclose(state, [0.0, 0.0]))
        structure_BaTiO3 = Structure.from_file(
            os.path.join(module_dir, "cifs", "BaTiO3_mp-2998_computed.cif")
        )
        cry_graph2 = Crystal2Graph(cutoff=5.0)
        a, b, c, d, e = cry_graph.process(structure_BaTiO3, {"O": 0, "Ti": 1, "Ba": 2})
        graph, state = cry_graph.get_graph(a, b, c, d, e)
        # check the number of nodes
        self.assertTrue(np.allclose(graph.num_nodes(), structure_BaTiO3.num_sites))
        # check the atomic features of atom 0
        self.assertTrue(np.allclose(graph.ndata["attr"][0], [0, 0, 1]))
        # check the atomic features of atom 1
        self.assertTrue(np.allclose(graph.ndata["attr"][1], [0, 1, 0]))
        # check the number of edges
        self.assertTrue(np.allclose(graph.num_edges(), 76))
        # check the edge features of bond between atom 0 and 3
        self.assertTrue(
            np.allclose(
                graph.edata["edge_attr"][0].numpy(),
                [
                    0.01702178,
                    0.03036285,
                    0.05181216,
                    0.08458085,
                    0.13208824,
                    0.19733654,
                    0.28203458,
                    0.38561046,
                    0.50436711,
                    0.63109708,
                    0.75543493,
                    0.86506635,
                    0.94766164,
                    0.99313593,
                    0.99567026,
                    0.95493513,
                    0.87616062,
                    0.76903325,
                    0.64574027,
                    0.51870716,
                ],
            )
        )
        # check the state features
        self.assertTrue(np.allclose(state, [0.0, 0.0]))

    def test_GetElementDictionary(self):
        structure_LiFePO4 = Structure.from_file(
            os.path.join(module_dir, "cifs", "LiFePO4_mp-19017_computed.cif")
        )
        structure_BaTiO3 = Structure.from_file(
            os.path.join(module_dir, "cifs", "BaTiO3_mp-2998_computed.cif")
        )
        structure = [structure_LiFePO4, structure_BaTiO3]
        elem_list = GetElementDictionary(structure)
        self.assertDictEqual(
            elem_list, {"Li": 0, "O": 1, "P": 2, "Ti": 3, "Fe": 4, "Ba": 5}
        )


if __name__ == "__main__":
    unittest.main()
