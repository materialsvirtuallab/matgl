from __future__ import annotations

import os

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from matgl.ext._pymatgen_pyg import Structure2GraphPYG, get_element_list

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestPmg2Graph:
    def test_get_graph_from_molecule(self, graph_CH4_pyg):
        mol, graph, state = graph_CH4_pyg
        # check the number of nodes
        assert np.allclose(graph.num_nodes, 5)
        # check the number of edges
        assert np.allclose(graph.num_edges, 20)
        # check the src_ids
        assert np.allclose(graph.edge_index[0].numpy(), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        # check the dst_ids
        assert np.allclose(graph.edge_index[1].numpy(), [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3])
        # check the atomic features of atom C
        assert np.allclose(graph.node_type.detach().numpy()[0], 1)
        # check the atomic features of atom H
        assert np.allclose(graph.node_type.detach().numpy()[1], 0)
        # check the shape of state features
        assert np.allclose(len(state), 2)
        # check the value of state features
        assert np.allclose(state, [3.208492, 2])
        # check the position of atom 0
        assert np.allclose(graph.pos[0], [0.0, 0.0, 0.0])

    def test_get_graph_from_structure(self, graph_LiFePO4_pyg):
        lfp, graph, state = graph_LiFePO4_pyg
        # check the number of nodes
        assert np.allclose(graph.num_nodes, lfp.num_sites)
        # check the atomic feature of atom 0
        assert np.allclose(graph.node_type.detach().numpy()[0], 0)
        # check the atomic feature of atom 4
        assert np.allclose(graph.node_type.detach().numpy()[4], 3)
        # check the number of bonds
        assert np.allclose(graph.num_edges, 704)
        # check the state features
        assert np.allclose(state, [0.0, 0.0])
        structure_BaTiO3 = Structure.from_prototype("perovskite", ["Ba", "Ti", "O"], a=4.04)
        element_types = get_element_list([structure_BaTiO3])
        p2g = Structure2GraphPYG(element_types=element_types, cutoff=4.0)
        graph, lattice, state = p2g.get_graph(structure_BaTiO3)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
        graph.pos = graph.frac_coords @ lattice[0]
        # check the number of nodes
        assert np.allclose(graph.num_nodes, structure_BaTiO3.num_sites)
        # check the atomic features of atom 0
        assert np.allclose(graph.node_type.detach().numpy()[0], 2)
        # check the atomic features of atom 1
        assert np.allclose(graph.node_type.detach().numpy()[1], 1)
        # check the number of edges
        assert np.allclose(graph.num_edges, 76)
        # check the state features
        assert np.allclose(state, [0.0, 0.0])
        # check the position of atom 0
        assert np.allclose(graph.pos[0], [0.0, 0.0, 0.0])
        # check the pbc offset from node 0 to image atom 6
        assert np.allclose(graph.pbc_offset[0], [-1, -1, -1])
        # check the lattice vector
        assert np.allclose(lattice[0].numpy(), [[4.04, 0.0, 0.0], [0.0, 4.04, 0.0], [0.0, 0.0, 4.04]])
        # check the volume
        assert np.allclose(torch.det(lattice).numpy(), [65.939264])

    def test_get_element_list(self):
        cscl = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        naf = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Na", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        elem_list = get_element_list([cscl, naf])
        assert elem_list == ("F", "Na", "Cl", "Cs")
