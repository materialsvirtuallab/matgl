from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    compute_theta_and_phi,
    create_line_graph,
    prune_edges_by_features,
    separate_node_edge_keys,
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
    Returns: a list of cosine theta values.
    """
    _, _, n_sites = torch.unique(graph.edge_index[0], return_inverse=True, return_counts=True)
    start_index = 0
    cos = []
    for n_site in n_sites:
        for i in range(n_site):
            for j in range(n_site):
                if i == j:
                    continue
                vi = graph.bond_vec[i + start_index].detach().numpy()
                vj = graph.bond_vec[j + start_index].detach().numpy()
                di = np.linalg.norm(vi)
                dj = np.linalg.norm(vj)
                if (di <= threebody_cutoff) and (dj <= threebody_cutoff):
                    cos.append(vi.dot(vj) / np.linalg.norm(vi) / np.linalg.norm(vj))
        start_index += n_site
    return cos


class TestCompute:
    def test_compute_pair_vector(self, graph_Mo):
        s1, g1, state1 = graph_Mo
        lattice = torch.tensor(s1.lattice.matrix, dtype=matgl.float_th).unsqueeze(dim=0)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lattice[0])
        g1.pos = g1.frac_coords @ lattice[0]
        bv, bd = compute_pair_vector_and_distance(g1)
        g1.bond_vec = bv
        d = torch.linalg.norm(g1.bond_vec, axis=1)

        _, _, _, d2 = s1.get_neighbor_list(r=5.0)

        np.testing.assert_array_almost_equal(np.sort(d), np.sort(d2))

    def test_compute_pair_vector_for_molecule(self, graph_CH4):
        s2, g2, state2 = graph_CH4
        lattice = torch.tensor(np.identity(3), dtype=matgl.float_th).unsqueeze(dim=0)
        g2.pbc_offshift = torch.matmul(g2.pbc_offset, lattice[0])
        g2.pos = g2.frac_coords @ lattice[0]
        bv, bd = compute_pair_vector_and_distance(g2)
        g2.bond_vec = bv
        d = torch.linalg.norm(g2.bond_vec, axis=1)

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

    def test_compute_angle(self, graph_Mo, graph_CH4):
        s1, g1, state1 = graph_Mo
        lattice = torch.tensor(s1.lattice.matrix, dtype=matgl.float_th).unsqueeze(dim=0)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lattice[0])
        g1.pos = g1.frac_coords @ lattice[0]
        bv, bd = compute_pair_vector_and_distance(g1)
        g1.bond_vec = bv
        g1.bond_dist = bd
        cos_loop = _calculate_cos_loop(g1, 4.0)

        line_graph = create_line_graph(g1, 4.0)
        line_graph = compute_theta_and_phi(line_graph)
        np.testing.assert_array_almost_equal(np.sort(np.array(cos_loop)), np.sort(np.array(line_graph.cos_theta)))

        # test only compute theta
        line_graph, key = compute_theta(line_graph, cosine=False, directed=False)
        theta = np.arccos(np.clip(cos_loop, -1.0 + 1e-7, 1.0 - 1e-7))
        np.testing.assert_array_almost_equal(np.sort(theta), np.sort(np.array(line_graph.theta)), decimal=4)

        # test only compute theta with cosine
        line_graph, key = compute_theta(line_graph, cosine=True, directed=False)
        np.testing.assert_array_almost_equal(np.sort(np.array(cos_loop)), np.sort(np.array(line_graph.cos_theta)))

        s2, g2, state2 = graph_CH4
        lattice = torch.tensor(np.identity(3), dtype=matgl.float_th).unsqueeze(dim=0)
        g2.pbc_offshift = torch.matmul(g2.pbc_offset, lattice[0])
        g2.pos = g2.frac_coords @ lattice[0]
        bv, bd = compute_pair_vector_and_distance(g2)
        g2.bond_vec = bv
        g2.bond_dist = bd
        cos_loop = _calculate_cos_loop(g2, 2.0)

        line_graph = create_line_graph(g2, 2.0)
        line_graph = compute_theta_and_phi(line_graph)
        np.testing.assert_array_almost_equal(np.sort(np.array(cos_loop)), np.sort(np.array(line_graph.cos_theta)))

        # test only compute theta
        line_graph, key = compute_theta(line_graph, directed=False)
        np.testing.assert_array_almost_equal(
            np.sort(np.arccos(np.array(cos_loop))), np.sort(np.array(line_graph.theta))
        )

        # test only compute theta with cosine
        line_graph, key = compute_theta(line_graph, cosine=True, directed=False)
        np.testing.assert_array_almost_equal(np.sort(np.array(cos_loop)), np.sort(np.array(line_graph.cos_theta)))

    def test_compute_three_body(self, graph_AcAla3NHMe):
        mol1, g1, _ = graph_AcAla3NHMe
        lattice = torch.tensor(np.identity(3), dtype=matgl.float_th).unsqueeze(dim=0)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lattice[0])
        g1.pos = g1.frac_coords @ lattice[0]
        bv, bd = compute_pair_vector_and_distance(g1)
        g1.bond_vec = bv
        g1.bond_dist = bd
        line_graph = create_line_graph(g1, 5.0)
        line_graph = compute_theta_and_phi(line_graph)
        np.testing.assert_allclose(line_graph.triple_bond_lengths.detach().numpy()[0], 1.777829)


def test_line_graph_extensive():
    structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(6.0 / np.sqrt(2)), ["Fe"], [[0, 0, 0]])

    element_types = get_element_list([structure])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    g1, lat1, _ = converter.get_graph(structure)
    g1.pbc_offshift = torch.matmul(g1.pbc_offset, lat1[0])
    g1.pos = g1.frac_coords @ lat1[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(g1)
    g1.bond_dist = bond_dist
    g1.bond_vec = bond_vec

    supercell = structure.copy()
    supercell.make_supercell([2, 1, 1])
    g2, lat2, _ = converter.get_graph(supercell)
    g2.pbc_offshift = torch.matmul(g2.pbc_offset, lat2[0])
    g2.pos = g2.frac_coords @ lat2[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(g2)
    g2.bond_dist = bond_dist
    g2.bond_vec = bond_vec

    lg1 = create_line_graph(g1, 3.0)
    lg2 = create_line_graph(g2, 3.0)

    assert 2 * g1.num_nodes == g2.num_nodes
    assert 2 * g1.num_edges == g2.num_edges
    assert 2 * lg1.num_nodes == lg2.num_nodes
    assert 2 * lg1.num_edges == lg2.num_edges


@pytest.mark.parametrize("keep_ndata", [True, False])
@pytest.mark.parametrize("keep_edata", [True, False])
def test_remove_edges_by_features(graph_Mo, keep_ndata, keep_edata):
    s1, g1, state1 = graph_Mo
    lattice = torch.tensor(s1.lattice.matrix, dtype=matgl.float_th).unsqueeze(dim=0)
    g1.pbc_offshift = torch.matmul(g1.pbc_offset, lattice[0])
    g1.pos = g1.frac_coords @ lattice[0]
    bv, bd = compute_pair_vector_and_distance(g1)
    g1.bond_vec = bv
    g1.bond_dist = bd

    new_cutoff = 3.0
    converter = Structure2Graph(element_types=get_element_list([s1]), cutoff=new_cutoff)
    g2, lat2, state2 = converter.get_graph(s1)
    g2.pbc_offshift = torch.matmul(g2.pbc_offset, lat2[0])
    g2.pos = g2.frac_coords @ lat2[0]
    # remove edges by features
    new_g, node_keys, edge_keys = prune_edges_by_features(
        g1,
        "bond_dist",
        condition=lambda x: x > new_cutoff,
        keep_ndata=keep_ndata,
        keep_edata=keep_edata,
        return_keys=True,
    )
    valid_edges = g1.bond_dist <= new_cutoff
    g1_node_keys, g1_edge_keys, _ = separate_node_edge_keys(g1)
    assert new_g.num_edges == g2.num_edges
    assert new_g.num_nodes == g2.num_nodes
    assert torch.allclose(new_g.edge_ids, valid_edges.nonzero().squeeze())

    if keep_ndata:
        assert sorted(node_keys) == sorted(g1_node_keys)

    if keep_edata:
        for key in edge_keys:
            if key != "edge_ids":
                assert torch.allclose(new_g[key], g1[key][valid_edges])


@pytest.mark.parametrize("cutoff", [2.0, 3.0, 4.0])
@pytest.mark.parametrize("graph_data", ["graph_Mo", "graph_CH4", "graph_MoS", "graph_LiFePO4", "graph_MoSH"])
def test_directed_line_graph(graph_data, cutoff, request):
    s1, g1, state1 = request.getfixturevalue(graph_data)
    lattice = (
        torch.tensor(s1.lattice.matrix, dtype=matgl.float_th).unsqueeze(dim=0)
        if graph_data != "graph_CH4"
        else torch.tensor(np.identity(3), dtype=matgl.float_th).unsqueeze(dim=0)
    )
    g1.edata["pbc_offshift"] = torch.matmul(g1.edata["pbc_offset"], lattice[0])
    g1.ndata["pos"] = g1.ndata["frac_coords"] @ lattice[0]
    bv, bd = compute_pair_vector_and_distance(g1)
    g1.edata["bond_vec"] = bv
    g1.edata["bond_dist"] = bd
    cos_loop = _calculate_cos_loop(g1, cutoff)
    theta_loop = np.arccos(np.clip(cos_loop, -1.0 + 1e-7, 1.0 - 1e-7))

    line_graph = create_line_graph(g1, cutoff, directed=True)
    line_graph.apply_edges(compute_theta)

    # this test might be lax with just 4 decimal places
    np.testing.assert_array_almost_equal(np.sort(theta_loop), np.sort(np.array(line_graph.edata["theta"])), decimal=4)


@pytest.mark.parametrize("graph_data", ["graph_Mo", "graph_CH4", "graph_LiFePO4", "graph_MoSH"])
def test_ensure_directed_line_graph_compat(graph_data, request):
    s, g, state = request.getfixturevalue(graph_data)
    lattice = (
        torch.tensor(s.lattice.matrix, dtype=matgl.float_th).unsqueeze(dim=0)
        if graph_data != "graph_CH4"
        else torch.tensor(np.identity(3), dtype=matgl.float_th).unsqueeze(dim=0)
    )
    g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lattice[0])
    g.ndata["pos"] = g.ndata["frac_coords"] @ lattice[0]
    bv, bd = compute_pair_vector_and_distance(g)
    g.edata["bond_vec"] = bv
    g.edata["bond_dist"] = bd
    line_graph = create_line_graph(g, 3.0, directed=True)
    edge_ids = line_graph.ndata["edge_ids"].clone()
    src_bond_sign = line_graph.ndata["src_bond_sign"].clone()
    line_graph.ndata["edge_ids"] = torch.zeros(line_graph.num_nodes(), dtype=torch.long)
    line_graph.ndata["src_bond_sign"] = torch.zeros(line_graph.num_nodes())

    assert not torch.allclose(line_graph.ndata["edge_ids"], edge_ids)
    assert not torch.allclose(line_graph.ndata["src_bond_sign"], src_bond_sign)

    # # test that the line graph is not compatible
    # line_graph = ensure_line_graph_compatibility(g, line_graph, 3.0, directed=True)
    # tt.assert_allclose(line_graph.ndata["edge_ids"], edge_ids)
    # tt.assert_allclose(line_graph.ndata["src_bond_sign"], src_bond_sign)
    #
    # with pytest.raises(RuntimeError):
    #     ensure_line_graph_compatibility(g, line_graph, 1.0, directed=True)
