"""
Compute three-body indices
"""
from __future__ import annotations

import dgl
import numpy as np
import torch


def compute_3body(bond_atom_indices: np.array, n_atoms: np.array):
    """
    Calculate the three body indices from pair atom indices

    Args:
        bond_atom_indices (np.ndarray): pair atom indices
        n_atoms (list): number of atoms in each structure.

    Returns:
        triple_bond_indices (np.ndarray): bond indices that form three-body
        n_triple_ij (np.ndarray): number of three-body angles for each bond
        n_triple_i (np.ndarray): number of three-body angles each atom
        n_triple_s (np.ndarray): number of three-body angles for each structure
    """
    n_atoms_total = np.sum(n_atoms)
    first_col = bond_atom_indices[:, 0].reshape(-1, 1)
    all_indices = np.arange(n_atoms_total).reshape(1, -1)
    n_bond_per_atom = np.count_nonzero(first_col == all_indices, axis=0)
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
    n_triple = np.sum(n_triple_i)
    n_triple_ij = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)
    triple_bond_indices = np.empty(shape=(n_triple, 2), dtype=np.int32)

    start = 0
    cs = 0
    for i, n in enumerate(n_bond_per_atom):
        if n > 0:
            """
            triple_bond_indices is generated from all pair permutations of atom indices. The
            numpy version below does this with much greater efficiency. The equivalent slow
            code is:

            ```
            for j, k in itertools.permutations(range(n), 2):
                triple_bond_indices[index] = [start + j, start + k]
            ```
            """
            r = np.arange(n)
            x, y = np.meshgrid(r, r)
            c = np.stack([y.ravel(), x.ravel()], axis=1)
            final = c[c[:, 0] != c[:, 1]]
            triple_bond_indices[start : start + (n * (n - 1)), :] = final + cs
            start += n * (n - 1)
            cs += n

    n_triple_s = []
    i = 0
    for n in n_atoms:
        j = i + n
        n_triple_s.append(np.sum(n_triple_i[i:j]))
        i = j

    return (
        triple_bond_indices,
        n_triple_ij,
        n_triple_i,
        np.array(n_triple_s, dtype=np.int32),
    )


class ThreeBodyInteraction(dgl.Module):
    """
    Include 3-body interactions in the bond update
    """

    def __init__(self, update_network_atom: dgl.Module, update_network_bond: dgl.Module, **kwargs):
        r"""
        Args:
            update_network_atom (dgl.Module): Module for updating the atom attributes before merging with 3-body
                interactions.
            update_network_bond (dgl.Module): Module for updating the bond information after merging with 3-body
                interactions
            **kwargs: Passthrough to dgl.Module superclass.
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(self, graph: dgl.DGLGraph, three_basis: torch.Tensor, three_cutoff: float, **kwargs) -> dgl.DGLGraph:
        """

        Args:
            graph (dgl.DGLGraph): Input graph
            three_basis (tf.Tensor): three body basis expansion
            three_cutoff (float): cutoff radius
            **kwargs:
        Returns: Updated dgl.DGLGraph
        """
        # graph = graph[:]
        # end_atom_index = tf.gather(graph[Index.BOND_ATOM_INDICES][:, 1], graph[Index.TRIPLE_BOND_INDICES][:, 1])
        # atoms = self.update_network(graph[Index.ATOMS])
        # atoms = tf.gather(atoms, end_atom_index)
        # basis = three_basis * atoms
        # n_bonds = tf.reduce_sum(graph[Index.N_BONDS])
        # weights = tf.reshape(tf.gather(three_cutoff, graph[Index.TRIPLE_BOND_INDICES]), (-1, 2))
        # weights = tf.math.reduce_prod(weights, axis=-1)
        # basis = basis * weights[:, None]
        # new_bonds = tf.math.unsorted_segment_sum(
        #     basis,
        #     get_segment_indices_from_n(graph[Index.N_TRIPLE_IJ]),
        #     num_segments=n_bonds,
        # )
        # graph[Index.BONDS] = graph[Index.BONDS] + self.update_network2(new_bonds)
        return graph
