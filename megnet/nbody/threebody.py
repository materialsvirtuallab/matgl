"""
Compute three-body indices
"""
from __future__ import annotations

import numpy as np


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
