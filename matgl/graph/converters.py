"""
Tools to convert materials representations from Pymatgen and other codes to DGLGraphs.
"""
from __future__ import annotations

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from ase import Atoms
from dgl.backend import tensor
from pymatgen.core import Element, Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres


def get_element_list(train_structures: list[Structure | Molecule]) -> tuple[str]:
    """Get the dictionary containing elements in the training set for atomic features

    Args:
        train_structures: pymatgen Molecule/Structure object

    Returns:
        Tuple of elements covered in training set
    """
    elements = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return tuple(sorted(elements, key=lambda el: Element(el).Z))  # type: ignore


class Pmg2Graph:
    """
    Construct a DGL graph from Pymatgen Molecules or Structures, ASE atoms is also added.
    """

    def __init__(
        self,
        element_types: tuple[str],
        cutoff: float = 5.0,
    ):
        """
        Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph_from_molecule(self, mol: Molecule) -> tuple[dgl.DGLGraph, list]:
        """
        Get a DGL graph from an input molecule.

        :param mol: pymatgen molecule object
        :return: (dgl graph, state features)
        """
        natoms = len(mol)
        R = mol.cart_coords
        element_types = self.element_types
        Z = np.array([np.eye(len(element_types))[element_types.index(site.specie.symbol)] for site in mol])
        np.array([site.specie.Z for site in mol])
        weight = mol.composition.weight / len(mol)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        dists = mol.distance_matrix.flatten()
        nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
        nbonds /= natoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
        adj = adj.tocoo()
        u, v = tensor(adj.row), tensor(adj.col)
        g = dgl.graph((u, v))
        g = dgl.to_bidirected(g)
        g.ndata["pos"] = tensor(R)
        g.ndata["attr"] = tensor(Z)
        g.ndata["node_type"] = tensor(np.hstack([[element_types.index(site.specie.symbol)] for site in mol]))
        g.edata["pbc_offset"] = torch.zeros(g.num_edges(), 3)
        g.edata["lattice"] = torch.zeros(g.num_edges(), 3, 3)
        state_attr = [weight, nbonds]
        g.edata["pbc_offshift"] = torch.zeros(g.num_edges(), 3)

        return g, state_attr

    def get_graph_from_structure(self, structure: Structure) -> tuple[dgl.DGLGraph, list]:
        """
        Get a DGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: DGL graph
            state_attr: state features
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        Z = np.array([np.eye(len(element_types))[element_types.index(site.specie.symbol)] for site in structure])
        atomic_number = np.array([site.specie.Z for site in structure])
        lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
        volume = structure.volume
        cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl.graph((u, v))
        g.edata["pbc_offset"] = torch.tensor(images)
        g.edata["lattice"] = tensor([[lattice_matrix] for i in range(g.num_edges())])
        g.ndata["attr"] = tensor(Z)
        g.ndata["node_type"] = tensor(np.hstack([[element_types.index(site.specie.symbol)] for site in structure]))
        g.ndata["pos"] = tensor(cart_coords)
        g.ndata["volume"] = tensor([volume for i in range(atomic_number.shape[0])])
        state_attr = [0.0, 0.0]
        g.edata["pbc_offshift"] = torch.matmul(tensor(images), tensor(lattice_matrix))
        return g, state_attr

    def get_graph_from_atoms(self, atoms: Atoms) -> tuple[dgl.DGLGraph, list]:
        """
        Get a DGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: DGL graph
            state_attr: state features
        """

        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        Z = np.array([np.eye(len(element_types))[element_types.index(i.symbol)] for i in atoms])
        atomic_number = np.array(atoms.get_atomic_numbers())
        lattice_matrix = np.ascontiguousarray(np.array(atoms.get_cell()), dtype=float)
        volume = atoms.get_volume()
        cart_coords = np.ascontiguousarray(np.array(atoms.get_positions()), dtype=float)
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl.graph((u, v))
        g.edata["pbc_offset"] = torch.tensor(images)
        g.edata["lattice"] = tensor([[lattice_matrix] for i in range(g.num_edges())])
        g.ndata["attr"] = tensor(Z)
        g.ndata["node_type"] = tensor(np.hstack([[element_types.index(i.symbol)] for i in atoms]))
        g.ndata["pos"] = tensor(cart_coords)
        g.ndata["volume"] = tensor([volume for i in range(atomic_number.shape[0])])
        state_attr = [0.0, 0.0]
        g.edata["pbc_offshift"] = torch.matmul(tensor(images), tensor(lattice_matrix))
        return g, state_attr
