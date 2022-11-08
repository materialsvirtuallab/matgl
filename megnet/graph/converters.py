from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from dgl.backend import tensor
from dgl.convert import graph as dgl_graph
from dgl.transforms import to_bidirected
from pymatgen.core import Element, Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres


def get_element_list(train_structures: list[Structure | Molecule]) -> list[str]:
    """Get the dictionary containing elements in the training set for atomic features
    Paramters
    ---------
    train_structures: pymatgen Molecule/Structure object

    Returns
    -------
        elem_dict: List of elements covered in training set
    """
    elements = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return list(sorted(elements, key=lambda el: Element(el).Z))


class GaussianExpansion(nn.Module):
    r"""
    Gaussian Radial Expansion.
    The bond distance is expanded to a vector of shape [m],
    where m is the number of Gaussian basis centers
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """
        Parameters
        ----------
        initial : float
                Location of initial Gaussian basis center.
        final : float
                Location of final Gaussian basis center
        number : int
                Number of Gaussian Basis functions
        width : float
                Width of Gaussian Basis functions
        """
        super().__init__()
        self.centers = np.linspace(initial, final, num_centers)
        self.centers = nn.Parameter(
            torch.tensor(self.centers).float(), requires_grad=False
        )
        if width is None:
            self.width = float(1.0 / np.diff(self.centers).mean())
        else:
            self.width = width

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False
        ).to(device)

    def forward(self, bond_dists):
        """Expand distances.

        Parameters
        ----------
        bond_dists :
            Bond (edge) distances between two atoms (nodes)

        Returns
        -------
        A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists - self.centers
        return torch.exp(-self.width * (diff**2))


class Molecule2Graph:
    """
    Construct a DGL molecular graph with fix radius cutoff
    """

    def __init__(
        self,
        element_types: list,
        cutoff: float = 5.0,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: float = 0.5,
    ):
        """
        Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        initial: Initial location of center for Gaussian expansion
        final: Final location of center for Gaussian expansion
        num_centers: Number of centers for Gaussian expansion
        width: Width of Gaussian function
        """
        self.element_types = element_types
        self.cutoff = cutoff
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width

    def get_graph(self, mol: Molecule):
        """
        Get a DGL graph from an input molecule.

        :param mol: pymatgen molecule object
        :return:
            g: dgl graph
            state_attr: state features
        """
        natoms = len(mol)
        R = mol.cart_coords
        element_types = self.element_types
        Z = [
            np.eye(len(element_types))[element_types.index(site.specie.symbol)]
            for site in mol
        ]
        Z = np.array(Z)
        weight = mol.composition.weight / len(mol)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        dists = mol.distance_matrix.flatten()
        nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
        nbonds /= natoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
        adj = adj.tocoo()
        u, v = tensor(adj.row), tensor(adj.col)
        edge_rbf_list = []
        g = dgl_graph((u, v))
        for i in range(g.num_edges()):
            rbf_dist = dist_converter(dist[u[i]][v[i]]).detach().numpy()
            edge_rbf_list += [rbf_dist]
        edge_rbf_list = np.array(edge_rbf_list).astype(np.float64)
        g = to_bidirected(g)
        g.edata["edge_attr"] = tensor(edge_rbf_list)
        g.ndata["attr"] = tensor(Z)
        state_attr = [weight, nbonds]
        return g, state_attr


class Crystal2Graph:
    """
    Convert a crystal structure into a DGL graph
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: float = 0.5,
    ):
        """
        Parameters:
        cutoff: Cutoff radius for graph representation
        initial: Initial location of center for Gaussian expansion
        final: Final location of center for Gaussian expansion
        num_centers: Number of centers for Gaussian expansion
        width: Width of Gaussian function
        """
        self.cutoff = cutoff
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width

    def process(self, cry: Structure, types: dict):
        """Process information from a pymatgen crystal.
        Parameters
        ----------
        cry: pymatgen structure object
        types: dictionary contains all elements appearing in the dataset

        Returns
        -------
        N: number of atoms in a crystal (int)
        src_id: central atom id (np.array)
        dst_id: neighbor atom id (np.array)
        bond_dist: bond distance between central and neighbor atoms (np.array)
        Z: atomic number in a crystal (np.array)
        """
        numerical_tol = 1.0e-8
        Z = []
        pbc = np.array([1, 1, 1], dtype=int)
        N = cry.num_sites
        lattice_matrix = np.ascontiguousarray(np.array(cry.lattice.matrix), dtype=float)
        cart_coords = np.ascontiguousarray(np.array(cry.cart_coords), dtype=float)
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        for atom_id in range(cry.num_sites):
            Z.append(np.eye(len(types))[types[cry.species[atom_id].symbol]])
        Z = np.array(Z)
        return (
            N,
            src_id[exclude_self],
            dst_id[exclude_self],
            bond_dist[exclude_self],
            Z,
        )

    def get_graph(self, N, src_id, dst_id, bond_dist, Z):
        """Convert a crystal into a DGL graph
        Parameters
        ----------
        N: number of atoms in a crystal (int)
        src_id: central atom id (np.array)
        dst_id: neighbor atom id (np.array)
        bond_dist: bond distance between central and neighbor atoms (np.array)
        Z: atomic number in a crystal (np.array)

        Returns
        -------
        g: dgl graph
        state_attr: state features
        """
        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        edge_rbf_list = []
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl_graph((u, v))
        for i in range(g.num_edges()):
            rbf_dist = dist_converter(bond_dist[i]).detach().numpy()
            edge_rbf_list += [rbf_dist]
        edge_rbf_list = np.array(edge_rbf_list).astype(np.float64)
        g.edata["edge_attr"] = tensor(edge_rbf_list)
        g.ndata["attr"] = tensor(Z)
        state_attr = [0.0, 0.0]

        return g, state_attr
