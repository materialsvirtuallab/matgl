from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import dgl
from dgl import backend as F
from dgl.convert import graph as dgl_graph
from dgl.transforms import to_bidirected
from pymatgen.core import Element, Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres


def GetElementDictionary(train_structures):
    """Get the dictionary containing elements in the training set for atomic features
    Paramters
    ---------
    train_structures: pymatgen Molecule/Structure object

    Returns
    -------
    elem_dict: dictionary including elements covered in training set
    """
    element_list = []
    for structure in train_structures:
        for atom_id in range(structure.num_sites):
            element_list.append(structure.species[atom_id].symbol)
    element_list = list(set(element_list))
    atomic_number_list = []
    for element in element_list:
        atomic_number_list.append(Element(element).Z - 1)
    atomic_number_list = np.argsort(atomic_number_list)
    sorted_list = []
    for sort_id in atomic_number_list:
        sorted_list.append(element_list[sort_id])
    elem_dict = {}
    count = 0
    for elem_id in sorted_list:
        elem_dict[elem_id] = count
        count = count + 1
    return elem_dict


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
        width: float = 0.5,
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
        cutoff: float = 5.0,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: float = 0.5,
    ):
        """
        Parameters
        ----------
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

    def process(self, mol: list[Molecule], types: dict):
        """Process information from a pymatgen molecule.
        Parameters
        ----------
        mol: pymatgen molecule object
        types: dictionary contains all elements appearing in the dataset

        Returns
        -------
        N: number of atoms in a molecule (np.array)
        R: cartesian coordinate (np.array)
        Z: atomic number (np.array)
        """
        R = []
        Z = []
        weights = 0.0
        R.append(mol.cart_coords)
        N = mol.num_sites
        for atom_id in range(mol.num_sites):
            Z.append(np.eye(len(types))[types[mol.species[atom_id].symbol]])
            weights = weights + mol.species[atom_id].atomic_mass
        mol_weights = weights / mol.num_sites
        weights = 0.0
        cart_array = np.array(R)
        R = cart_array.reshape(-1, 3)
        Z = np.array(Z)
        return N, R, Z, mol_weights

    def get_graph(self, N, R, Z, mol_weights):
        """Convert a molecule into a DGL graph
        Parameters
        ----------
        N: number of atoms in a molecule (int)
        R: cartesian coordinate in a molecule (np.array)
        Z: atomic number in a molecule (np.array)
        mol_weights: molcular weight per atom (np.array)

        Returns
        -------
        g: dgl graph
        state_attr: state features
        """
        n_atoms = N
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        number_of_bonds = 0
        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        for iatom in range(n_atoms):
            for jatom in range(iatom + 1, n_atoms):
                if dist[iatom][jatom] <= self.cutoff:
                    number_of_bonds = number_of_bonds + 1
        number_of_bonds = number_of_bonds / n_atoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(n_atoms, dtype=np.bool_)
        adj = adj.tocoo()
        u, v = F.tensor(adj.row), F.tensor(adj.col)
        edge_rbf_list = []
        g = dgl_graph((u, v))
        for i in range(g.num_edges()):
            rbf_dist = dist_converter(dist[u[i]][v[i]]).detach().numpy()
            edge_rbf_list += [rbf_dist]
        edge_rbf_list = np.array(edge_rbf_list).astype(np.float64)
        g = to_bidirected(g)
        g.edata["edge_attr"] = F.tensor(edge_rbf_list)
        g.ndata["attr"] = F.tensor(Z)
        state_attr = [mol_weights, number_of_bonds]
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

    def process(self, cry: list[Structure], types: dict):
        """Process information from a pymatgen crystal.
        Parameters
        ----------
        cry: pymatgen structure object
        types: dictionary contains all elements appearing in the dataset

        Returns
        -------
        N: number of atoms in a molecule (np.array)
        src_id: central atom id (list)
        dst_id: neighbor atom id (list)
        bond_dist: bond distance between central and neighbor atoms (list)
        Z: atomic number in a crystal (np.array)
        """
        numerical_tol = 1.0e-8
        Z = []
        bond_dist = []
        pbc = np.array([1, 1, 1], dtype=int)
        N = cry.num_sites
        lattice_matrix = np.ascontiguousarray(np.array(cry.lattice.matrix), dtype=float)
        cart_coords = np.ascontiguousarray(np.array(cry.cart_coords), dtype=float)
        center_indices, neighbor_indices, images, distances = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (center_indices != neighbor_indices) | (
            distances > numerical_tol
        )
        for atom_id in range(cry.num_sites):
            Z.append(np.eye(len(types))[types[cry.species[atom_id].symbol]])
        Z = np.array(Z)
        return (
            N,
            center_indices[exclude_self],
            neighbor_indices[exclude_self],
            distances[exclude_self],
            Z,
        )

    def get_graph(self, N, src_id, dst_id, bond_dist, Z):
        """Convert a crystal into a DGL graph
        Parameters
        ----------
        N: number of atoms in a crystal (int)
        src_id: central atom id (list)
        dst_id: neighbor atom id (list)
        bond_dist: bond distance between central and neighbor atoms (list)
        Z: atomic number in a crystal (np.array)

        Returns
        -------
        g: dgl graph
        state_attr: state features
        """
        n_atoms = N
        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        edge_rbf_list = []
        u, v = F.tensor(src_id), F.tensor(dst_id)
        g = dgl_graph((u, v))
        for i in range(g.num_edges()):
            rbf_dist = dist_converter(bond_dist[i]).detach().numpy()
            edge_rbf_list += [rbf_dist]
        edge_rbf_list = np.array(edge_rbf_list).astype(np.float64)
        g.edata["edge_attr"] = F.tensor(edge_rbf_list)
        g.ndata["attr"] = F.tensor(Z)
        state_attr = [0.0, 0.0]

        return g, state_attr
