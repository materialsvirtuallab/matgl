import numpy as np
from pymatgen.core import Lattice, Structure, Molecule

import scipy.sparse as sp
import dgl

import torch
import torch.nn as nn
from typing import Optional
from dgl.convert import graph as dgl_graph
from dgl import backend as F
from dgl.transforms import to_bidirected


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
        super(GaussianExpansion, self).__init__()
        self.centers = np.linspace(initial, final, num_centers)
        self.centers = nn.Parameter(
            torch.tensor(self.centers).float(), requires_grad=False
        )
        if width is None:
            self.width = 1.0 / np.diff(self.centers).mean()
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
        width: Optional[float] = None,
    ):
        self.cutoff = cutoff
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.width = width

    def process(self, mols: Molecule, types: list):
        """Process information from a set of pymatgen molecules.
        Parameters
        ----------
        mols: pymatgen molecule object
        types: dictionary contains all elements appearing in the dataset

        Returns
        -------
        N: number of atoms in a molecule (np.array)
        R: cartesian coordinate (np.array)
        Z: atomic number (np.array)
        """
        N = []
        R = []
        Z = []
        mol_weights = []
        weights = 0.0
        for mol in mols:
            R.append(mol.cart_coords)
            N.append(mol.num_sites)
            for atom_id in range(mol.num_sites):
                Z.append(np.eye(len(types))[types[mol.species[atom_id].symbol]])
                weights = weights + mol.species[atom_id].atomic_mass
            mol_weights.append(weights / mol.num_sites)
            weights = 0.0
        N = np.array(N)
        cart_array = np.array(R)
        R = cart_array.reshape(-1, 3)
        Z = np.array(Z)
        N_cumsum = np.concatenate([[0], np.cumsum(N)])
        return N, R, Z, N_cumsum, mol_weights

    def convert(self, N, R, Z, N_cumsum, mol_weights, idx):
        """Convert a set of molecules into a set of DGL graphs
        Parameters
        ----------
        N: number of atoms in a molecule (np.array)
        R: cartesian coordinate in a molecule (np.array)
        Z: atomic number in a molecule (np.array)
        Z_cumsum: atom id (np.array)
        mol_weights: molcular weight per atom (np.array)
        idx: molecular id (np.array)
        rcut: cutoff radius

        Returns
        -------
        g: dgl graph
        state_attr: state features
        """
        n_atoms = N[idx]
        R = R[N_cumsum[idx] : N_cumsum[idx + 1]]
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        number_of_bonds = 0
        dist_converter = GaussianExpansion(
            initial=self.initial,
            final=self.final,
            num_centers=self.num_centers,
            width=self.width,
        )
        for iatom in range(N[idx]):
            for jatom in range(iatom + 1, N[idx]):
                if dist[iatom][jatom] <= self.cutoff:
                    number_of_bonds = number_of_bonds + 1
        number_of_bonds = number_of_bonds / N[idx]
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
        g.ndata["attr"] = F.tensor(Z[N_cumsum[idx] : N_cumsum[idx + 1]])
        state_attr = [mol_weights[idx], number_of_bonds]
        return g, state_attr
