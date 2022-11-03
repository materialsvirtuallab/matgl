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


class RBFExpansion(nn.Module):
    r"""Expand distances between nodes by radial basis functions.

    .. math::
        \exp(- \gamma * ||d - \mu||^2)

    where :math:`d` is the distance between two nodes and :math:`\mu` helps centralizes
    the distances. We use multiple centers evenly distributed in the range of
    :math:`[\text{low}, \text{high}]` with the difference between two adjacent centers
    being :math:`gap`.

    The number of centers is decided by :math:`(\text{high} - \text{low}) / \text{gap}`.
    Choosing fewer centers corresponds to reducing the resolution of the filter.

    Parameters
    ----------
    low : float
        Smallest center. Default to 0.
    high : float
        Largest center. Default to 4.
    gap : float
        Difference between two adjacent centers. :math:`\gamma` will be computed as the
        reciprocal of gap. Default to 0.1.
    gamma : float
        Width of Gaussian function
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 4.0,
        gap: float = 0.2,
        gamma: Optional[float] = None,
    ):
        super(RBFExpansion, self).__init__()

        num_centers = int(np.ceil((high - low) / gap))
        self.centers = np.linspace(low, high, num_centers)
        self.centers = nn.Parameter(
            torch.tensor(self.centers).float(), requires_grad=False
        )
        if gamma is None:
            self.gamma = 1.0 / np.diff(self.centers).mean()
        else:
            self.gamma = gamma

    def reset_parameters(self):
        """Reinitialize model parameters."""
        device = self.centers.device
        self.centers = nn.Parameter(
            self.centers.clone().detach().float(), requires_grad=False
        ).to(device)

    def forward(self, edge_dists):
        """Expand distances.

        Parameters
        ----------
        edge_dists : float32 tensor of shape (E, 1)
            Distances between end nodes of edges, E for the number of edges.

        Returns
        -------
        float32 tensor of shape (E, len(self.centers))
            Expanded distances.
        """
        radial = edge_dists - self.centers
        coef = -self.gamma
        return torch.exp(coef * (radial**2))


class SimpleMolecularGraph:
    """
    Construct a DGL simple molecular graph with fix radius cutoff
    """

    def process(mols: Molecule, types: list):
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

    def convert(N, R, Z, N_cumsum, mol_weights, idx, rcut):
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
        dist_converter = RBFExpansion(gamma=0.5)
        for iatom in range(N[idx]):
            for jatom in range(iatom + 1, N[idx]):
                if dist[iatom][jatom] <= rcut:
                    number_of_bonds = number_of_bonds + 1
        number_of_bonds = number_of_bonds / N[idx]
        adj = sp.csr_matrix(dist <= rcut) - sp.eye(n_atoms, dtype=np.bool_)
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


# KK:for testing purpose
# coords = [
#    [0.000000, 0.000000, 0.000000],
#    [0.000000, 0.000000, 1.089000],
#    [1.026719, 0.000000, -0.363000],
#    [-0.513360, -0.889165, -0.363000],
#    [-0.513360, 0.889165, -0.363000],
# ]
# methane = Molecule(["C", "H", "H", "H", "H"], coords)
# Molecules = [methane, methane]
# mol_graph = SimpleMolecularGraph()
# a, b, c, d, e = SimpleMolecularGraph.process(Molecules, types={"H": 0, "C": 1})
# graph, state = SimpleMolecularGraph.convert(a, b, c, d, e, 1, 2.0)
# print(graph, state)
