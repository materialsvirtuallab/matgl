"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import torch
from dgl.backend import tensor
from pymatgen.core import Molecule, Structure
import dgl


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    # def __init__(
    #     self,
    #     element_types: tuple[str, ...],
    #     cutoff: float = 5.0,
    # ):

    @abc.abstractmethod
    def get_graph(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        Z,
        element_types,
        cart_coords,
        volume=None,
    ) -> tuple[dgl.DGLGraph, list]:
        """Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            Z: Atomic number information of all atoms in the structure.
            element_types: Element symbols of all atoms in the structure.
            cart_coords: Cartisian coordinates of all atoms in the structure.
            volume: Volume of the structure.
        Returns:
            DGLGraph object, state_attr
        """
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl.graph((u, v))
        isolated_atoms = list(set(range(len(structure))).difference(src_id))
        if isolated_atoms:
            g.add_nodes(len(isolated_atoms))
        # if not isolated_atoms:
            # u, v = tensor(src_id), tensor(dst_id)
        # else:
        #     u, v = tensor(np.concatenate([src_id, isolated_atoms])), tensor(
        #         np.concatenate([dst_id, isolated_atoms])
        #     )
            # images = np.concatenate(
            #     [images, np.repeat([[1.0, 0.0, 0.0]], len(isolated_atoms), axis=0)]
            # )
        # g = dgl.graph((u, v))
        g.edata["pbc_offset"] = torch.tensor(images)
        g.edata["lattice"] = tensor(np.repeat(lattice_matrix, g.num_edges(), axis=0))
        g.ndata["attr"] = tensor(Z)
        g.ndata["node_type"] = tensor(
            np.hstack([[element_types.index(site.specie.symbol)] for site in structure])
        )
        g.ndata["pos"] = tensor(cart_coords)
        g.ndata["volume"] = tensor([volume] * g.num_nodes())
        state_attr = [0.0, 0.0]
        g.edata["pbc_offshift"] = torch.matmul(tensor(images), tensor(lattice_matrix[0]))
        return g, state_attr
