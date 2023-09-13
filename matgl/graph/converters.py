"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""
from __future__ import annotations

import abc

import dgl
import numpy as np
import torch

import matgl


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[dgl.DGLGraph, list]:
        """Args:
        structure: Input crystals or molecule.

        Returns:
        DGLGraph object, state_attr
        """

    @staticmethod
    def _index(element_types, site, allow_other_atoms):
        """Returns the index of the provided Pymatgen site in the list of
        element types. If allow_other_atoms is True, and the site is not found
        in the provided list, len(element_types) will be returned (this is an
        extra "catch all" index for any element not found in the list).
        """

        try:
            return element_types.index(site.specie.symbol)
        except ValueError:
            if allow_other_atoms:
                return len(element_types)
            raise ValueError(
                "allow_other_atoms is False and atom type "
                f"{site.specie.symbol} was not found in the element types "
                f"list {element_types}"
            )

    @staticmethod
    def _index_ase(element_to_index, elem, allow_other_atoms):
        """See _index above (this does the same thing, basically, except it
        deals with a dictionary instead of a list).
        """

        try:
            return element_to_index[elem]
        except KeyError:
            if allow_other_atoms:
                return len(element_to_index)
            raise KeyError(
                "allow_other_atoms is False and atom type "
                f"{elem} was not found in the element types "
                f"list {element_to_index}"
            )

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        element_types,
        cart_coords,
        is_atoms: bool = False,
        allow_other_atoms: bool = False,
    ) -> tuple[dgl.DGLGraph, list]:
        """Construct a dgl graph from processed structure and bond information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            cart_coords: Cartisian coordinates of all atoms in the structure.
            is_atoms: whether the input structure object is ASE atoms object or not.
            allow_other_atoms: if an atom found in a provided structure is not found
                in the element_types list, it will be provided the index len(element_types),
                a "catch all" bin for all other elements

        Returns:
            DGLGraph object, state_attr

        """
        u, v = torch.tensor(src_id), torch.tensor(dst_id)
        g = dgl.graph((u, v), num_nodes=len(structure))
        pbc_offset = torch.tensor(images, dtype=torch.float64)
        g.edata["pbc_offset"] = pbc_offset.to(matgl.int_th)
        # Note: pbc_ offshift and pos needs to be float64 to handle cases where bonds are exactly at cutoff
        g.edata["pbc_offshift"] = torch.matmul(pbc_offset, torch.tensor(lattice_matrix[0]))
        g.edata["lattice"] = torch.tensor(np.repeat(lattice_matrix, g.num_edges(), axis=0), dtype=matgl.float_th)

        # Not an ASE atoms object
        if not is_atoms:
            node_type = np.array([self._index(element_types, site, allow_other_atoms) for site in structure])

        # Is an ASE atoms object
        else:
            element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
            node_type = np.array(
                [
                    self._index_ase(element_to_index, elem, allow_other_atoms)
                    for elem in structure.get_chemical_symbols()
                ]
            )

        g.ndata["node_type"] = torch.tensor(node_type, dtype=matgl.int_th)
        g.ndata["pos"] = torch.tensor(cart_coords, dtype=torch.float64)
        state_attr = np.array([0.0, 0.0]).astype(matgl.float_np)
        return g, state_attr
