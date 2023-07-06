"""Implementation of Interatomic Potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import grad

from matgl.layers import AtomRef
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    import dgl
    import numpy as np


class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 1

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | None = None,
        data_std: torch.Tensor | None = None,
        element_refs: np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.element_refs: AtomRef | None
        if element_refs is not None:
            self.element_refs = AtomRef(property_offset=element_refs)
        else:
            self.element_refs = None

        self.data_mean = data_mean if data_mean is not None else torch.zeros(1)
        self.data_std = data_std if data_std is not None else torch.ones(1)

    def forward(
        self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, l_g: dgl.DGLGraph | None = None
    ) -> tuple:
        """Args:
            g: DGL graph
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            energies, forces, stresses, hessian: torch.Tensor
        """
        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)
        total_energies = self.data_std * self.model(g=g, state_attr=state_attr, l_g=l_g) + self.data_mean
        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset

        if self.calc_forces:
            grads = grad(
                total_energies,
                [g.ndata["pos"], g.edata["bond_vec"]],
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )

            forces = -grads[0]
            if self.calc_hessian:
                r = -grads[0].view(-1)
                s = r.size(0)
                hessian = total_energies.new_zeros((s, s))
                for iatom in range(s):
                    tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s)[0]
                    if tmp is not None:
                        hessian[iatom] = tmp.view(-1)
        if self.calc_stresses:
            f_ij = -grads[1]
            sts: list = []
            count_edge = 0
            count_node = 0
            for graph_id in range(g.batch_size):
                num_edges = g.batch_num_edges()[graph_id]
                num_nodes = 0
                sts.append(
                    -1
                    * (
                        160.21766208
                        * torch.matmul(
                            g.edata["bond_vec"][count_edge : count_edge + num_edges].T,
                            f_ij[count_edge : count_edge + num_edges],
                        )
                        / g.ndata["volume"][count_node + num_nodes]
                    )
                )
                count_edge = count_edge + num_edges
                num_nodes = g.batch_num_nodes()[graph_id]
                count_node = count_node + num_nodes
            stresses = torch.cat(sts)
        return total_energies, forces, stresses, hessian
