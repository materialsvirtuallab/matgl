"""
M3GNet potentials
"""
from __future__ import annotations

import dgl
import torch
import torch.nn as nn
from torch.autograd import grad


class Potential(nn.Module):
    """
    A M3GNet potential class.
    """

    def __init__(
        self,
        model: nn.Module,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
    ):
        """
        :param model: M3GNet model
        :param calc_forces: Enable force calculations
        :param calc_stresses: Enable stress calculations
        :param calc_hessian: Enable hessian calculations
        """
        super().__init__()
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian

    def forward(
        self, g: dgl.DGLGraph, graph_attr: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Args:
        g: DGL graph
        graph_attr: State attrs

        Returns:
        energies, forces, stresses, hessian: torch.tensor
        """
        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        total_energies = self.model(g, graph_attr)

        if self.calc_forces:

            grads = grad(
                total_energies,
                g.ndata["pos"],
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )[0]

            forces = -grads

            if self.calc_hessian:
                r = -grads.view(-1)
                s = r.size(0)
                hessian = total_energies.new_zeros((s, s))
                for iatom in range(s):
                    tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=(iatom < s))[0]
                    if tmp is not None:
                        hessian[iatom] = tmp.view(-1)
        if self.calc_stresses:

            grads = grad(
                total_energies,
                g.edata["bond_vec"],
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            f_ij = -grads[0]
            stresses = -1 * (160.21766208 * torch.matmul(g.edata["bond_vec"].T, f_ij) / g.ndata["volume"][0])

        return total_energies, forces, stresses, hessian
