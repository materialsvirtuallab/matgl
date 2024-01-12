---
layout: default
title: matgl.apps.md
nav_exclude: true
---

# matgl.apps package

This package implements specific applications of matgl models. An example is their use for fitting interatomic
potentials parameterizing the potential energy surface (PES).

## matgl.apps.pes module

Implementation of Interatomic Potentials.

### *class* matgl.apps.pes.Potential(model: nn.Module, data_mean: torch.Tensor | None = None, data_std: torch.Tensor | None = None, element_refs: np.ndarray | None = None, calc_forces: bool = True, calc_stresses: bool = True, calc_hessian: bool = False, calc_site_wise: bool = False)

Bases: `Module`, `IOMixIn`

A class representing an interatomic potential.

Initialize Potential from a model and elemental references.

* **Parameters:**
  * **model** – Model for predicting energies.
  * **data_mean** – Mean of target.
  * **data_std** – Std dev of target.
  * **element_refs** – Element reference values for each element.
  * **calc_forces** – Enable force calculations.
  * **calc_stresses** – Enable stress calculations.
  * **calc_hessian** – Enable hessian calculations.
  * **calc_site_wise** – Enable site-wise property calculation.

#### forward(g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, l_g: dgl.DGLGraph | None = None)

* **Parameters:**
  * **g** – DGL graph
  * **state_attr** – State attrs
  * **l_g** – Line graph.
* **Returns:**
  (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)