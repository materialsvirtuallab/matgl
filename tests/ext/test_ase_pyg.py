from __future__ import annotations

import os.path

import numpy as np
import pytest

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
import torch
from ase.build import molecule
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.apps._pes_pyg import Potential
from matgl.ext._ase_pyg import Atoms2Graph, M3GNetCalculator, MolecularDynamics, PESCalculator, Relaxer
from matgl.models._tensornet_pyg import TensorNet


@pytest.fixture
def model_tensornet():
    return TensorNet(
        elment_types=["Mo", "S"], is_intensive=False, units=64, use_smooth=True, max_n=5, rbf_type="SphericalBessel"
    )


def test_PESCalculator_and_M3GNetCalculator(MoS, model_tensornet):
    adaptor = AseAtomsAdaptor()
    s_ase = adaptor.get_atoms(MoS)  # type: ignore
    ff = Potential(model=model_tensornet)
    ff.calc_hessian = True
    calc = PESCalculator(potential=ff, state_attr=None, stress_unit="eV/A3")
    s_ase.set_calculator(calc)
    assert isinstance(s_ase.get_potential_energy(), float)
    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]
    #    np.testing.assert_allclose(s_ase.get_potential_energy(), -10.824362, atol=1e-5, rtol=1e-6)

    calc = PESCalculator(potential=ff, state_attr=torch.tensor([0.0, 0.0]))
    s_ase.set_calculator(calc)
    assert isinstance(s_ase.get_potential_energy(), float)
    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]
    #    np.testing.assert_allclose(s_ase.get_potential_energy(), -10.824362, atol=1e-5, rtol=1e-6)

    calc = M3GNetCalculator(potential=ff)
    s_ase.set_calculator(calc)
    assert isinstance(s_ase.get_potential_energy(), float)
    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]
    #    np.testing.assert_allclose(s_ase.get_potential_energy(), -10.824362, atol=1e-5, rtol=1e-6)
    with pytest.raises(ValueError, match=r"Unsupported stress_unit: Pa. Must be 'GPa' or 'eV/A3'."):
        PESCalculator(potential=ff, stress_unit="Pa")


def test_PESCalculator_mol(AcAla3NHMe):
    adaptor = AseAtomsAdaptor()
    mol = adaptor.get_atoms(AcAla3NHMe)
    ff = matgl.load_model("pretrained_models/TensorNetPYG-MatPES-PBE-v2025.1-PES/")
    calc = PESCalculator(potential=ff)
    mol.set_calculator(calc)
    assert isinstance(mol.get_potential_energy(), float)
    assert list(mol.get_forces().shape) == [42, 3]
    np.testing.assert_allclose(mol.get_potential_energy(), -247.286789, atol=1e-3)


def test_Relaxer(MoS):
    pot = matgl.load_model("pretrained_models/TensorNetPYG-MatPES-PBE-v2025.1-PES//")
    r = Relaxer(pot)
    results = r.relax(MoS, traj_file="MoS_relax.traj")
    s = results["final_structure"]
    traj = results["trajectory"].as_pandas()
    assert s.lattice.a < 3.5
    assert traj["energies"].iloc[-1] < traj["energies"].iloc[0]
    for t in results["trajectory"]:
        assert len(t) == 5
    assert os.path.exists("MoS_relax.traj")
    os.remove("MoS_relax.traj")


def test_get_graph_from_atoms(LiFePO4):
    adaptor = AseAtomsAdaptor()
    structure_ase = adaptor.get_atoms(LiFePO4)
    a2g = Atoms2Graph(element_types=["Li", "Fe", "P", "O"], cutoff=4.0)
    graph, _, state = a2g.get_graph(structure_ase)
    # check the number of nodes
    assert np.allclose(graph.num_nodes, len(structure_ase.get_atomic_numbers()))
    # check the atomic feature of atom 0
    assert np.allclose(graph.node_type.detach().numpy()[0], 0)
    # check the atomic feature of atom 4
    assert np.allclose(graph.node_type.detach().numpy()[4], 1)
    # check the number of bonds
    assert np.allclose(graph.num_edges, 704)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])


def test_get_graph_from_atoms_mol():
    mol = molecule("CH4")
    a2g = Atoms2Graph(element_types=["H", "C"], cutoff=4.0)
    graph, _, state = a2g.get_graph(mol)
    # check the number of nodes
    assert np.allclose(graph.num_nodes, len(mol.get_atomic_numbers()))
    # check the atomic feature of atom 0
    assert np.allclose(graph.node_type.detach().numpy()[0], 1)
    # check the atomic feature of atom 4
    assert np.allclose(graph.node_type.detach().numpy()[1], 0)
    # check the number of bonds
    assert np.allclose(graph.num_edges, 20)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])


def test_molecular_dynamics(MoS2):
    pot = matgl.load_model("pretrained_models/TensorNetPYG-MatPES-PBE-v2025.1-PES//")
    for ensemble in [
        "nvt",
        "nve",
        "nvt_langevin",
        "nvt_andersen",
        "nvt_bussi",
        "npt",
        "npt_berendsen",
        "npt_nose_hoover",
    ]:
        md = MolecularDynamics(MoS2, potential=pot, ensemble=ensemble, taut=0.1, taup=0.1, compressibility_au=10)
        md.run(10)
        assert md.dyn is not None
        md.set_atoms(MoS2)
    md = MolecularDynamics(MoS2, potential=pot, ensemble=ensemble, taut=None, taup=None, compressibility_au=10)
    md.run(10)
    with pytest.raises(ValueError, match="Ensemble not supported"):
        MolecularDynamics(MoS2, potential=pot, ensemble="notanensemble")
