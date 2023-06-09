"""
Interfaces to the Atomic Simulation Environment package for dynamic simulations.
"""

from __future__ import annotations

import contextlib
import io
import pickle
import sys

import dgl
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import ExpCellFilter
from ase.io import Trajectory
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from dgl.backend import tensor
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.optimization.neighbors import find_points_in_spheres

from matgl.apps.pes import Potential
from matgl.graph.converters import GraphConverter

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


class Atoms2Graph(GraphConverter):
    """
    Construct a DGL graph from ASE atoms.
    """

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """
        Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, atoms: Atoms) -> tuple[dgl.DGLGraph, list]:
        """
        Get a DGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: DGL graph
            state_attr: state features
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        Z = np.array([np.eye(len(element_types))[element_types.index(i.symbol)] for i in atoms])
        atomic_number = np.array(atoms.get_atomic_numbers())
        lattice_matrix = np.ascontiguousarray(np.array(atoms.get_cell()), dtype=float)
        volume = atoms.get_volume()
        cart_coords = np.ascontiguousarray(np.array(atoms.get_positions()), dtype=float)
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        u, v = tensor(src_id), tensor(dst_id)
        g = dgl.graph((u, v))
        g.edata["pbc_offset"] = torch.tensor(images)
        g.edata["lattice"] = tensor([[lattice_matrix] for i in range(g.num_edges())])
        g.ndata["attr"] = tensor(Z)
        g.ndata["node_type"] = tensor(np.hstack([[element_types.index(i.symbol)] for i in atoms]))
        g.ndata["pos"] = tensor(cart_coords)
        g.ndata["volume"] = tensor([volume for i in range(atomic_number.shape[0])])
        state_attr = [0.0, 0.0]
        g.edata["pbc_offshift"] = torch.matmul(tensor(images), tensor(lattice_matrix))
        return g, state_attr


class M3GNetCalculator(Calculator):
    """
    M3GNet calculator based on ase Calculator.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "hessian"]

    def __init__(
        self,
        potential: Potential,
        state_attr: torch.tensor = None,
        stress_weight: float = 1.0,
        **kwargs,
    ):
        r"""
        Args:
            potential (Potential): m3gnet.models.Potential
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = potential.calc_stresses
        self.compute_hessian = potential.calc_hessian
        self.stress_weight = stress_weight
        self.state_attr = state_attr
        self.element_types = potential.model.element_types  # type: ignore
        self.cutoff = potential.model.cutoff

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        graph, state_attr_default = Atoms2Graph(self.element_types, self.cutoff).get_graph(atoms)  # type: ignore
        if self.state_attr is not None:
            energies, forces, stresses, hessians = self.potential(graph, self.state_attr)
        else:
            energies, forces, stresses, hessians = self.potential(graph, state_attr_default)
        self.results.update(
            energy=energies.detach().cpu().numpy(),
            free_energy=energies.detach().cpu().numpy(),
            forces=forces.detach().cpu().numpy(),
        )
        if self.compute_stress:
            self.results.update(stress=stresses.detach().cpu().numpy() * self.stress_weight)
        if self.compute_hessian:
            self.results.update(hessian=hessians.detach().cpu().numpy())


class Relaxer:
    """
    Relaxer is a class for structural relaxation.
    """

    def __init__(
        self,
        potential: Potential = None,
        state_attr: torch.tensor = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 0.01,
    ):
        """
        Args:
            potential (Potential): a M3GNet potential, a str path to a saved model or a short name for saved model
                that comes with M3GNet distribution
            state_attr (torch.tensor): State attr.
            optimizer (str or ase Optimizer): the optimization algorithm.
                Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            stress_weight (float): the stress weight for relaxation.
        """
        if isinstance(optimizer, str):
            optimizer_obj = OPTIMIZERS.get(optimizer, None)
        elif optimizer is None:
            raise ValueError("Optimizer cannot be None")
        else:
            optimizer_obj = optimizer

        self.opt_class: Optimizer = optimizer_obj
        self.calculator = M3GNetCalculator(
            potential=potential, state_attr=state_attr, stress_weight=stress_weight  # type: ignore
        )
        self.relax_cell = relax_cell
        self.potential = potential
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str = None,
        interval=1,
        verbose=False,
        **kwargs,
    ):
        r"""
        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            **kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = ExpCellFilter(atoms)
            optimizer = self.opt_class(atoms, **kwargs)
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            obs()
        if traj_file is not None:
            obs.save(traj_file)
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms

        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
        }


class TrajectoryObserver:
    """
    Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """
        The logic for saving the properties of an Atoms during the relaxation.
        """
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """Calculate the potential energy."""
        energy = self.atoms.get_potential_energy()
        return energy

    def save(self, filename: str) -> None:
        """
        Save the trajectory to file
        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)


class MolecularDynamics:
    """
    Molecular dynamics class.
    """

    def __init__(
        self,
        atoms: Atoms,
        potential: Potential,
        state_attr: torch.tensor = None,
        ensemble: str = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        compressibility_au: float | None = None,
        trajectory: str | Trajectory | None = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
    ):
        """
        Args:
            atoms (Atoms): atoms to run the MD
            potential (Potential): potential for calculating the energy, force,
                stress of the atoms
            state_attr (torch.tensor): State attr.
            ensemble (str): choose from 'nvt' or 'npt'. NPT is not tested,
                use with extra caution
            temperature (float): temperature for MD simulation, in K
            timestep (float): time step in fs
            pressure (float): pressure in eV/A^3
            taut (float): time constant for Berendsen temperature coupling
            taup (float): time constant for pressure coupling
            compressibility_au (float): compressibility of the material in A^3/eV
            trajectory (str or Trajectory): Attach trajectory object
            logfile (str): open this file for recording MD outputs
            loginterval (int): write to log file every interval steps
            append_trajectory (bool): Whether to append to prev trajectory.
        """
        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms = atoms
        self.atoms.set_calculator(M3GNetCalculator(potential=potential, state_attr=state_attr))

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs

        if ensemble.lower() == "nvt":
            self.dyn = NVTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt":
            """
            NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
            This is a more flexible scheme that fixes three angles of the unit
            cell but allows three lattice parameter to change independently.
            """

            self.dyn = Inhomogeneous_NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                # append_trajectory=append_trajectory,
                # this option is not supported in ASE at this point (I have sent merge request there)
            )

        elif ensemble.lower() == "npt_berendsen":
            """

            This is a similar scheme to the Inhomogeneous_NPTBerendsen.
            This is a less flexible scheme that fixes the shape of the
            cell - three angles are fixed and the ratios between the three
            lattice constants.

            """

            self.dyn = NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        else:
            raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep

    def run(self, steps: int):
        """
        Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """
        Set new atoms to run MD
        Args:
            atoms (Atoms): new atoms for running MD.
        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.set_calculator(calculator)
