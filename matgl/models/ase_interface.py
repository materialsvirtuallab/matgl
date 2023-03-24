"""
Dynamics calculations using M3GNet
"""

from __future__ import annotations

import contextlib
import io
import pickle
import sys

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
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .potential import Potential

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


class M3GNetCalculator(Calculator):
    """
    M3GNet calculator based on ase Calculator
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress", "hessian"]

    def __init__(
        self,
        potential: Potential,
        graph_attr: torch.tensor = None,
        compute_stress: bool = True,
        stress_weight: float = 1.0,
        **kwargs,
    ):
        """

        Args:
            potential (Potential): m3gnet.models.Potential
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = potential.calc_stresses
        self.compute_hessian = potential.calc_hessian
        self.stress_weight = stress_weight
        self.graph_attr = graph_attr

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
        Returns:

        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        graph, graph_attr_default = self.potential.graph_converter.get_graph_from_atoms(atoms)
        if self.graph_attr is not None:
            energies, forces, stresses, hessians = self.potential(graph, self.graph_attr)
        else:
            energies, forces, stresses, hessians = self.potential(graph, graph_attr_default)
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
    Relaxer is a class for structural relaxation
    """

    def __init__(
        self,
        potential: Potential = None,
        graph_attr: torch.tensor = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 0.01,
    ):
        """

        Args:
            potential (Potential): a M3GNet potential,
                a str path to a saved model or a short name for saved model
                that comes with M3GNet distribution
            optimizer (str or ase Optimizer): the optimization algorithm.
                Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            stress_weight (float): the stress weight for relaxation
        """

        if isinstance(optimizer, str):
            optimizer_obj = OPTIMIZERS.get(optimizer, None)
        elif optimizer is None:
            raise ValueError("Optimizer cannot be None")
        else:
            optimizer_obj = optimizer

        self.opt_class: Optimizer = optimizer_obj
        self.calculator = M3GNetCalculator(
            potential=potential, graph_attr=graph_attr, stress_weight=stress_weight  # type: ignore
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
        """

        Args:
            atoms (Atoms): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
                Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            **kwargs:
        Returns:
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
    intermediate structures
    """

    def __init__(self, atoms: Atoms):
        """
        Args:
            atoms (Atoms): the structure to observe
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """
        The logic for saving the properties of an Atoms during the relaxation
        Returns:
        """
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """
        calculate the energy, here we just use the potential energy
        Returns:
        """
        energy = self.atoms.get_potential_energy()
        return energy

    def save(self, filename: str):
        """
        Save the trajectory to file
        Args:
            filename (str): filename to save the trajectory
        Returns:
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "energy": self.energies,
                    "forces": self.forces,
                    "stresses": self.stresses,
                    "atom_positions": self.atom_positions,
                    "cell": self.cells,
                    "atomic_number": self.atoms.get_atomic_numbers(),
                },
                f,
            )


class MolecularDynamics:
    """
    Molecular dynamics class
    """

    def __init__(
        self,
        atoms: Atoms,
        potential: Potential,
        graph_attr: torch.tensor = None,
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
            append_trajectory (bool): Whether to append to prev trajectory
        """

        if isinstance(atoms, (Structure, Molecule)):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms = atoms
        self.atoms.set_calculator(M3GNetCalculator(potential=potential, graph_attr=graph_attr))

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
        Thin wrapper of ase MD run
        Args:
            steps (int): number of MD steps
        Returns:

        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """
        Set new atoms to run MD
        Args:
            atoms (Atoms): new atoms for running MD

        Returns:

        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.set_calculator(calculator)
