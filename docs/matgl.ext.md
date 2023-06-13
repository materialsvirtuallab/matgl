---
layout: default
title: matgl.ext.md
nav_exclude: true
---
# matgl.ext package

This package implements interfaces to external packages such as Pymatgen and the Atomic Simulation Environment.


## matgl.ext.ase module

Interfaces to the Atomic Simulation Environment package for dynamic simulations.


### _class_ matgl.ext.ase.Atoms2Graph(element_types: tuple[str, ...], cutoff: float = 5.0)
Bases: [`GraphConverter`](matgl.graph.md#matgl.graph.converters.GraphConverter)

Construct a DGL graph from ASE Atoms.


#### get_graph(atoms: Atoms)
Get a DGL graph from an input Atoms.

Args:

    atoms: Atoms object.

Returns:

    g: DGL graph
    state_attr: state features


### _class_ matgl.ext.ase.M3GNetCalculator(potential: [Potential](matgl.apps.md#matgl.apps.pes.Potential), state_attr: tensor | None = None, stress_weight: float = 1.0, \*\*kwargs)
Bases: `Calculator`

M3GNet calculator based on ase Calculator.


#### calculate(atoms: Atoms | None = None, properties: list | None = None, system_changes: list | None = None)
Args:

    atoms (ase.Atoms): ase Atoms object
    properties (list): list of properties to calculate
    system_changes (list): monitor which properties of atoms were

    > changed for new calculation. If not, the previous calculation
    > results will be loaded.


#### implemented_properties(_: List[str_ _ = ['energy', 'free_energy', 'forces', 'stress', 'hessian'_ )
Properties calculator can handle (energy, forces, …)


### _class_ matgl.ext.ase.MolecularDynamics(atoms: Atoms, potential: [Potential](matgl.apps.md#matgl.apps.pes.Potential), state_attr: torch.tensor = None, ensemble: str = 'nvt', temperature: int = 300, timestep: float = 1.0, pressure: float = 6.324209121801212e-07, taut: float | None = None, taup: float | None = None, compressibility_au: float | None = None, trajectory: str | Trajectory | None = None, logfile: str | None = None, loginterval: int = 1, append_trajectory: bool = False)
Bases: `object`

Molecular dynamics class.


#### run(steps: int)
Thin wrapper of ase MD run.

Args:

    steps (int): number of MD steps


#### set_atoms(atoms: Atoms)
Set new atoms to run MD
Args:

> atoms (Atoms): new atoms for running MD.


### _class_ matgl.ext.ase.Relaxer(potential: [Potential](matgl.apps.md#matgl.apps.pes.Potential) = None, state_attr: torch.tensor = None, optimizer: Optimizer | str = 'FIRE', relax_cell: bool = True, stress_weight: float = 0.01)
Bases: `object`

Relaxer is a class for structural relaxation.


#### relax(atoms: Atoms, fmax: float = 0.1, steps: int = 500, traj_file: str | None = None, interval=1, verbose=False, \*\*kwargs)
Args:

    atoms (Atoms): the atoms for relaxation
    fmax (float): total force tolerance for relaxation convergence.

    > Here fmax is a sum of force and stress forces

    steps (int): max number of steps for relaxation
    traj_file (str): the trajectory file for saving
    interval (int): the step interval for saving the trajectories
    verbose (bool): Whether to have verbose output.


    ```
    **
    ```

    kwargs: Kwargs pass-through to optimizer.


### _class_ matgl.ext.ase.TrajectoryObserver(atoms: Atoms)
Bases: `object`

Trajectory observer is a hook in the relaxation process that saves the
intermediate structures.


#### compute_energy()
Calculate the potential energy.


#### save(filename: str)
Save the trajectory to file
Args:

> filename (str): filename to save the trajectory.

## matgl.ext.pymatgen module

Interface with pymatgen objects.


### _class_ matgl.ext.pymatgen.Molecule2Graph(element_types: tuple[str, ...], cutoff: float = 5.0)
Bases: [`GraphConverter`](matgl.graph.md#matgl.graph.converters.GraphConverter)

Construct a DGL graph from Pymatgen Molecules.


#### get_graph(mol: Molecule)
Get a DGL graph from an input molecule.


* **Parameters**

    **mol** – pymatgen molecule object



* **Returns**

    (dgl graph, state features)



### _class_ matgl.ext.pymatgen.Structure2Graph(element_types: tuple[str, ...], cutoff: float = 5.0)
Bases: [`GraphConverter`](matgl.graph.md#matgl.graph.converters.GraphConverter)

Construct a DGL graph from Pymatgen Structure.


#### get_graph(structure: Structure)
Get a DGL graph from an input Structure.


* **Parameters**

    **structure** – pymatgen structure object



* **Returns**

    g: DGL graph
    state_attr: state features



### matgl.ext.pymatgen.get_element_list(train_structures: list[Structure | Molecule])
Get the dictionary containing elements in the training set for atomic features.

Args:

    train_structures: pymatgen Molecule/Structure object

Returns:

    Tuple of elements covered in training set
