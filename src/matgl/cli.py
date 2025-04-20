"""Command line interface for matgl."""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import MolecularDynamics, Relaxer

warnings.simplefilter("ignore")
logger = logging.getLogger("MGL")


def relax_structure(args):
    """
    Relax crystals.

    Args:
        args: Args from CLI.
    """
    for fn in args.infile:
        structure = Structure.from_file(fn)

        if args.verbose:
            logging.basicConfig(level=logging.INFO)

        logger.info(f"Initial structure\n{structure}")
        logger.info("Loading model...")
        pot = matgl.load_model(args.model)
        logger.info("Relaxing...")
        relaxer = Relaxer(potential=pot)
        relax_results = relaxer.relax(structure, fmax=0.01)
        final_structure = relax_results["final_structure"]

        if args.suffix:
            basename, ext = os.path.splitext(fn)
            outfn = f"{basename}{args.suffix}{ext}"
            final_structure.to(filename=outfn)
            print(f"Structure written to {outfn}!")
        elif args.outfile is not None:
            final_structure.to(filename=args.outfile)
            print(f"Structure written to {args.outfile}!")
        else:
            print("Lattice parameters")
            old_lattice = structure.lattice
            new_lattice = final_structure.lattice
            for param in ("a", "b", "c", "alpha", "beta", "gamma"):
                print(f"{param}: {getattr(old_lattice, param):.3f} -> {getattr(new_lattice, param):.3f}")
            print("Sites (Fractional coordinates)")

            def fmt_fcoords(fc):
                return np.array2string(fc, formatter={"float_kind": lambda x: f"{x:.5f}"})

            for old_site, new_site in zip(structure, final_structure, strict=False):
                print(f"{old_site.species}: {fmt_fcoords(old_site.frac_coords)} -> {fmt_fcoords(new_site.frac_coords)}")

    return 0


def predict_structure(args):
    """
    Use MatGL models to perform predictions on structures.

    Args:
        args: Args from CLI.
    """
    model = matgl.load_model(args.model)
    if args.infile:
        if args.model == "MEGNet-MP-2019.4.1-BandGap-mfi":
            state_dict = ["PBE", "GLLB-SC", "HSE", "SCAN"]
            for count, f in enumerate(args.infile):
                s = args.state_attr[count]  # Get the corresponding state attribute
                structure = Structure.from_file(f)
                val = model.predict_structure(structure, torch.tensor(int(s)))
                print(f"{args.model} prediction for {f} with {state_dict[int(s)]} bandgap: {val} eV.")

        else:
            for f in args.infile:
                structure = Structure.from_file(f)
                val = model.predict_structure(structure)
                print(f"{args.model} prediction for {f}: {val} eV/atom.")
    if args.mpids:
        mpr = MPRester()
        for mid in args.mpids:
            structure = mpr.get_structure_by_material_id(mid)
            val = model.predict_structure(structure)
            print(f"{args.model} prediction for {mid} ({structure.composition.reduced_formula}): {val}.")


def molecular_dynamics(args):
    """
    Use MaGL models to perform MD simulations on structures.

    Args:
        args: Args from CLI.
    """
    for file in args.infile:
        name = file.split(".")[0]
        structure = Structure.from_file(file)
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)

        logger.info(f"Initial structure\n{structure}")
        logger.info("Loading model...")
        pot = matgl.load_model(args.model)
        logger.info("Running MD...")
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)
        md = MolecularDynamics(
            atoms,
            potential=pot,
            ensemble=args.ensemble,
            pressure=args.pressure,
            timestep=args.stepsize,
            trajectory=name + ".traj",
            logfile=name + ".log",
            temperature=args.temp,
            taut=args.taut,
            taup=args.taup,
            friction=args.friction,
            andersen_prob=args.andersen_prob,
            ttime=args.ttime,
            pfactor=args.pfactor,
            external_stress=args.external_stress,
            compressibility_au=args.compressibility_au,
            loginterval=args.loginterval,
            append_trajectory=args.append_trajectory,
            mask=args.mask,
        )
        md.run(args.nsteps)
    return 0


def clear_cache(args):
    """
    Clear cache command.

    Args:
        args: Args from CLI.
    """
    matgl.clear_cache(not args.yes)


def main():
    """Handle main."""
    parser = argparse.ArgumentParser(
        description="""
    This script works based on several sub-commands with their own options. To see the options for the
    sub-commands, type "mgl sub-command -h".""",
        epilog="""Author: MatGL Development Team""",
    )

    subparsers = parser.add_subparsers()

    p_relax = subparsers.add_parser("relax", help="Relax crystal structures.")

    p_relax.add_argument(
        "-i",
        "--infile",
        dest="infile",
        nargs="+",
        required=True,
        help="Input files containing structure. Any format supported by pymatgen's Structure.from_file method.",
    )

    p_relax.add_argument(
        "-m",
        "--model",
        dest="model",
        choices=[m for m in matgl.get_available_pretrained_models() if m.endswith("PES")],
        default="M3GNet-MP-2021.2.8-DIRECT-PES",
        help="Model to use.",
    )

    p_relax.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output.",
    )

    groups = p_relax.add_mutually_exclusive_group(required=False)
    groups.add_argument(
        "-s",
        "--suffix",
        dest="suffix",
        help="Suffix to be added to input file names for relaxed structures. E.g., _relax.",
    )

    groups.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        help="Output filename.",
    )

    p_relax.set_defaults(func=relax_structure)

    p_predict = subparsers.add_parser("predict", help="Perform a prediction with pre-trained models.")

    groups = p_predict.add_mutually_exclusive_group(required=True)
    groups.add_argument(
        "-p",
        "--mpids",
        dest="mpids",
        nargs="+",
        help="Materials Project IDs. Requires mp-api to be installed and set up.",
    )

    groups.add_argument(
        "-i",
        "--infile",
        dest="infile",
        nargs="+",
        help="Input files containing structure. Any format supported by pymatgen's Structure.from_file method.",
    )

    p_predict.add_argument(
        "-s",
        "--state",
        dest="state_attr",
        nargs="+",
        help="state attributes containing label. This should be an integer.",
    )

    p_predict.add_argument(
        "-m",
        "--model",
        dest="model",
        choices=matgl.get_available_pretrained_models(),
        required=True,
        help="Model to use",
    )

    p_predict.set_defaults(func=predict_structure)

    # MD simulations
    p_md = subparsers.add_parser("md", help="Perform MD simulations with pre-trained and customized models.")

    p_md.add_argument(
        "-i",
        "--infile",
        nargs="+",
        dest="infile",
        required=True,
        help="Input files containing structure. Any format supported by pymatgen Structure.from_file method.",
    )

    p_md.add_argument(
        "-m",
        "--model",
        dest="model",
        #        choices=[m for m in matgl.get_available_pretrained_models() if m.endswith("PES")],
        default="M3GNet-MP-2021.2.8-DIRECT-PES",
        help="Path for loading MLIPs trained from MatGL. Default='M3GNet-MP-2021.2.8-DIRECT-PES'.",
    )

    p_md.add_argument(
        "-e",
        "--ensemble",
        dest="ensemble",
        choices=["nve", "nvt", "nvt_langevin", "nvt_andersen", "npt", "npt_berendsen", "npt_nose_hoover"],
        default="nve",
        help="Ensemble used for MD simulation. Default='nve'.",
    )

    p_md.add_argument(
        "-n",
        "--nsteps",
        dest="nsteps",
        type=int,
        default=100,
        help="Number of steps used for MD simulation. Default=100.",
    )

    p_md.add_argument(
        "--stepsize",
        dest="stepsize",
        type=float,
        default=1.0,
        help="Step size used for MD simulation. Default=1.0 fs.",
    )

    p_md.add_argument(
        "-t",
        "--temp",
        dest="temp",
        type=float,
        default=300.0,
        help="Temperature used for MD simulation. Default=300.0 in K.",
    )

    p_md.add_argument(
        "-p",
        "--pressure",
        dest="pressure",
        type=float,
        default=1.01325,
        help="Pressure used for MD simulation. Default=1.01325 in Bar.",
    )

    p_md.add_argument(
        "--taut",
        dest="taut",
        type=float,
        default=None,
        help="Time constant for Berendsen temperature coupling. Default is None.",
    )

    p_md.add_argument(
        "--taup",
        dest="taup",
        type=float,
        default=None,
        help="Time constant for Berendsen pressure coupling. Default is None.",
    )

    p_md.add_argument(
        "--andersen_prob",
        dest="andersen_prob",
        type=float,
        default=0.01,
        help="Random collision probability for nvt_andersen. Default is 0.01.",
    )

    p_md.add_argument(
        "--friction",
        dest="friction",
        type=float,
        default=0.001,
        help="Friction coefficient for nvt_langevin. Default is 0.001.",
    )

    p_md.add_argument(
        "--ttime",
        dest="ttime",
        type=float,
        default=25.0,
        help="Characteristic timescale of the thermostat in ASE internal units. Default is 25.0.",
    )

    p_md.add_argument(
        "--pfactor",
        dest="pfactor",
        type=float,
        default=75.0**2.0,
        help="A constant in the barostat differential equation. Default is 25.0 in eV/A$^{3}$.",
    )

    p_md.add_argument(
        "--external_stress",
        dest="external_stress",
        type=float,
        default=None,
        help="The external stress either 3x3 tensor, 6-vector or a scalar in eV/A$^{3}$. Default is None.",
    )

    p_md.add_argument(
        "--compressibility_au",
        dest="compressibility_au",
        type=float,
        default=None,
        help="Compressibility of the material in eV/A^{3}. Default is None.",
    )

    p_md.add_argument(
        "--loginterval",
        dest="loginterval",
        type=int,
        default=1,
        help="Write to log file every interval steps. Default is 1.",
    )

    p_md.add_argument(
        "--append_trajectory",
        dest="append_trajectory",
        type=bool,
        default=False,
        help="Whether to append to prev trajectory. Default is False.",
    )

    p_md.add_argument(
        "--mask",
        dest="mask",
        type=np.array,
        default=None,
        help="a symmetric 3x3 array indicating, which strain values may change for NPT simulations",
    )

    p_md.set_defaults(func=molecular_dynamics)

    p_clear = subparsers.add_parser("clear", help="Clear cache.")

    p_clear.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Skip confirmation.",
    )

    p_clear.set_defaults(func=clear_cache)

    args = parser.parse_args()

    return args.func(args)
