"""Command line interface for matgl."""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.ext.ase import Relaxer

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
                return np.array2string(fc, formatter={"float_kind": lambda x: "%.5f" % x})

            for old_site, new_site in zip(structure, final_structure):
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
