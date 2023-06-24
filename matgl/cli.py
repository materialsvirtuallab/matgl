"""Command line interface for matgl."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

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
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
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
            print("Final structure")
            print(final_structure)

    return 0


def predict_structure(args):
    """
    Use MatGL models to perform predictions on structures.

    Args:
        args: Args from CLI.
    """
    model = matgl.load_model(args.model)
    if args.infile:
        for f in args.infile:
            structure = Structure.from_file(f)
            val = model.predict_structure(structure)
            print(f"{args.model} prediction for {f}: {val}.")
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

    try:
        return args.func(args)
    except AttributeError:
        parser.print_help()
        sys.exit(-1)
