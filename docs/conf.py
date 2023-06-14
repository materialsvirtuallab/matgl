# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
#
# sys.path.insert(0, os.path.abspath("."))
# sys.path.insert(0, os.path.dirname(".."))
# sys.path.insert(0, os.path.dirname("../matgl"))
# sys.path.insert(0, os.path.dirname("../.."))

# -- Project information -----------------------------------------------------
from __future__ import annotations

project = "matgl"
copyright = "2022, Materials Virtual Lab"
author = "Tsz Wai Ko, Marcel Nassar, Ji Qi, Santiago Miret, Shyue Ping Ong"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Napoleon is necessary to parse Google style docstrings. Markdown builder allows the generation of markdown output.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser", "sphinx_markdown_builder"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autoclass_content = "both"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

myst_heading_anchors = 3

autodoc_default_options = {"private-members": False}
