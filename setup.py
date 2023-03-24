from __future__ import annotations

import os
import re

import numpy
from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("matgl/__init__.py", encoding="utf-8") as fd:
    for line in fd.readlines():
        m = re.search('__version__ = "(.*)"', line)
        if m:
            version = m.group(1)
            break

setup(
    name="matgl",
    version=version,
    author="Tsz Wai Ko, Marcel Nassar, Ji Qi, Santiago Miret, Shyue Ping Ong",
    author_email="ongsp@eng.ucsd.edu",
    maintainer="Shyue Ping Ong",
    maintainer_email="ongsp@eng.ucsd.edu",
    description="MatGL (Materials Graph Library) is a framework for graph deep learning for materials science. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "materials",
        "interatomic potential",
        "force field",
        "science",
        "property prediction",
        "AI",
        "machine learning",
        "graph",
        "deep learning",
    ],
    packages=find_packages(),
    package_data={
        "matgl": ["*.json", "*.md"],
        "matgl.utils": ["*.npy"],
    },
    include_package_data=True,
    install_requires=(
        "torch",
        "dgl",
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_dirs=[numpy.get_include()],
)
