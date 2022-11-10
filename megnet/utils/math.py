"""
Various math function implementations.
"""

from __future__ import annotations

import os

import numpy as np

CWD = os.path.dirname(os.path.abspath(__file__))

"""
Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index） root of
order l Spherical Bessel function is the `[l, n]` entry.
"""
SPHERICAL_BESSEL_ROOTS = np.loadtxt(os.path.join(CWD, "sb_roots.txt"))
