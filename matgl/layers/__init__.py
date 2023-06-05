"""
This package implements the layers for M*GNet.
"""
from __future__ import annotations

from matgl.layers._activations import SoftExponential, SoftPlus2
from matgl.layers._atom_ref import AtomRef
from matgl.layers._bond import BondExpansion
from matgl.layers._core import MLP, EdgeSet2Set, GatedMLP
from matgl.layers._embedding import EmbeddingBlock
from matgl.layers._graph_convolution import M3GNetBlock, M3GNetGraphConv, MEGNetBlock, MEGNetGraphConv
from matgl.layers._readout import ReduceReadOut, Set2SetReadOut, WeightedReadOut, WeightedReadOutPair
from matgl.layers._three_body import SphericalBesselWithHarmonics, ThreeBodyInteractions
