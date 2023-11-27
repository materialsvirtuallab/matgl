"""Entrypoints for Pytorch Hub."""
from __future__ import annotations

dependencies = ["torch", "matgl"]

import matgl  # noqa


# resnet18 is the name of entrypoint
def m3gnet_universal_potential(version="MP-2021.2.8-DIRECT", **kwargs):
    """M3GNet Universal Potential model.

    Args:
        version (str): Defaults to "MP-2021.2.8-DIRECT". Other versions available.
        **kwargs: Pass through to matgl.load_model.
    """
    return matgl.load_model(f"M3GNet-{version}-PES", **kwargs)


def m3gnet_formation_energy(**kwargs):
    """M3GNet Formation Energy Model.

    Args:
        version (str): Defaults to "MP-2018.6.1-EForm". Other versions available.
        **kwargs: Pass through to matgl.load_model.
    """
    return matgl.load_model("M3GNet-MP-2018.6.1-Eform", **kwargs)


def megnet_formation_energy(**kwargs):
    """MEGNet Formation Energy Model.

    Args:
        **kwargs: Pass through to matgl.load_model.
    """
    return matgl.load_model("MEGNet-MP-2018.6.1-Eform", **kwargs)


def megnet_band_gap_mfi(**kwargs):
    """MEGNet Multi-fidelity Band Gap Model.

    Args:
        **kwargs: Pass through to matgl.load_model.
    """
    return matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi", **kwargs)
