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
    # Call the model, load pretrained weights
    return matgl.load_model(f"M3GNet-{version}", **kwargs)
