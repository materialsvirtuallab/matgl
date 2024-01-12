"""This is an integration test file that checks on pre-trained models to ensure they still work."""
from __future__ import annotations

import matgl
import pytest


def test_form_e(LiFePO4):
    model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    for _i in range(3):
        # This loop ensures there is no stochasticity in the prediction, a problem in v1 of the models.
        assert model.predict_structure(LiFePO4) == pytest.approx(-2.5489, 3)


def test_loading_all_models():
    """
    Test that all pre-trained models at least load.
    """
    for m in matgl.get_available_pretrained_models():
        assert matgl.load_model(m) is not None
