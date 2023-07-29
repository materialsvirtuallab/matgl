"""This is an integration test file that checks on pre-trained models to ensure they still work."""
from __future__ import annotations

import pytest

import matgl


def test_form_e(LiFePO4):
    model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    for _i in range(3):
        # This loop ensures there is no stochasticity in the prediction, a problem in v1 of the models.
        assert model.predict_structure(LiFePO4) == pytest.approx(-2.5489, 3)
