"""
Implementations of pseudomodels that wraps other models.
"""
from __future__ import annotations

from torch import nn

from matgl.data.transformer import Transformer


class TransformedTargetModel(nn.Module):
    """
    A model where the target is first transformed prior to training and the reverse transformation is performed for
    predictions. This is modelled after scikit-learn's TransformedTargetRegressor
    """

    def __init__(self, model: nn.Module, target_transformer: Transformer):
        """

        Args:
            model:
            target_transformer:
        """
        self.model = model
        self.transformer = target_transformer

    def forward(self, *args, **kwargs):
        r"""
        Args:
            *args: Passthrough to parent model.forward method.
            **kwargs: Passthrough to parent model.forward method.

        Returns:
            Inverse transformed output.
        """
        output = self.model.forward(*args, **kwargs)
        return self.transformer.inverse_transform(output)

    def __repr__(self):
        return f"TransformedTargetModel:\nModel: {self.model.__repr()}\nTransformer: {self.transformer.__repr__()}"
