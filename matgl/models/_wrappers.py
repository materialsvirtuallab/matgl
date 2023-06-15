"""Implementations of pseudomodels that wraps other models."""
from __future__ import annotations

from torch import nn

from matgl.data.transformer import Transformer
from matgl.utils.io import IOMixIn


class TransformedTargetModel(nn.Module, IOMixIn):
    """A model where the target is first transformed prior to training and the reverse transformation is performed for
    predictions. This is modelled after scikit-learn's TransformedTargetRegressor. It should be noted that this model
    is almost never used for training since the general idea is to use the transformed target for loss computation.
    Instead, it is created after a model has been fitted for serialization for end user to call the model to perform
    predictions without having to worry about what target transformations have been performed.
    """

    # Model version number.
    __version__ = 1

    def __init__(self, model: nn.Module, target_transformer: Transformer):
        """Args:
        model: Input model
        target_transformer: Transformer for target.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.transformer = target_transformer

    def forward(self, *args, **kwargs):
        """Args:
            *args: Passthrough to parent model.forward method.
            **kwargs: Passthrough to parent model.forward method.

        Returns:
            Inverse transformed output.
        """
        output = self.model.forward(*args, **kwargs)
        return self.transformer.inverse_transform(output)

    def __repr__(self):
        return f"TransformedTargetModel:\n\tModel: {self.model.__repr()}\n\tTransformer: {self.transformer.__repr__()}"

    def predict_structure(self, *args, **kwargs):
        """Pass through to parent model.predict_structure with inverse transform.

        Args:
            *args: Pass-through to self.model.predict_structure.
            **kwargs: Pass-through to self.model.predict_structure.

        Returns:
            Transformed answer.
        """
        return self.transformer.inverse_transform(self.model.predict_structure(*args, **kwargs))
