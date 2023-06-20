---
layout: page
title: Developer Guide
nav_order: 4
---

# Developer Guide

This is a developer's guide to writing new models in matgl. It is still a work in progress.

MatGL is based on the [Deep Graph Library (DGL)][dgl]. Most of the questions relating to how to implement graph deep
learning models, etc. is already covered in the extensive documentation for DGL. You are encouraged to read those. Here,
we will focus our discussion on improvements / additions / design elements specifically for matgl.

## Modular components

To maintain maximum flexibility, we have implemented the steps of using graph deep learning for materials in separate,
reusable modular components. These steps include:
- Converting materials to DGL graphs from pymatgen, ase and other materials science codes (`matgl.ext`).
- Dataset loading (`matgl.graphs`).
- Actual graph model implementations (`matgl.models`) and their internal components (`matgl.layers`).
- Special use cases of models, e.g., for interatomic potentials (`matgl.apps`).

## Controlled API exposure

The exposed API is controlled to allow for code refactoring and development. Any module that is preceded by an
underscore is a "private" implementation by convention and there are no guarantees as to backwards compatibility.
For example, the MEGNet and M3GNet models are exposed via `matgl.models` in the `__init__.py` while the actual
implementations are in `_megnet.py` and `_m3gnet.py`, respectively. This is similar to the convention adopted by
scikit-learn. As far as possible, do imports only from exposed APIs.

## Nested Models

Often, a simple input->output model architecture is often insufficient. For instance, one might want to fit a model
on a scaled or transformed version of the target. For instance, the current best practice is to fit models to the log
of the bulk modulus as it spans several orders of magnitude. Similarly, an interatomic potential has to compute
derivatives such as forces and stresses from the energies. This presents a difficulty when the model is actually
being used for end predictions as the user has to remember to invert the scaling or transformation or perform
additional steps.

To make matgl models much more friendly for end users, we implement a nested model concept. Examples include
`matgl.apps.pes.Potential`, which is an interatomic potential model that wraps around a graph model (e.g. M3GNet),
and the `matgl.models.TransformedTargetModel`, which is modelled after scikit-learn's TransformedTargetRegressor. The
goal is for users to be able to use such models directly without having to worry about the internal transformations.

## Model IO

All models subclass the `matgl.utils.io.IOMixIn` class, which is basically a wrapper around PyTorch's mechanisms for
loading and saving model state but with additional metadata to make it simpler to load models. To use this class
properly, models should subclass `torch.nn.Module` *and* `IOMixIn`. In addition, the `save_args` method should be
called after the super class call. A simpel example is as follows:

```python
import torch
from matgl.utils.io import IOMixIn

class MyModel(torch.nn.Module, IOMixIn):

    __version__ = 1

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_args(locals(), kwargs)
```

Models implementing this protocol can be saved and loaded using `model.save(path)`, `Model.load(path)` and the
convenience `matgl.load_model(path)` methods.

## Model versioning

The IOMixIn supports optional model versioning. To enable this, the model class should have an integer `__version__`
class variable. The goal is to increment this variable when architectural changes occur and saved pre-trained models
need to be invalidated. If not specified, the model is not versioned at all.

## Testing

All code contributions must be accompanied by comprehensive unittests. These tests should be added to the
appropriate mirror directory in the `tests` folder.

We use pytest. Useful fixtures have been written in the `conftest.py` file in the `tests` directory, which provides
crystals, molecules, and pre-generated graphs for reuse in tests.

## Documentation

We use Google doc style.

[dgl]: https://www.dgl.ai "DGL website"
