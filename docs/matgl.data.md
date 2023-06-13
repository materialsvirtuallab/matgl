---
layout: default
title: matgl.data.md
nav_exclude: true
---
# matgl.data package

This package implements data manipulation tools.


## matgl.data.transformer module

Module implementing various data transformers for PyTorch.


### _class_ matgl.data.transformer.LogTransformer()
Bases: `Transformer`

Performs a natural log of the data.


#### inverse_transform(data)
Invert the log (exp).

Args:

    data: Input data

Returns:

    exp(data)


#### transform(data)
Take the log of the data.

Args:

    data: Input data

Returns:

    Scaled data


### _class_ matgl.data.transformer.Normalizer(mean: float, std: float)
Bases: `Transformer`

Performs a scaling of the data by centering to the mean and dividing by the standard deviation.


#### _classmethod_ from_data(data)
Create Normalizer from data.

Args:

    data: Input data.

Returns:

    Normalizer


#### inverse_transform(data)
Invert the scaling.

Args:

    data: Scaled data

Returns:

    Unscaled data


#### transform(data)
z-score the data by subtracting the mean and dividing by the standard deviation.

Args:

    data: Input data

Returns:

    Scaled data


### _class_ matgl.data.transformer.Transformer()
Bases: `object`

Abstract base class defining a data transformer.


#### _abstract_ inverse_transform(data: Tensor)
Inverse transformation to be performed on data.

Args:

    data: Input data

Returns:

    Inverse-transformed data.


#### _abstract_ transform(data: Tensor)
Transformation to be performed on data.

Args:

    data: Input data

Returns:

    Transformed data.
