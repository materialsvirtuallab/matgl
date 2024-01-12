---
layout: default
title: matgl.data.md
nav_exclude: true
---

# matgl.data package

This package implements data manipulation tools.

## matgl.data.transformer module

Module implementing various data transformers for PyTorch.

### *class* matgl.data.transformer.LogTransformer

Bases: `Transformer`

Performs a natural log of the data.

#### inverse_transform(data)

Invert the log (exp).

* **Parameters:**
  **data** – Input data
* **Returns:**
  exp(data)

#### transform(data)

Take the log of the data.

* **Parameters:**
  **data** – Input data
* **Returns:**
  Scaled data

### *class* matgl.data.transformer.Normalizer(mean: float, std: float)

Bases: `Transformer`

Performs a scaling of the data by centering to the mean and dividing by the standard deviation.

* **Parameters:**
  * **mean** – Mean of the data
  * **std** – Standard deviation of the data.

#### *classmethod* from_data(data)

Create Normalizer from data.

* **Parameters:**
  **data** – Input data.
* **Returns:**
  Normalizer

#### inverse_transform(data)

Invert the scaling.

* **Parameters:**
  **data** – Scaled data
* **Returns:**
  Unscaled data

#### transform(data)

z-score the data by subtracting the mean and dividing by the standard deviation.

* **Parameters:**
  **data** – Input data
* **Returns:**
  Scaled data

### *class* matgl.data.transformer.Transformer

Bases: `object`

Abstract base class defining a data transformer.

#### *abstract* inverse_transform(data: Tensor)

Inverse transformation to be performed on data.

* **Parameters:**
  **data** – Input data
* **Returns:**
  Inverse-transformed data.

#### *abstract* transform(data: Tensor)

Transformation to be performed on data.

* **Parameters:**
  **data** – Input data
* **Returns:**
  Transformed data.