---
layout: page
title: Change Log
nav_order: 2
---

# Change Log

## 0.5.6
- Minor internal refactoring of basis expansions into `_basis.py`. (@lbluque)

## 0.5.5
- Critical bug fix for code regression affecting pre-loaded models.

## 0.5.4
- M3GNet Formation energy model added, with example notebook.
- M3GNet.predict_structure method added.
- Massively improved documentation at http://matgl.ai.

## 0.5.3
- Minor doc and code usability improvements.

## 0.5.2
- Minor improvements to model versioning scheme.
- Added `matgl.get_available_pretrained_models()` to help with model discovery.
- Misc doc and error message improvements.

## 0.5.1
- Model versioning scheme implemented.
- Added convenience method to clear cache.

## 0.5.0
- Model serialization has been completely rewritten to make it easier to use models out of the box.
- Convenience method `matgl.load_model` is now the default way to load models.
- Added a TransformedTargetModel.
- Enable serialization of Potential.
- IMPORTANT: Pre-trained models have been reserialized. These models can only be used with v0.5.0+!

## 0.4.0
- Pre-trained M3GNet universal potential
- Pytorch lightning training utility.

## v0.3.0
- Major refactoring of MEGNet and M3GNet models and organization of internal implementations. Only key API are exposed
  via matgl.models or matgl.layers to hide internal implementations (which may change).
- Pre-trained models ported over to new implementation.
- Model download now implemented.

## v0.2.1
- Fixes for pre-trained model download.
- Speed up M3GNet 3-body computations.

## v0.2.0
- Pre-trained MEGNet models for formation energies and band gaps are now available.
- MEGNet model implemented with `predict_structure` convenience method.
- Example notebook demonstrating pre-trained model usage is available.

## v0.1.0
- Initial working version with m3gnet and megnet.
