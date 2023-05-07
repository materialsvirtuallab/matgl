# Changelog

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
