# Aim

This model is a MEGNet formation energy model for 89 elements of the periodic table. It contains the formation
energy for most materials. This is essentially a retrained version of the MEGNet formation energy model
originally implemented in tensorflow.

# Training dataset

MP-2018.6.1: Materials Project formation energy as of 2018.6.1.
- Training set size: : 62315
- Validation set size: 3461
- Test set size: 3463

# Performance metrics

MAE of formation energy in eV/atom.
- Training: 0.010 eV/atom
- Validation: 0.029 eV/atom
- Test: 0.028 eV/atom

# References

```txt
Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.
```
