# Aim

This model is a M3GNet formation energy model for 89 elements of the periodic table. It contains the formation
<<<<<<< HEAD
energy for most of materials. This is essentially a retrained version of the MEGNet formation energy model
=======
energy for most materials. This is essentially a retrained version of the M3GNet formation energy model
>>>>>>> f92699c7eacec60bfec25149cca96ffe21838b5a
originally implemented in tensorflow.

# Training dataset

MP-2018.6.1: Materials Project formation energy as of 2018.6.1.
- Training set size: : 62315
- Validation set size: 3461
- Test set size: 3463

# Performance metrics

MAE of formation energy in eV/atom.
- Training: 0.008 eV/atom
- Validation: 0.022 eV/atom
- Test: 0.022 eV/atom

# References

```txt
Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.
```
