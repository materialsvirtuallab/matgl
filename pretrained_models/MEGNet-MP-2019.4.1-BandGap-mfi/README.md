# Aim

This model is a multif-fidelity MEGNet band gap model for 89 elements of the periodic table. It contains the band gap
for most of materials with 4 fidelities ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN). This is essentially a retrained
version of the MEGNet band gap model originally implemented in tensorflow.

# Training dataset

MP-2019.4.1: Materials Project formation energy as of 2019.4.1.
Number of structures for training: 48912
Number of structures for validation: 879

# Performance metrics

The reported numbers are mean absolute error of formation energy in eV
Train: 0.075 eV
Valid: 0.314 eV

# References

```txt
Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. Learning Properties of Ordered and Disordered Materials from
Multi-Fidelity Data. Nature Computational Science 2021, 1, 46–53. https://doi.org/10.1038/s43588-020-00002-x.
```
