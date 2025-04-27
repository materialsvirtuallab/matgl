#Description

This model is a TensorNet universal potential for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials.

# Training dataset

MatPESv2025.1-r2SCAN: 300K MD simulations and Materials Project ground state calculations.
- Training set size: 349017
- Validation set size: 19394
- Test set size: 19396

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 32 meV/atom, 139 meV/A, 0.653 GPa
- Validation: 34 meV/atom, 163 meV/A, 0.754 GPa
- Test: 34 meV/atom, 163 meV/A, 0.754 GPa

# References

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```
