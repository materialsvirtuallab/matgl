# Description

This model is a M3GNet potential for 4 elements including H, C, N, O. It has broad applications in the
dynamic simulations of organic molecules.

# Training dataset

ANI-1x-Subset: 300K MD simulations and Materials Project ground state calculations.
- Training set size: 991735
- Validation set size: 248355
- Test set size: 248355

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 4.565 meV/atom, 92 meV/A
- Validation: 4.592 meV/atom, 93 meV/A
- Test: 3.746 meV/atom, 85 meV/A

# References

```txt
Ko, T.W., Deng, B., Nassar, M. et al. Materials Graph Library (MatGL), an open-source graph deep learning library for materials science and chemistry. npj Computation Materials 11, 253 (2025).
```
