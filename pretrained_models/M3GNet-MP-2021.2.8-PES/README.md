# Aim

This model is a M3GNet universal potential for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials. This is essentially a retrained version of the M3GNet universal potential originally
implemented in tensorflow. It should be noted that the structures with high forces (> 100 eV/A) and no bond are removed
in the dataset.

# Training dataset

MP-2021.2.8: Materials Project structure relaxations as of 2021.2.8.
Number of structures for training: 167237
Number of structures for validation: 18584

# Performance metrics

The reported numbers are mean absolute error of energies, forces and stresses
Train: 19.977 eV/atom, 0.063 eV/A, 0.272 GPa
Valid: 23.713 eV/atom, 0.070 eV/A, 0.394 GPa

# References
Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
2, 718–728 (2022).
