# Description

This model is a CHGNet universal potential trained from the Materials Project trajectory (MPtrj) dataset
that contains over 1.5 million structures with 89 elements.
This Matgl implementation has slight modification from original pytorch implementation by adding directed edge updates.

# Training dataset

MPtrj-2022.9: Materials Project trajectory dataset that contains GGA and GGA+U static and relaxation calculations.
- Train set size:
- Validation set size: 9284
- Test set size:

# Performance metrics
## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 26.45 meV/atom, 0.049 eV/A, 0.173 GPa
- Validation: 30.31 meV/atom, 0.070 eV/A, 0.297 GPa
- Test: 30.80 meV/atom, 0.066 eV/A, 0.296 GPa

# References

```txt
Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.
Nat. Mach. Intell. 1–11 (2023) doi:10.1038/s42256-023-00716-3.
  ```
