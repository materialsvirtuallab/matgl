# Description

This model is a CHGNet universal potential trained from the Materials Potential Energy Surface (MatPES) dataset
that contains over 1.5 million structures with 89 elements.
This Matgl implementation has slight modification from original pytorch implementation by adding directed edge updates.

# Training dataset

MatPES-PBE-2024.11: Materials Energy Surface dataset that contains off-equilibrium PBE static calculations.
- Train-Val-Test splitting with mp-id: 0.9 - 0.5 - 0.5
- Train set size: 391241
- Validation set size: 21736
- Test set size: 21735

# Performance metrics
## Training and validation errors

| partition  | Energy (meV/atom) | Force (meV/A) | stress (GPa) | magmom (muB) |
| ---------- |-------------------|---------------|--------------|--------------|
| Train      | 26.54             | 81.4          | 0.375        | 0.066        |
| Validation | 32.02             | 123.8         | 0.617        | 0.067        |
| Test       | 30.70             | 136.0         | 0.642        | 0.066        |


# References

```txt
Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.
Nat. Mach. Intell. 1–11 (2023) doi:10.1038/s42256-023-00716-3.
  ```

####  Date: 2025.2.10
####  Author: Bowen Deng