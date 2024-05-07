# Description

This model is a CHGNet universal potential trained from the Materials Project trajectory (MPtrj) dataset
that contains over 1.5 million structures with 89 elements.
This Matgl implementation has slight modification from original pytorch implementation by adding directed edge updates.

# Training dataset

MPtrj-2022.9: Materials Project trajectory dataset that contains GGA and GGA+U static and relaxation calculations.
- Train-Val-Test splitting with mp-id: 0.95 - 0.5
- Train set size: 1499043
- Validation set size: 79719
- Test set size: 0

# Performance metrics
## Training and validation errors

| partition  | Energy (meV/atom) | Force (meV/A) | stress (GPa) | magmom (muB) |
| ---------- | ----------------- | ------------- | ------------ | ------------ |
| Train      | 25.6              | 47.6          | 0.177        | 0.017        |
| Validation | 27.7              | 62.5          | 0.288        | 0.017        |


# References

```txt
Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.
Nat. Mach. Intell. 1–11 (2023) doi:10.1038/s42256-023-00716-3.
  ```

####  Date: 2024.2.13
####  Author: Bowen Deng
