# Description

This model is a CHGNet universal potential trained from the Materials Project trajectory (MPtrj) dataset
that contains over 1.5 million structures with 89 elements.
This Matgl implementation has slight modification from original pytorch implementation by adding directed edge updates.

# Training dataset

MPtrj-2022.9: Materials Project trajectory dataset that contains GGA and GGA+U static and relaxation calculations.
- Train-Val-Test splitting with mp-id: 0.9 - 0.5 - 0.5
- Train set size: 1419861
- Validation set size: 79719
- Test set size: 79182

# Performance metrics
## Training and validation errors

| partition  | Energy (meV/atom) | Force (meV/A) | stress (GPa) | magmom (muB) |
| ---------- | ----------------- | ------------- | ------------ | ------------ |
| Train      | 26.45             | 49            | 0.173        | 0.036        |
| Validation | 30.31             | 70            | 0.297        | 0.037        |
| Test       | 30.80             | 66            | 0.296        | 0.038        |


# References

```txt
Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling.
Nat. Mach. Intell. 1–11 (2023) doi:10.1038/s42256-023-00716-3.
  ```

####  Date: 2023.12.1
####  Author: Bowen Deng
