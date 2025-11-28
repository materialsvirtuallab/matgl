# Description

This model is a M3GNet cumtomized potential for Na-B-H system. It has broad applications in the
dynamic simulations of materials.

# Training dataset

The training dataset includes all Na–B–H structures from the Materials Project and ICSD, along with relaxation trajectories and MD-generated configurations across different temperatures and strains. Representative snapshots were selected using DIRECT sampling, and r2SCAN static calculations were used to obtain the corresponding energies, forces, and stresses.
- Training set size: 4476
- Test set size: 560

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 20.96 meV/atom, 0.15 eV/A, 0.25 GPa
- Test: 30.20 meV/atom, 0.16 eV/A, 0.29 GPa

# References

```txt
Oh, J. A. S.; Yu, Z.; Huang, C.-J.; Ridley, P.; Liu, A.; Zhang, T.; Hwang, B. J.; Griffith, K. J.; Ong, S. P.; Meng, Y. S. Metastable Sodium Closo-Hydridoborates for All-Solid-State Batteries with Thick Cathodes. Joule 2025, 9 (10), 102130. https://doi.org/10.1016/j.joule.2025.102130.
```
