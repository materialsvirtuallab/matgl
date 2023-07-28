# Description

This model is a M3GNet universal potential for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials. This is essentially a retrained version of the M3GNet universal potential originally
implemented in tensorflow. It should be noted that the structures with high forces (> 100 eV/A) and no bond are removed
in the dataset.

# Training dataset

MP-2021.2.8: Materials Project structure relaxations as of 2021.2.8.
- Training set size: 167237
- Validation set size: 18584

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 19.977 meV/atom, 0.063 eV/A, 0.272 GPa
- Validation: 23.713 meV/atom, 0.070 eV/A, 0.394 GPa

## Cubic crystals benchmark

This is the same benchmark used for the TF implementation. The error characteristics are consistent between this
new implementation and the old version.

| Material    | Crystal structure   |   a (Å) |   MP a (Å) |   Predicted a (Å) | % error vs Expt   | % error vs MP   |
|:------------|:--------------------|--------:|-----------:|------------------:|:------------------|:----------------|
| Nb          | BCC                 | 3.3008  |    3.31763 |           3.31778 | 0.515%            | 0.005%          |
| Mo          | BCC                 | 3.142   |    3.16762 |           3.16674 | 0.787%            | -0.028%         |
| NbN         | Halite              | 4.392   |    4.45247 |           4.45441 | 1.421%            | 0.044%          |
| Au          | FCC                 | 4.065   |    4.17129 |           4.16861 | 2.549%            | -0.064%         |
| V           | BCC                 | 3.0399  |    2.9824  |           2.98519 | -1.800%           | 0.093%          |
| ZnO         | Halite (FCC)        | 4.58    |    4.33888 |           4.33463 | -5.357%           | -0.098%         |
| GaAs        | Zinc blende (FCC)   | 5.653   |    5.75018 |           5.7445  | 1.619%            | -0.099%         |
| ZrC0.97     | Halite              | 4.698   |    4.71287 |           4.71764 | 0.418%            | 0.101%          |
| TiN         | Halite              | 4.249   |    4.24125 |           4.24602 | -0.070%           | 0.113%          |
| ScN         | Halite              | 4.52    |    4.51094 |           4.51692 | -0.068%           | 0.133%          |
| LiF         | Halite              | 4.03    |    4.08343 |           4.08902 | 1.465%            | 0.137%          |
| TiC         | Halite              | 4.328   |    4.33144 |           4.33778 | 0.226%            | 0.146%          |
| Li          | BCC                 | 3.49    |    3.43931 |           3.44449 | -1.304%           | 0.150%          |
| TaC0.99     | Halite              | 4.456   |    4.4678  |           4.47471 | 0.420%            | 0.155%          |
| VN          | Halite              | 4.136   |    4.12493 |           4.13145 | -0.110%           | 0.158%          |
| Th          | FCC                 | 5.08    |    5.04725 |           5.05576 | -0.477%           | 0.169%          |
| Al          | FCC                 | 4.046   |    4.03893 |           4.04695 | 0.024%            | 0.199%          |
| Sr          | FCC                 | 6.08    |    6.06721 |           6.05488 | -0.413%           | -0.203%         |
| Eu          | BCC                 | 4.61    |    4.4876  |           4.49699 | -2.451%           | 0.209%          |
| HfC0.99     | Halite              | 4.64    |    4.63428 |           4.64439 | 0.095%            | 0.218%          |
| BP          | Zinc blende (FCC)   | 4.538   |    4.53214 |           4.54335 | 0.118%            | 0.247%          |
| W           | BCC                 | 3.155   |    3.17032 |           3.17823 | 0.736%            | 0.250%          |
| Si          | Diamond (FCC)       | 5.43102 |    5.4437  |           5.45841 | 0.504%            | 0.270%          |
| CaVO3       | Cubic perovskite    | 3.767   |    3.83041 |           3.84097 | 1.964%            | 0.276%          |
| SrVO3       | Cubic perovskite    | 3.838   |    3.90089 |           3.91185 | 1.924%            | 0.281%          |
| PbTe        | Halite (FCC)        | 6.462   |    6.54179 |           6.56034 | 1.522%            | 0.284%          |
| BN          | Zinc blende (FCC)   | 3.615   |    3.626   |           3.61513 | 0.004%            | -0.300%         |
| InSb        | Zinc blende (FCC)   | 6.479   |    6.63322 |           6.65313 | 2.688%            | 0.300%          |
| AlP         | Zinc blende (FCC)   | 5.451   |    5.47297 |           5.4894  | 0.704%            | 0.300%          |
| CsI         | Caesium chloride    | 4.567   |    4.66521 |           4.6793  | 2.459%            | 0.302%          |
| K           | BCC                 | 5.23    |    5.39512 |           5.37726 | 2.816%            | -0.331%         |
| VC0.97      | Halite              | 4.166   |    4.16195 |           4.14514 | -0.501%           | -0.404%         |
| Na          | BCC                 | 4.23    |    4.20805 |           4.22615 | -0.091%           | 0.430%          |
| Ir          | FCC                 | 3.84    |    3.85393 |           3.87263 | 0.850%            | 0.485%          |
| Ta          | BCC                 | 3.3058  |    3.30986 |           3.32666 | 0.631%            | 0.508%          |
| ZrN         | Halite              | 4.577   |    4.58853 |           4.61229 | 0.771%            | 0.518%          |
| HfN         | Halite              | 4.392   |    4.51172 |           4.53537 | 3.264%            | 0.524%          |
| Pd          | FCC                 | 3.859   |    3.9173  |           3.93826 | 2.054%            | 0.535%          |
| CdS         | Zinc blende (FCC)   | 5.832   |    5.88591 |           5.92222 | 1.547%            | 0.617%          |
| GaP         | Zinc blende (FCC)   | 5.4505  |    5.45162 |           5.48814 | 0.691%            | 0.670%          |
| AlSb        | Zinc blende (FCC)   | 6.1355  |    6.18504 |           6.22894 | 1.523%            | 0.710%          |
| Pb          | FCC                 | 4.92    |    4.98951 |           5.0257  | 2.148%            | 0.725%          |
| Ni          | FCC                 | 3.499   |    3.47515 |           3.50337 | 0.125%            | 0.812%          |
| SrTiO3      | Cubic perovskite    | 3.98805 |    3.9127  |           3.94484 | -1.084%           | 0.821%          |
| C (diamond) | Diamond (FCC)       | 3.567   |    3.56075 |           3.59053 | 0.660%            | 0.836%          |
| Pt          | FCC                 | 3.912   |    3.94315 |           3.97644 | 1.647%            | 0.844%          |
| Ce          | FCC                 | 5.16    |    4.67243 |           4.71224 | -8.678%           | 0.852%          |
| Fe          | BCC                 | 2.856   |    2.86304 |           2.83816 | -0.625%           | -0.869%         |
| AlAs        | Zinc blende (FCC)   | 5.6605  |    5.6758  |           5.7309  | 1.244%            | 0.971%          |
| InP         | Zinc blende (FCC)   | 5.869   |    5.90395 |           5.96173 | 1.580%            | 0.979%          |
| CdTe        | Zinc blende (FCC)   | 6.482   |    6.56423 |           6.63031 | 2.288%            | 1.007%          |
| LiCl        | Halite              | 5.14    |    5.08424 |           5.1376  | -0.047%           | 1.049%          |
| NaI         | Halite              | 6.47    |    6.43731 |           6.5061  | 0.558%            | 1.069%          |
| Rh          | FCC                 | 3.8     |    3.80597 |           3.84673 | 1.230%            | 1.071%          |
| Cu          | FCC                 | 3.597   |    3.57743 |           3.61689 | 0.553%            | 1.103%          |
| KI          | Halite              | 7.07    |    7.08487 |           7.16469 | 1.339%            | 1.127%          |
| Ca          | FCC                 | 5.58    |    5.57682 |           5.51357 | -1.190%           | -1.134%         |
| CdSe        | Zinc blende (FCC)   | 6.05    |    6.14054 |           6.21125 | 2.665%            | 1.151%          |
| CrN         | Halite              | 4.149   |    4.19086 |           4.14204 | -0.168%           | -1.165%         |
| ZnS         | Zinc blende (FCC)   | 5.42    |    5.38737 |           5.45396 | 0.626%            | 1.236%          |
| RbBr        | Halite              | 6.89    |    6.9119  |           7.00138 | 1.616%            | 1.295%          |
| LiI         | Halite              | 6.01    |    5.96835 |           6.05377 | 0.728%            | 1.431%          |
| Ac          | FCC                 | 5.31    |    5.69621 |           5.61351 | 5.716%            | -1.452%         |
| GaSb        | Zinc blende (FCC)   | 6.0959  |    6.13721 |           6.22862 | 2.177%            | 1.490%          |
| InAs        | Zinc blende (FCC)   | 6.0583  |    6.10712 |           6.19817 | 2.309%            | 1.491%          |
| RbCl        | Halite              | 6.59    |    6.61741 |           6.72117 | 1.990%            | 1.568%          |
| RbF         | Halite              | 5.65    |    5.63228 |           5.72259 | 1.285%            | 1.603%          |
| CsCl        | Caesium chloride    | 4.123   |    4.1437  |           4.21141 | 2.144%            | 1.634%          |
| Ag          | FCC                 | 4.079   |    4.10436 |           4.1722  | 2.285%            | 1.653%          |
| KBr         | Halite              | 6.6     |    6.58908 |           6.69863 | 1.494%            | 1.663%          |
| KCl         | Halite              | 6.29    |    6.28374 |           6.38911 | 1.576%            | 1.677%          |
| MgO         | Halite (FCC)        | 4.212   |    4.194   |           4.26457 | 1.248%            | 1.683%          |
| Ge          | Diamond (FCC)       | 5.658   |    5.67485 |           5.77178 | 2.011%            | 1.708%          |
| LiBr        | Halite              | 5.5     |    5.44538 |           5.54244 | 0.772%            | 1.782%          |
| Yb          | FCC                 | 5.49    |    5.38726 |           5.48394 | -0.110%           | 1.795%          |
| RbI         | Halite              | 7.35    |    7.38233 |           7.52581 | 2.392%            | 1.944%          |
| NaCl        | Halite              | 5.64    |    5.58813 |           5.69721 | 1.014%            | 1.952%          |
| CsF         | Halite              | 6.02    |    6.01228 |           6.1354  | 1.917%            | 2.048%          |
| KF          | Halite              | 5.34    |    5.30883 |           5.42535 | 1.598%            | 2.195%          |
| NaBr        | Halite              | 5.97    |    5.92308 |           6.06065 | 1.518%            | 2.323%          |
| NaF         | Halite              | 4.63    |    4.57179 |           4.71661 | 1.871%            | 3.168%          |
| Ne          | FCC                 | 4.43    |    4.19502 |           4.33852 | -2.065%           | 3.421%          |
| Cr          | BCC                 | 2.88    |    2.9689  |           2.8593  | -0.719%           | -3.692%         |
| Cs          | BCC                 | 6.05    |    6.25693 |           5.31818 | -12.096%          | -15.003%        |
| Ar          | FCC                 | 5.26    |    5.36316 |           4.1454  | -21.190%          | -22.706%        |

# References

```txt
Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.
```
