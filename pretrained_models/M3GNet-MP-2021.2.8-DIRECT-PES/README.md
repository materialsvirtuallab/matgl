# Description

This model is a M3GNet universal potential trained for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials. This is essentially a retrained version of the M3GNet universal potential originally
implemented in tensorflow. It should be noted that this is a preliminary version, which require further optimizations of
hyper-parameters for better accuracy. The accuracy is expected to be worse than the MP-2021.2.8 version since the structures
and corresponding energies, forces and stresses are more diverse. However, we expected that the extrapolation should be more reliable.

# Training dataset

MP-2021.2.8-DIRECT: Materials Project structure relaxations as of 2021.2.8 combined with DIRECT sampling.
- Training set size: 167192
- Validation set size: 9284

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 27.185 meV/atom, 0.056 eV/A, 0.246 GPa
- Validation: 37.254 meV/atom, 0.089 eV/A, 0.558 GPa

## Cubic crystals benchmark

This is the same benchmark used for the TF implementation. The error characteristics are consistent between this
new implementation and the old version. It should be noted that the large deviation of lattice constant in Li-BCC
is attributed to the poor initialization. If the initial lattice constant is set to less than 4.3 Å, it gets down
to 1 %


| Material    | Crystal structure   |   a (Å) |   MP a (Å) |   Predicted a (Å) | % error vs Expt   | % error vs MP   |
|:------------|:--------------------|--------:|-----------:|------------------:|:------------------|:----------------|
| ScN         | Halite              | 4.52    |    4.51094 |           4.51017 | -0.218%           | -0.017%         |
| Mo          | BCC                 | 3.142   |    3.16762 |           3.16587 | 0.760%            | -0.055%         |
| TaC0.99     | Halite              | 4.456   |    4.4678  |           4.47214 | 0.362%            | 0.097%          |
| Eu          | BCC                 | 4.61    |    4.4876  |           4.48287 | -2.758%           | -0.105%         |
| BN          | Zinc blende (FCC)   | 3.615   |    3.626   |           3.62992 | 0.413%            | 0.108%          |
| ZrC0.97     | Halite              | 4.698   |    4.71287 |           4.71856 | 0.438%            | 0.121%          |
| EuTiO3      | Cubic perovskite    | 7.81    |    3.90388 |           3.91029 | -49.932%          | 0.164%          |
| LiF         | Halite              | 4.03    |    4.08343 |           4.09049 | 1.501%            | 0.173%          |
| Ca          | FCC                 | 5.58    |    5.57682 |           5.58657 | 0.118%            | 0.175%          |
| TiN         | Halite              | 4.249   |    4.24125 |           4.23383 | -0.357%           | -0.175%         |
| V           | BCC                 | 3.0399  |    2.9824  |           2.98789 | -1.711%           | 0.184%          |
| BP          | Zinc blende (FCC)   | 4.538   |    4.53214 |           4.52317 | -0.327%           | -0.198%         |
| Au          | FCC                 | 4.065   |    4.17129 |           4.17972 | 2.822%            | 0.202%          |
| RbBr        | Halite              | 6.89    |    6.9119  |           6.92641 | 0.528%            | 0.210%          |
| KI          | Halite              | 7.07    |    7.08487 |           7.09979 | 0.421%            | 0.211%          |
| Sr          | FCC                 | 6.08    |    6.06721 |           6.05396 | -0.428%           | -0.218%         |
| KF          | Halite              | 5.34    |    5.30883 |           5.32066 | -0.362%           | 0.223%          |
| GaAs        | Zinc blende (FCC)   | 5.653   |    5.75018 |           5.76565 | 1.993%            | 0.269%          |
| CsI         | Caesium chloride    | 4.567   |    4.66521 |           4.67786 | 2.427%            | 0.271%          |
| CaVO3       | Cubic perovskite    | 3.767   |    3.83041 |           3.8409  | 1.962%            | 0.274%          |
| Nb          | BCC                 | 3.3008  |    3.31763 |           3.3268  | 0.788%            | 0.276%          |
| RbI         | Halite              | 7.35    |    7.38233 |           7.3619  | 0.162%            | -0.277%         |
| VC0.97      | Halite              | 4.166   |    4.16195 |           4.15025 | -0.378%           | -0.281%         |
| C (diamond) | Diamond (FCC)       | 3.567   |    3.56075 |           3.57077 | 0.106%            | 0.282%          |
| Th          | FCC                 | 5.08    |    5.04725 |           5.03259 | -0.933%           | -0.291%         |
| TiC         | Halite              | 4.328   |    4.33144 |           4.34674 | 0.433%            | 0.353%          |
| SrTiO3      | Cubic perovskite    | 3.98805 |    3.9127  |           3.92659 | -1.541%           | 0.355%          |
| InSb        | Zinc blende (FCC)   | 6.479   |    6.63322 |           6.65725 | 2.751%            | 0.362%          |
| NbN         | Halite              | 4.392   |    4.45247 |           4.4356  | 0.993%            | -0.379%         |
| W           | BCC                 | 3.155   |    3.17032 |           3.18331 | 0.897%            | 0.410%          |
| SrVO3       | Cubic perovskite    | 3.838   |    3.90089 |           3.91735 | 2.067%            | 0.422%          |
| Al          | FCC                 | 4.046   |    4.03893 |           4.02163 | -0.602%           | -0.428%         |
| HfC0.99     | Halite              | 4.64    |    4.63428 |           4.65743 | 0.376%            | 0.499%          |
| PbTe        | Halite (FCC)        | 6.462   |    6.54179 |           6.57573 | 1.760%            | 0.519%          |
| Si          | Diamond (FCC)       | 5.43102 |    5.4437  |           5.4728  | 0.769%            | 0.534%          |
| Pt          | FCC                 | 3.912   |    3.94315 |           3.96494 | 1.353%            | 0.553%          |
| Ta          | BCC                 | 3.3058  |    3.30986 |           3.32933 | 0.712%            | 0.588%          |
| RbF         | Halite              | 5.65    |    5.63228 |           5.66891 | 0.335%            | 0.650%          |
| Ir          | FCC                 | 3.84    |    3.85393 |           3.87954 | 1.030%            | 0.664%          |
| VN          | Halite              | 4.136   |    4.12493 |           4.09623 | -0.962%           | -0.696%         |
| Fe          | BCC                 | 2.856   |    2.86304 |           2.84213 | -0.486%           | -0.730%         |
| Pd          | FCC                 | 3.859   |    3.9173  |           3.95178 | 2.404%            | 0.880%          |
| Na          | BCC                 | 4.23    |    4.20805 |           4.1689  | -1.445%           | -0.931%         |
| AlSb        | Zinc blende (FCC)   | 6.1355  |    6.18504 |           6.24323 | 1.756%            | 0.941%          |
| MgO         | Halite (FCC)        | 4.212   |    4.194   |           4.23513 | 0.549%            | 0.981%          |
| InP         | Zinc blende (FCC)   | 5.869   |    5.90395 |           5.96218 | 1.588%            | 0.986%          |
| ZrN         | Halite              | 4.577   |    4.58853 |           4.63425 | 1.251%            | 0.996%          |
| Ce          | FCC                 | 5.16    |    4.67243 |           4.72024 | -8.522%           | 1.023%          |
| CdS         | Zinc blende (FCC)   | 5.832   |    5.88591 |           5.94639 | 1.961%            | 1.028%          |
| NaCl        | Halite              | 5.64    |    5.58813 |           5.64691 | 0.122%            | 1.052%          |
| Cu          | FCC                 | 3.597   |    3.57743 |           3.61593 | 0.526%            | 1.076%          |
| Yb          | FCC                 | 5.49    |    5.38726 |           5.44536 | -0.813%           | 1.079%          |
| Rh          | FCC                 | 3.8     |    3.80597 |           3.84798 | 1.263%            | 1.104%          |
| AlAs        | Zinc blende (FCC)   | 5.6605  |    5.6758  |           5.73893 | 1.386%            | 1.112%          |
| Ni          | FCC                 | 3.499   |    3.47515 |           3.51492 | 0.455%            | 1.145%          |
| ZnO         | Halite (FCC)        | 4.58    |    4.33888 |           4.39046 | -4.138%           | 1.189%          |
| KBr         | Halite              | 6.6     |    6.58908 |           6.67326 | 1.110%            | 1.278%          |
| CsF         | Halite              | 6.02    |    6.01228 |           6.0925  | 1.204%            | 1.334%          |
| GaP         | Zinc blende (FCC)   | 5.4505  |    5.45162 |           5.52573 | 1.380%            | 1.359%          |
| Ac          | FCC                 | 5.31    |    5.69621 |           5.61402 | 5.725%            | -1.443%         |
| Pb          | FCC                 | 4.92    |    4.98951 |           5.06252 | 2.897%            | 1.463%          |
| ZnS         | Zinc blende (FCC)   | 5.42    |    5.38737 |           5.46637 | 0.856%            | 1.466%          |
| CdSe        | Zinc blende (FCC)   | 6.05    |    6.14054 |           6.23124 | 2.996%            | 1.477%          |
| RbCl        | Halite              | 6.59    |    6.61741 |           6.71582 | 1.909%            | 1.487%          |
| Ge          | Diamond (FCC)       | 5.658   |    5.67485 |           5.75956 | 1.795%            | 1.493%          |
| AlP         | Zinc blende (FCC)   | 5.451   |    5.47297 |           5.38931 | -1.132%           | -1.529%         |
| InAs        | Zinc blende (FCC)   | 6.0583  |    6.10712 |           6.20406 | 2.406%            | 1.587%          |
| CrN         | Halite              | 4.149   |    4.19086 |           4.12346 | -0.616%           | -1.608%         |
| Ag          | FCC                 | 4.079   |    4.10436 |           4.17358 | 2.319%            | 1.687%          |
| CdTe        | Zinc blende (FCC)   | 6.482   |    6.56423 |           6.67554 | 2.986%            | 1.696%          |
| KTaO3       | Cubic perovskite    | 3.9885  |    3.99488 |           4.06825 | 1.999%            | 1.836%          |
| GaSb        | Zinc blende (FCC)   | 6.0959  |    6.13721 |           6.257   | 2.643%            | 1.952%          |
| K           | BCC                 | 5.23    |    5.39512 |           5.28761 | 1.102%            | -1.993%         |
| HfN         | Halite              | 4.392   |    4.51172 |           4.61078 | 4.981%            | 2.196%          |
| LiI         | Halite              | 6.01    |    5.96835 |           6.11641 | 1.771%            | 2.481%          |
| Ne          | FCC                 | 4.43    |    4.19502 |           4.30647 | -2.789%           | 2.657%          |
| NaBr        | Halite              | 5.97    |    5.92308 |           6.11725 | 2.466%            | 3.278%          |
| NaI         | Halite              | 6.47    |    6.43731 |           6.65961 | 2.931%            | 3.453%          |
| CsCl        | Caesium chloride    | 4.123   |    4.1437  |           4.28721 | 3.983%            | 3.463%          |
| KCl         | Halite              | 6.29    |    6.28374 |           6.50232 | 3.375%            | 3.478%          |
| Cr          | BCC                 | 2.88    |    2.9689  |           2.85108 | -1.004%           | -3.968%         |
| Ar          | FCC                 | 5.26    |    5.36316 |           5.65291 | 7.470%            | 5.403%          |
| NaF         | Halite              | 4.63    |    4.57179 |           4.83623 | 4.454%            | 5.784%          |
| Cs          | BCC                 | 6.05    |    6.25693 |           5.81752 | -3.843%           | -7.023%         |
| LiCl        | Halite              | 5.14    |    5.08424 |           4.05333 | -21.141%          | -20.277%        |
| LiBr        | Halite              | 5.5     |    5.44538 |           4.3005  | -21.809%          | -21.025%        |
| Li          | BCC                 | 3.49    |    3.43931 |           6.36292 | 82.319%           | 85.006%         |

# References

```txt
Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.
```
