[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matgl)](https://github.
com/materialsvirtuallab/matgl/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/matgl/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Testing/badge.svg)
[![Downloads](https://pepy.tech/badge/matgl)](https://pepy.tech/project/matgl)

# Introduction

MatGL (Materials Graph Library) is a graph deep learning library for materials. Mathematical graphs are a natural
representation for a collection of atoms (e.g., molecules or crystals). Graph deep learning models have been shown
to consistently deliver exceptional performance as surrogate models for the prediction of materials properties.

In this repository, we have reimplemented the [MatErials 3-body Graph Network (m3gnet)](https://github.com/materialsvirtuallab/m3gnet)
and its predecessor, [MEGNet](https://github.com/materialsvirtuallab/megnet) using the [Deep Graph Library (DGL)](https://www.dgl.ai).
The goal is to improve the usability, extensibility and scalability of these models. The original M3GNet and MEGNet were
implemented in TensorFlow.

This effort is a collaboration between the [Materials Virtual Lab](http://materialsvirtuallab.org) and Intel Labs
(Santiago Miret, Marcel Nassar, Carmelo Gonzales).

# Status

Feb 16 2023: Both initial implementations of M3GNet and MEGNet architectures have been completed. Expect bugs!

## M3GNet

<img src="https://github.com/materialsvirtuallab/matgl/blob/main/assets/M3GNet_schematic.png"  width="50%">

[M3GNet](https://www.nature.com/articles/s43588-022-00349-3) is a new materials graph neural network architecture that
incorporates 3-body interactions. A key difference with prior materials graph implementations such as
[MEGNet](https://github.com/materialsvirtuallab/megnet) is the addition of the coordinates for atoms and the 3×3
lattice matrix in crystals, which are necessary for obtaining tensorial quantities such as forces and stresses via
auto-differentiation.

As a framework, M3GNet has diverse applications, including:

- **Interatomic potential development.** With the same training data, M3GNet performs similarly to state-of-the-art
  machine learning interatomic potentials (ML-IAPs). However, a key feature of a graph representation is its
  flexibility to scale to diverse chemical spaces. One of the key accomplishments of M3GNet is the development of a
  *universal IAP* that can work across the entire periodic table of the elements by training on relaxations performed
  in the [Materials Project](http://materialsproject.org).
- **Surrogate models for property predictions.** Like the previous MEGNet architecture, M3GNet can be used to develop
  surrogate models for property predictions, achieving in many cases accuracies that better or similar to other
  state-of-the-art ML models.

For detailed performance benchmarks, please refer to the publication in the [References](#references) section. The
API documentation is available via the [Github Page](http://materialsvirtuallab.github.io/matgl).

# References

Please cite the following works:

- M3GNet
    ```txt
    Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
    2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.
    ```
- MEGNET
    ```txt
    Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
    Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.
    ```

# Acknowledgements

This work was primarily supported by the Materials Project, funded by the U.S. Department of Energy, Office of Science,
Office of Basic Energy Sciences, Materials Sciences and Engineering Division under contract no.
DE-AC02-05-CH11231: Materials Project program KC23MP. This work used the Expanse supercomputing cluster at the Extreme
Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number
ACI-1548562.
