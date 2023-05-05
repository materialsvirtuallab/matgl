[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matgl)](https://github.com/materialsvirtuallab/matgl/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/matgl/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Testing/badge.svg)
[![Downloads](https://pepy.tech/badge/matgl)](https://pepy.tech/project/matgl)

# Table of Contents
* [Introduction](#introduction)
* [Status](#status)
* [Architectures](#architectures)
* [Installation](#installation)
* [Usage](#usage)
* [Documentation](#documentation)
* [References](#references)

<a name="introduction"></a>
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

<a name="status"></a>
# Status

- Apr 26 2023: Pre-trained MEGNet models now available for formation energies and band gaps!
- Feb 16 2023: Both initial implementations of M3GNet and MEGNet architectures have been completed. Expect bugs!

<a name="architectures"></a>
# Architectures

## MEGNet

<img src="https://github.com/materialsvirtuallab/matgl/blob/main/assets/MEGNet.png?raw=true"  width="50%">

The MatErials Graph Network (MEGNet) is an implementation of DeepMind's graph networks for universal machine
learning in materials science. We have demonstrated its success in achieving very low prediction errors in a broad
array of properties in both molecules and crystals (see "Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals"). New releases have included our recent work on multi-fidelity materials property modeling
(See "Learning properties of ordered and disordered materials from multi-fidelity data").

Briefly, Figure 1 shows the sequential update steps of the graph network, whereby bonds, atoms, and global state
attributes are updated using information from each other, generating an output graph.

## M3GNet

<img src="https://github.com/materialsvirtuallab/matgl/blob/main/assets/M3GNet.png?raw=true"  width="50%">

[M3GNet](https://www.nature.com/articles/s43588-022-00349-3) is a new materials graph neural network architecture that
incorporates 3-body interactions in MEGNet. An additional difference is the addition of the coordinates for atoms and
the 3×3 lattice matrix in crystals, which are necessary for obtaining tensorial quantities such as forces and
stresses via auto-differentiation.

As a framework, M3GNet has diverse applications, including:

- **Interatomic potential development.** With the same training data, M3GNet performs similarly to state-of-the-art
  machine learning interatomic potentials (ML-IAPs). However, a key feature of a graph representation is its
  flexibility to scale to diverse chemical spaces. One of the key accomplishments of M3GNet is the development of a
  *universal IAP* that can work across the entire periodic table of the elements by training on relaxations performed
  in the [Materials Project](http://materialsproject.org).
- **Surrogate models for property predictions.** Like the previous MEGNet architecture, M3GNet can be used to develop
  surrogate models for property predictions, achieving in many cases accuracies that better or similar to other
  state-of-the-art ML models.

For detailed performance benchmarks, please refer to the publication in the [References](#references) section.

<a name="installation"></a>
# Installation

Matgl can be installed via pip for the latest stable version:

```bash
pip install matgl
```

For the latest dev version, please clone this repo and install using:

```bash
python setup.py -e .
```


<a name="usage"></a>
# Usage

The pre-trained MEGNet models for the Materials Project formation energy and multi-fidelity band gap are now available.
The following is an example of a prediction of the formation energy for CsCl.

```python
from pymatgen.core import Structure, Lattice
from matgl.models._megnet import MEGNet

# load the pre-trained MEGNet model for formation energy model.
model = MEGNet.load("MEGNet-MP-2018.6.1-Eform")
# This is the structure obtained from the Materials Project.
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.14), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform.numpy()):5f} eV/atom.")
```

A full example is in [here](examples/Using%20MEGNet%20Pre-trained%20Models%20for%20Property%20Predictions.ipynb).


<a name="documentation"></a>
# Additional information
- [Documentation Page](http://materialsvirtuallab.github.io/matgl)
- [API documentation](https://materialsvirtuallab.github.io/matgl/modules.html)

<a name="references"></a>
# References

Please cite the following works:

- MEGNet
    ```txt
    Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
    Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.
    ```
- Multi-fidelity MEGNet
    ```txt
    Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. Learning Properties of Ordered and Disordered Materials from
    Multi-Fidelity Data. Nature Computational Science 2021, 1, 46–53. https://doi.org/10.1038/s43588-020-00002-x.
    ```
- M3GNet
    ```txt
    Chen, C., Ong, S.P. A universal graph deep learning interatomic potential for the periodic table. Nat Comput Sci,
    2, 718–728 (2022). https://doi.org/10.1038/s43588-022-00349-3.
    ```

# Acknowledgements

This work was primarily supported by the Materials Project, funded by the U.S. Department of Energy, Office of Science,
Office of Basic Energy Sciences, Materials Sciences and Engineering Division under contract no.
DE-AC02-05-CH11231: Materials Project program KC23MP. This work used the Expanse supercomputing cluster at the Extreme
Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number
ACI-1548562.
