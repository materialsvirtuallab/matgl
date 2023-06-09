# MatGL (Materials Graph Library)

[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matgl)](https://github.com/materialsvirtuallab/matgl/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/matgl/workflows/Testing%20-%20main/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Testing/badge.svg)
[![Downloads](https://pepy.tech/badge/matgl)](https://pepy.tech/project/matgl)

## Table of Contents

- [Introduction](#introduction)
- [Status](#status)
- [Architectures](#architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Docs](#docs)
- [References](#references)

## Introduction

MatGL (Materials Graph Library) is a graph deep learning library for materials science. Mathematical graphs are a
natural representation for a collection of atoms (e.g., molecules or crystals). Graph deep learning models have been
shown to consistently deliver exceptional performance as surrogate models for the prediction of materials properties.

In this repository, we have reimplemented the [MatErials 3-body Graph Network (m3gnet)](https://github.com/materialsvirtuallab/m3gnet)
and its predecessor, [MEGNet](https://github.com/materialsvirtuallab/megnet) using the [Deep Graph Library (DGL)](https://www.dgl.ai).
The goal is to improve the usability, extensibility and scalability of these models. The original M3GNet and MEGNet were
implemented in TensorFlow (TF). Here are some key improvements over the TF implementations:
- A more intuitive API and class structure based on the Deep Graph Library.
- Multi-GPU support via PyTorch Lightning. A training utility module has been developed.

This effort is a collaboration between the [Materials Virtual Lab](http://materialsvirtuallab.org) and Intel Labs
(Santiago Miret, Marcel Nassar, Carmelo Gonzales).

## Status

Major milestones are summarized below. Full change log is provided [here](https://materialsvirtuallab.github.io/matgl/changes).
- v0.5.1 (Jun 9 2023): Model versioning implemented.
- v0.5.0 (Jun 8 2023): Simplified saving and loading of models. Now models can be loaded with one line of code!
- v0.4.0 (Jun 7 2023): Near feature parity with original TF implementations. Re-trained M3Gnet universal potential now
  available.
- v0.1.0 (Feb 16 2023): Initial implementations of M3GNet and MEGNet architectures have been completed. Expect
  bugs!

## Architectures

<img src="https://github.com/materialsvirtuallab/matgl/blob/main/assets/MxGNet.png?raw=true" alt="m3gnet_schematic" width="50%">

## MEGNet

The MatErials Graph Network (MEGNet) is an implementation of DeepMind's graph networks for universal machine
learning in materials science. We have demonstrated its success in achieving very low prediction errors in a broad
array of properties in both molecules and crystals (see "Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals"). New releases have included our recent work on multi-fidelity materials property modeling
(See "Learning properties of ordered and disordered materials from multi-fidelity data").

Briefly, Figure 1 shows the sequential update steps of the graph network, whereby bonds, atoms, and global state
attributes are updated using information from each other, generating an output graph.

## M3GNet

[M3GNet](https://www.nature.com/articles/s43588-022-00349-3) is a new materials graph neural network architecture that
incorporates 3-body interactions in MEGNet. An additional difference is the addition of the coordinates for atoms and
the 3×3 lattice matrix in crystals, which are necessary for obtaining tensorial quantities such as forces and
stresses via auto-differentiation.

As a framework, M3GNet has diverse applications, including:

- **Interatomic potential development.** With the same training data, M3GNet performs similarly to state-of-the-art
  machine learning interatomic potentials (MLIPs). However, a key feature of a graph representation is its
  flexibility to scale to diverse chemical spaces. One of the key accomplishments of M3GNet is the development of a
  *universal IP* that can work across the entire periodic table of the elements by training on relaxations performed
  in the [Materials Project](http://materialsproject.org).
- **Surrogate models for property predictions.** Like the previous MEGNet architecture, M3GNet can be used to develop
  surrogate models for property predictions, achieving in many cases accuracies that are better or similar to other
  state-of-the-art ML models.

For detailed performance benchmarks, please refer to the publications in the [References](#references) section.

## Installation

Matgl can be installed via pip for the latest stable version:

```bash
pip install matgl
```

For the latest dev version, please clone this repo and install using:

```bash
python setup.py -e .
```

## Usage

Pre-trained M3GNet universal potential and MEGNet models for the Materials Project formation energy and
multi-fidelity band gap are now available. Users who just want to use the models out of the box should use the newly
implemented convenience method:

```python
import matgl
model = matgl.load_model("<model_name>")
```

The following is an example of a prediction of the formation energy for CsCl.

```python
from pymatgen.core import Lattice, Structure
import matgl

model = matgl.load_model("MEGNet-MP-2018.6.1-Eform")

# This is the structure obtained from the Materials Project.
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform.numpy()):.3f} eV/atom.")
```

## Example notebooks

Primary usage documentation will be done via Jupyter notes, which are available [here](examples).

## Docs

<http://materialsvirtuallab.github.io/matgl>

## References

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

## Acknowledgments

This work was primarily supported by the Materials Project, funded by the U.S. Department of Energy, Office of Science,
Office of Basic Energy Sciences, Materials Sciences and Engineering Division under contract no.
DE-AC02-05-CH11231: Materials Project program KC23MP. This work used the Expanse supercomputing cluster at the Extreme
Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number
ACI-1548562.
