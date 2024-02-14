[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/matgl)](https://github.com/materialsvirtuallab/matgl/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/matgl/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/matgl/actions/workflows/testing.yml/badge.svg)](https://github.com/materialsvirtuallab/matgl/actions/workflows/testing.yml)
[![Downloads](https://static.pepy.tech/badge/matgl)](https://pepy.tech/project/matgl)
[![codecov](https://codecov.io/gh/materialsvirtuallab/matgl/branch/main/graph/badge.svg?token=3V3O79GODQ)](https://codecov.io/gh/materialsvirtuallab/matgl)
[![PyPI](https://img.shields.io/pypi/v/matgl?logo=pypi&logoColor=white)](https://pypi.org/project/matgl?logo=pypi&logoColor=white)

# Materials Graph Library <img src="https://github.com/materialsvirtuallab/matgl/blob/main/assets/MatGL.png?raw=true" alt="matgl" width="30%" style="float: right">

## Official Documentation

<https://matgl.ai>

## Introduction

MatGL (Materials Graph Library) is a graph deep learning library for materials science. Mathematical graphs are a
natural representation for a collection of atoms. Graph deep learning models have been shown to consistently deliver
exceptional performance as surrogate models for the prediction of materials properties.

MatGL is built on the [Deep Graph Library (DGL)][dgl] and PyTorch, with suitable adaptations for materials-specific
applications. The goal is for MatGL to serve as an extensible platform to develop and share materials graph deep
learning models, including the [MatErials 3-body Graph Network (M3GNet)][m3gnet] and its predecessor, [MEGNet].

This effort is a collaboration between the [Materials Virtual Lab][mavrl] and Intel Labs (Santiago Miret, Marcel
Nassar, Carmelo Gonzales).

## Status

Major milestones are summarized below. Please refer to the [changelog] for details.

- v1.0.0 (Feb 14 2024): Implementation of [TensorNet] and [SO3Net].
- v0.5.1 (Jun 9 2023): Model versioning implemented.
- v0.5.0 (Jun 8 2023): Simplified saving and loading of models. Now models can be loaded with one line of code!
- v0.4.0 (Jun 7 2023): Near feature parity with original TF implementations. Re-trained M3Gnet universal potential now
  available.
- v0.1.0 (Feb 16 2023): Initial implementations of M3GNet and MEGNet architectures have been completed. Expect
  bugs!

## Current Architectures

Here, we summarize the currently implemented architectures in MatGL. It should be stressed that this is by no means
an exhaustive list, and we expect new architectures to be added by the core MatGL team as well as other contributors
in future.

<div style="float: left; padding: 10px; width: 300px">
<img src="https://github.com/materialsvirtuallab/matgl/blob/main/assets/MxGNet.png?raw=true" alt="m3gnet_schematic">
<p>Figure: Schematic of M3GNet/MEGNet</p>
</div>

### M3GNet

[Materials 3-body Graph Network (M3GNet)][m3gnet] is a new materials graph neural network architecture that
incorporates 3-body interactions in MEGNet. An additional difference is the addition of the coordinates for atoms and
the 3×3 lattice matrix in crystals, which are necessary for obtaining tensorial quantities such as forces and
stresses via auto-differentiation. As a framework, M3GNet has diverse applications, including:

- **Interatomic potential development.** With the same training data, M3GNet performs similarly to state-of-the-art
  machine learning interatomic potentials (MLIPs). However, a key feature of a graph representation is its
  flexibility to scale to diverse chemical spaces. One of the key accomplishments of M3GNet is the development of a
  [*universal IP*][m3gnet] that can work across the entire periodic table of the elements by training on relaxations
  performed in the [Materials Project][mp].
- **Surrogate models for property predictions.** Like the previous MEGNet architecture, M3GNet can be used to develop
  surrogate models for property predictions, achieving in many cases accuracies that are better or similar to other
  state-of-the-art ML models.

For detailed performance benchmarks, please refer to the publications in the [References](#references) section.

MatGL reimplemennts M3GNet using DGL and Pytorch. Compared to the original Tensorflow implementation, some key
improvements over the TF implementations are:

- A more intuitive API and class structure based on DGL.
- Multi-GPU support via PyTorch Lightning.

### MEGNet

[MatErials Graph Network (MEGNet)][megnet] is an implementation of DeepMind's [graph networks][graphnetwork] for
machine learning in materials science. We have demonstrated its success in achieving low prediction errors in a broad
array of properties in both [molecules and crystals][megnet]. New releases have included our recent work on
[multi-fidelity materials property modeling][mfimegnet]. Figure 1 shows the sequential update steps of the graph
network, whereby bonds, atoms, and global state attributes are updated using information from each other, generating an
output graph.

### Other models

We have implemented other models in matgl as well. A non-exhaustive list is given below.

- [TensorNet], an O(3)-equivariant message-passing neural network architecture that
  leverages Cartesian tensor representations.
- [SO3Net],  a minimalist SO(3)-equivariant neural network.

## Installation

Matgl can be installed via pip for the latest stable version:

```bash
pip install matgl
```

For the latest dev version, please clone this repo and install using:

```bash
pip install -e .
```

### CUDA (GPU) installation

If you intend to use CUDA (GPU) to speed up training, it is important to install the appropriate versions of PyTorch
and DGL. The basic instructions are given below, but it is recommended that you consult the
[PyTorch docs](https://pytorch.org/get-started/locally/) and [DGL docs](https://www.dgl.ai/pages/start.html) if you
run into any problems.

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

## Usage

Pre-trained M3GNet universal potential and MEGNet models for the Materials Project formation energy and
multi-fidelity band gap are now available.

### Command line (from v0.6.2)

A CLI tool now provides the capability to perform quick relaxations or predictions using pre-trained models, as well
as other simple administrative tasks (e.g., clearing the cache). Some simple examples:

1. To perform a relaxation,

    ```bash
    mgl relax --infile Li2O.cif --outfile Li2O_relax.cif
    ```

2. To use one of the pre-trained property models,

    ```bash
    mgl predict --model M3GNet-MP-2018.6.1-Eform --infile Li2O.cif
    ```

3. To clear the cache,

    ```bash
    mgl clear
    ```

For a full range of options, use `mgl -h`.

### Code

Users who just want to use the models out of the box should use the newly implemented `matgl.load_model` convenience
method. The following is an example of a prediction of the formation energy for CsCl.

```python
from pymatgen.core import Lattice, Structure
import matgl

model = matgl.load_model("MEGNet-MP-2018.6.1-Eform")

# This is the structure obtained from the Materials Project.
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform.numpy()):.3f} eV/atom.")
```

To obtain a listing of available pre-trained models,

```python
import matgl
print(matgl.get_available_pretrained_models())
```

## Pytorch Hub

The pre-trained models are also available on Pytorch hub. To use these models, simply install matgl and use the
following commands:

```python
import torch

# To obtain a listing of models
torch.hub.list("materialsvirtuallab/matgl", force_reload=True)

# To load a model
model = torch.hub.load("materialsvirtuallab/matgl", 'm3gnet_universal_potential')
```

## Tutorials

We wrote [tutorials] on how to use MatGL. These were generated from [Jupyter notebooks]
[jupyternb], which can be directly run on [Google Colab].

## Resources

- [API docs][apidocs] for all classes and methods.
- [Developer Guide](developer.md) outlines the key design elements of `matgl`, especially for developers wishing to
  train and contribute matgl models.
- AdvancedSoft has implemented a [LAMMPS interface](https://github.com/advancesoftcorp/lammps/tree/based-on-lammps_2Jun2022/src/ML-M3GNET)
  to both the TF and MatGL version of M3Gnet.

## References

A MatGL publication is currently being written. For now, pls refer to the CITATION.cff file for the citation
information. If you are using any of the pretrained models, please cite the relevant works below:

> **MEGNet**
>
> Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. *Graph Networks as a Universal Machine Learning Framework for
> Molecules and Crystals.* Chem. Mater. 2019, 31 (9), 3564–3572. DOI: [10.1021/acs.chemmater.9b01294][megnet].

> **Multi-fidelity MEGNet**
>
> Chen, C.; Zuo, Y.; Ye, W.; Li, X.; Ong, S. P. *Learning Properties of Ordered and Disordered Materials from
> Multi-Fidelity Data.* Nature Computational Science, 2021, 1, 46–53. DOI: [10.1038/s43588-020-00002-x][mfimegnet].

> **M3GNet**
>
> Chen, C., Ong, S.P. *A universal graph deep learning interatomic potential for the periodic table.* Nature
> Computational Science, 2023, 2, 718–728. DOI: [10.1038/s43588-022-00349-3][m3gnet].

## FAQs

1. **The `M3GNet-MP-2021.2.8-PES` differs from the original TensorFlow (TF) implementation!**

   *Answer:* `M3GNet-MP-2021.2.8-PES` is a refitted model with some data improvements and minor architectural changes.
   Porting over the weights from the TF version to DGL/PyTorch is non-trivial. We have performed reasonable benchmarking
   to ensure that the new implementation reproduces the broad error characteristics of the original TF implementation
   (see [examples][jupyternb]). However, it is not expected to reproduce the TF version exactly. This refitted model
   serves as a baseline for future model improvements. We do not believe there is value in expending the resources
   to reproduce the TF version exactly.

2. **I am getting errors with `matgl.load_model()`!**

   *Answer:* The most likely reason is that you have a cached older version of the model. We often refactor models to
   ensure the best implementation. This can usually be solved by updating your `matgl` to the latest version
   and clearing your cache using the following command `mgl clear`. On the next run, the latest model will be
   downloaded. With effect from v0.5.2, we have implemented a model versioning scheme that will detect code vs model
   version conflicts and alert the user of such problems.

3. **What pre-trained models should I be using?**

   *Answer:* There is no one definitive answer. In general, the newer the architecture and dataset, the more likely
   the model performs better. However, it should also be noted that a model operating on a more diverse dataset may
   compromise on  performance on a specific system. The best way is to look at the READMEs included with each model
   and do some tests on the systems you are interested in.

4. **How do I contribute to matgl?**

   *Answer:* For code contributions, please fork and submit pull requests. You should read the
   [developer guide](developer.md) to understand the general design guidelines. We welcome pre-trained model
   contributions as well, which should also be submitted via PRs. Please follow the folder structure of the
   pretrained models. In particular, we expect all models to come with a `README.md` and notebook
   documenting its use and its key performance metrics. Also, we expect contributions to be on new properties
   or systems or to significantly outperform the existing models. We will develop an alternative means for model
   sharing in the future.

5. **None of your models do what I need. Where can I get help?**

   *Answer:* Please contact [Prof Ong][ongemail] with a brief description of your needs. For simple problems, we are
   glad to advise and point you in the right direction. For more complicated problems, we are always open to
   academic collaborations or projects. We also offer [consulting services][mqm] for companies with unique needs,
   including but not limited to custom data generation, model development and materials design.

## Acknowledgments

This work was primarily supported by the [Materials Project][mp], funded by the U.S. Department of Energy, Office of
Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division under contract no.
DE-AC02-05-CH11231: Materials Project program KC23MP. This work used the Expanse supercomputing cluster at the Extreme
Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number
ACI-1548562.

[m3gnetrepo]: https://github.com/materialsvirtuallab/m3gnet "M3GNet repo"
[megnetrepo]: https://github.com/materialsvirtuallab/megnet "MEGNet repo"
[dgl]: https://www.dgl.ai "DGL website"
[mavrl]: http://materialsvirtuallab.org "MAVRL website"
[changelog]: https://matgl.ai/changes "Changelog"
[graphnetwork]: https://arxiv.org/abs/1806.01261 "Deepmind's paper"
[megnet]: https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294 "MEGNet paper"
[mfimegnet]: https://nature.com/articles/s43588-020-00002-x "mfi MEGNet paper"
[m3gnet]: https://nature.com/articles/s43588-022-00349-3 "M3GNet paper"
[mp]: http://materialsproject.org "Materials Project"
[apidocs]: https://matgl.ai/matgl.html "MatGL API docs"
[doc]: https://matgl.ai "MatGL Documentation"
[google colab]: https://colab.research.google.com/ "Google Colab"
[jupyternb]: https://github.com/materialsvirtuallab/matgl/tree/main/examples
[ongemail]: mailto:ongsp@ucsd.edu "Email"
[mqm]: https://materialsqm.com "MaterialsQM"
[tutorials]: https://matgl.ai/tutorials "Tutorials"
[tensornet]: https://arxiv.org/abs/2306.06482 "TensorNet"
[so3net]: https://pubs.aip.org/aip/jcp/article-abstract/158/14/144801/2877924/SchNetPack-2-0-A-neural-network-toolbox-for "SO3Net"
