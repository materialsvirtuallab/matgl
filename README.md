# Introduction

Mathematical graphs are a natural representation for a collection of atoms (e.g., molecules or crystals). Graph deep
learning models have been shown to consistently deliver exceptional performance as surrogate models for the prediction
of materials properties.

This repository is a unified reimplementation of the [3-body MatErials Graph Network (m3gnet)](https://github.com/materialsvirtuallab/m3gnet)
and its predecessors, [MEGNet](https://github.com/materialsvirtuallab/megnet) using the [Deep Graph Library (DGL)](https://www.dgl.ai).
The goal is to improve the usability, extensibility and scalability of these models. The original M3GNet and MEGNet were
implemented in TensorFlow.

This effort is a collaboration between the [Materials Virtual Lab](http://materialsvirtuallab.org) and Intel Labs
(Santiago Miret, Marcel Nassar, Carmelo Gonzales).

# Status

At the present moment, only the simpler MEGNet architecture has been implemented. The implementation has been
extensively tested. It is reasonably robust and performs several times faster than the original TF implementation.

The plan is to complete implementation of M3GNet by end of 2022.

For users wishing to use the pre-trained models, we recommend you check out the official [MEGNet](https://github.com/materialsvirtuallab/megnet)
and [M3GNet](https://github.com/materialsvirtuallab/m3gnet) implementations. For new users wishing to train new MEGNet
models, we welcome you to use this DGL implementation. Any contributions, e.g., code improvements or issue reports, are
very welcome!

# References

Please cite the following works:

- MEGNET
    ```txt
    Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. Graph Networks as a Universal Machine Learning Framework for
    Molecules and Crystals. Chem. Mater. 2019, 31 (9), 3564–3572. https://doi.org/10.1021/acs.chemmater.9b01294.
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
