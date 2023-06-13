---
layout: default
title: Home
nav_exclude: true
---

# matgl.models package

Package containing model implementations.

## Submodules

## matgl.models._m3gnet module

Implementation of Materials 3-body Graph Network (M3GNet) model. The main improvement over MEGNet is the addition of
many-body interactios terms, which improves efficiency of representation of local interactions for applications such as
interatomic potentials. For more details on M3GNet, please refer to:

```default
```
Chen, C., Ong, S.P. _A universal graph deep learning interatomic potential for the periodic table._ Nature
Computational Science, 2023, 2, 718–728. DOI: 10.1038/s43588-022-00349-3.
```
```


### _class_ matgl.models._m3gnet.M3GNet(element_types: tuple[str], dim_node_embedding: int = 64, dim_edge_embedding: int = 64, dim_state_embedding: int | None = None, dim_state_types: int | None = None, dim_state_feats: int | None = None, max_n: int = 3, max_l: int = 3, nblocks: int = 3, rbf_type='SphericalBessel', is_intensive: bool = True, readout_type: str = 'weighted_atom', task_type: str = 'regression', cutoff: float = 5.0, threebody_cutoff: float = 4.0, units: int = 64, ntargets: int = 1, use_smooth: bool = False, use_phi: bool = False, niters_set2set: int = 3, nlayers_set2set: int = 3, field: str = 'node_feat', include_state: bool = False, activation_type: str = 'swish', \*\*kwargs)
Bases: `Module`, [`IOMixIn`](matgl.utils.md#matgl.utils.io.IOMixIn)

The main M3GNet model.


#### forward(g: dgl.DGLGraph, state_attr: torch.tensor | None = None, l_g: dgl.DGLGraph | None = None)
Performs message passing and updates node representations.

Args:

    g : DGLGraph for a batch of graphs.

state_attr: State attrs for a batch of graphs.
l_g : DGLGraph for a batch of line graphs.

Returns:

    output: Output property for a batch of graphs


#### training(_: boo_ )
## matgl.models._megnet module

Implementation of MatErials Graph Network (MEGNet) model. Graph networks are a new machine learning (ML) paradigm that
supports both relational reasoning and combinatorial generalization. For more details on MEGNet, please refer to:

```default
```
Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. _Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals._ Chem. Mater. 2019, 31 (9), 3564-3572. DOI: 10.1021/acs.chemmater.9b01294.
```
```


### _class_ matgl.models._megnet.MEGNet(dim_node_embedding: int = 16, dim_edge_embedding: int = 100, dim_state_embedding: int = 2, ntypes_state: int | None = None, nblocks: int = 3, hidden_layer_sizes_input: tuple[int, ...] = (64, 32), hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32), hidden_layer_sizes_output: tuple[int, ...] = (32, 16), nlayers_set2set: int = 1, niters_set2set: int = 2, activation_type: str = 'softplus2', is_classification: bool = False, include_state: bool = True, dropout: float | None = None, graph_transformations: list | None = None, element_types: tuple[str, ...] = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'), bond_expansion: [BondExpansion](matgl.layers.md#matgl.layers._bond.BondExpansion) | None = None, cutoff: float = 4.0, gauss_width: float = 0.5, \*\*kwargs)
Bases: `Module`, [`IOMixIn`](matgl.utils.md#matgl.utils.io.IOMixIn)

DGL implementation of MEGNet.


#### forward(graph: DGLGraph, edge_feat: Tensor, node_feat: Tensor, state_feat: Tensor)
Forward pass of MEGnet. Executes all blocks.

Args:

    graph: Input graph
    edge_feat: Edge features
    node_feat: Node features
    state_feat: State features.

Returns:

    Prediction


#### predict_structure(structure, state_feats: torch.tensor | None = None, graph_converter: [GraphConverter](matgl.graph.md#matgl.graph.converters.GraphConverter) | None = None)
Convenience method to directly predict property from structure.

Args:

    structure: An input crystal/molecule.
    state_feats (torch.tensor): Graph attributes
    graph_converter: Object that implements a get_graph_from_structure.

Returns:

    output (torch.tensor): output property


#### training(_: boo_ )
## matgl.models._wrappers module

Implementations of pseudomodels that wraps other models.


### _class_ matgl.models._wrappers.TransformedTargetModel(model: Module, target_transformer: [Transformer](matgl.data.md#matgl.data.transformer.Transformer))
Bases: `Module`, [`IOMixIn`](matgl.utils.md#matgl.utils.io.IOMixIn)

A model where the target is first transformed prior to training and the reverse transformation is performed for
predictions. This is modelled after scikit-learn’s TransformedTargetRegressor. It should be noted that this model
is almost never used for training since the general idea is to use the transformed target for loss computation.
Instead, it is created after a model has been fitted for serialization for end user to call the model to perform
predictions without having to worry about what target transformations have been performed.


#### forward(\*args, \*\*kwargs)
Args:

    

    ```
    *
    ```

    args: Passthrough to parent model.forward method.


    ```
    **
    ```

    kwargs: Passthrough to parent model.forward method.

Returns:

    Inverse transformed output.


#### predict_structure(\*args, \*\*kwargs)
Pass through to parent model.predict_structure with inverse transform.

Args:

    

    ```
    *
    ```

    args: Pass-through to self.model.predict_structure.


    ```
    **
    ```

    kwargs: Pass-through to self.model.predict_structure.

Returns:

    Transformed answer.


#### training(_: boo_ )
