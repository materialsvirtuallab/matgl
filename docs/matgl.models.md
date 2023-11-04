---
layout: default
title: matgl.models.md
nav_exclude: true
---

# matgl.models package

Package containing model implementations.

## matgl.models._m3gnet module

Implementation of Materials 3-body Graph Network (M3GNet) model.

The main improvement over MEGNet is the addition of many-body interactios terms, which improves efficiency of
representation of local interactions for applications such as interatomic potentials. For more details on M3GNet,
please refer to:

```default
Chen, C., Ong, S.P. _A universal graph deep learning interatomic potential for the periodic table._ Nature
Computational Science, 2023, 2, 718-728. DOI: 10.1038/s43588-022-00349-3.
```

### *class* matgl.models._m3gnet.M3GNet(element_types: tuple[str], dim_node_embedding: int = 64, dim_edge_embedding: int = 64, dim_state_embedding: int | None = None, ntypes_state: int | None = None, dim_state_feats: int | None = None, max_n: int = 3, max_l: int = 3, nblocks: int = 3, rbf_type=’SphericalBessel’, is_intensive: bool = True, readout_type: str = ‘weighted_atom’, task_type: str = ‘regression’, cutoff: float = 5.0, threebody_cutoff: float = 4.0, units: int = 64, ntargets: int = 1, use_smooth: bool = False, use_phi: bool = False, niters_set2set: int = 3, nlayers_set2set: int = 3, field: Literal[‘node_feat’, ‘edge_feat’] = ‘node_feat’, include_state: bool = False, activation_type: str = ‘swish’, \*\*kwargs)

Bases: `Module`, `IOMixIn`

The main M3GNet model.

* **Parameters:**
  * **element_types** (*tuple*) – list of elements appearing in the dataset
  * **dim_node_embedding** (*int*) – number of embedded atomic features
  * **dim_edge_embedding** (*int*) – number of edge features
  * **dim_state_embedding** (*int*) – number of hidden neurons in state embedding
  * **dim_state_feats** (*int*) – number of state features after linear layer
  * **ntypes_state** (*int*) – number of state labels
  * **max_n** (*int*) – number of radial basis expansion
  * **max_l** (*int*) – number of angular expansion
  * **nblocks** (*int*) – number of convolution blocks
  * **rbf_type** (*str*) – radial basis function. choose from ‘Gaussian’ or ‘SphericalBessel’
  * **is_intensive** (*bool*) – whether the prediction is intensive
  * **readout_type** (*str*) – the readout function type. choose from set2set,
  * **reduce_atom** (*weighted_atom and*) –
  * **weighted_atom** (*default to*) –
  * **task_type** (*str*) – classification or regression, default to
  * **regression** –
  * **cutoff** (*float*) – cutoff radius of the graph
  * **threebody_cutoff** (*float*) – cutoff radius for 3 body interaction
  * **units** (*int*) – number of neurons in each MLP layer
  * **ntargets** (*int*) – number of target properties
  * **use_smooth** (*bool*) – whether using smooth Bessel functions
  * **use_phi** (*bool*) – whether using phi angle
  * **field** (*str*) – using either “node_feat” or “edge_feat” for Set2Set and Reduced readout
  * **niters_set2set** (*int*) – number of set2set iterations
  * **nlayers_set2set** (*int*) – number of set2set layers
  * **include_state** (*bool*) – whether to include states features
  * **activation_type** (*str*) – activation type. choose from ‘swish’, ‘tanh’, ‘sigmoid’, ‘softplus2’, ‘softexp’
  * **\*\*kwargs** – For future flexibility. Not used at the moment.

#### forward(g: DGLGraph, state_attr: Tensor | None = None, l_g: DGLGraph | None = None)

Performs message passing and updates node representations.

* **Parameters:**
  * **g** – DGLGraph for a batch of graphs.
  * **state_attr** – State attrs for a batch of graphs.
  * **l_g** – DGLGraph for a batch of line graphs.
* **Returns:**
  Output property for a batch of graphs
* **Return type:**
  output

#### predict_structure(structure, state_feats: torch.Tensor | None = None, graph_converter: GraphConverter | None = None)

Convenience method to directly predict property from structure.

* **Parameters:**
  * **structure** – An input crystal/molecule.
  * **state_feats** (*torch.tensor*) – Graph attributes
  * **graph_converter** – Object that implements a get_graph_from_structure.
* **Returns:**
  output property
* **Return type:**
  output (torch.tensor)

## matgl.models._megnet module

Implementation of MatErials Graph Network (MEGNet) model.

Graph networks are a new machine learning (ML) paradigm that supports both relational reasoning and combinatorial
generalization. For more details on MEGNet, please refer to:

```default
Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P. _Graph Networks as a Universal Machine Learning Framework for
Molecules and Crystals._ Chem. Mater. 2019, 31 (9), 3564-3572. DOI: 10.1021/acs.chemmater.9b01294.
```

### *class* matgl.models._megnet.MEGNet(dim_node_embedding: int = 16, dim_edge_embedding: int = 100, dim_state_embedding: int = 2, ntypes_state: int | None = None, nblocks: int = 3, hidden_layer_sizes_input: tuple[int, …] = (64, 32), hidden_layer_sizes_conv: tuple[int, …] = (64, 64, 32), hidden_layer_sizes_output: tuple[int, …] = (32, 16), nlayers_set2set: int = 1, niters_set2set: int = 2, activation_type: str = ‘softplus2’, is_classification: bool = False, include_state: bool = True, dropout: float | None = None, element_types: tuple[str, …] = (‘H’, ‘He’, ‘Li’, ‘Be’, ‘B’, ‘C’, ‘N’, ‘O’, ‘F’, ‘Ne’, ‘Na’, ‘Mg’, ‘Al’, ‘Si’, ‘P’, ‘S’, ‘Cl’, ‘Ar’, ‘K’, ‘Ca’, ‘Sc’, ‘Ti’, ‘V’, ‘Cr’, ‘Mn’, ‘Fe’, ‘Co’, ‘Ni’, ‘Cu’, ‘Zn’, ‘Ga’, ‘Ge’, ‘As’, ‘Se’, ‘Br’, ‘Kr’, ‘Rb’, ‘Sr’, ‘Y’, ‘Zr’, ‘Nb’, ‘Mo’, ‘Tc’, ‘Ru’, ‘Rh’, ‘Pd’, ‘Ag’, ‘Cd’, ‘In’, ‘Sn’, ‘Sb’, ‘Te’, ‘I’, ‘Xe’, ‘Cs’, ‘Ba’, ‘La’, ‘Ce’, ‘Pr’, ‘Nd’, ‘Pm’, ‘Sm’, ‘Eu’, ‘Gd’, ‘Tb’, ‘Dy’, ‘Ho’, ‘Er’, ‘Tm’, ‘Yb’, ‘Lu’, ‘Hf’, ‘Ta’, ‘W’, ‘Re’, ‘Os’, ‘Ir’, ‘Pt’, ‘Au’, ‘Hg’, ‘Tl’, ‘Pb’, ‘Bi’, ‘Ac’, ‘Th’, ‘Pa’, ‘U’, ‘Np’, ‘Pu’), bond_expansion: BondExpansion | None = None, cutoff: float = 4.0, gauss_width: float = 0.5, \*\*kwargs)

Bases: `Module`, `IOMixIn`

DGL implementation of MEGNet.

Useful defaults for all arguments have been specified based on MEGNet formation energy model.

* **Parameters:**
  * **dim_node_embedding** – Dimension of node embedding.
  * **dim_edge_embedding** – Dimension of edge embedding.
  * **dim_state_embedding** – Dimension of state embedding.
  * **ntypes_state** – Number of state types.
  * **nblocks** – Number of blocks.
  * **hidden_layer_sizes_input** – Architecture of dense layers before the graph convolution
  * **hidden_layer_sizes_conv** – Architecture of dense layers for message and update functions
  * **nlayers_set2set** – Number of layers in Set2Set layer
  * **niters_set2set** – Number of iterations in Set2Set layer
  * **hidden_layer_sizes_output** – Architecture of dense layers for concatenated features after graph convolution
  * **activation_type** – Activation used for non-linearity
  * **is_classification** – Whether this is classification task or not
  * **layer_node_embedding** – Architecture of embedding layer for node attributes
  * **layer_edge_embedding** – Architecture of embedding layer for edge attributes
  * **layer_state_embedding** – Architecture of embedding layer for state attributes
  * **include_state** – Whether the state embedding is included
  * **dropout** – Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
    a Bernoulli distribution
  * **element_types** – Elements included in the training set
  * **bond_expansion** – Gaussian expansion for edge attributes
  * **cutoff** – cutoff for forming bonds
  * **gauss_width** – width of Gaussian function for bond expansion
  * **\*\*kwargs** – For future flexibility. Not used at the moment.

#### forward(graph: dgl.DGLGraph, edge_feat: torch.Tensor, node_feat: torch.Tensor, state_feat: torch.Tensor)

Forward pass of MEGnet. Executes all blocks.

* **Parameters:**
  * **graph** – Input graph
  * **edge_feat** – Edge features
  * **node_feat** – Node features
  * **state_feat** – State features.
* **Returns:**
  Prediction

#### predict_structure(structure, state_feats: torch.Tensor | None = None, graph_converter: GraphConverter | None = None)

Convenience method to directly predict property from structure.

* **Parameters:**
  * **structure** – An input crystal/molecule.
  * **state_feats** (*torch.tensor*) – Graph attributes
  * **graph_converter** – Object that implements a get_graph_from_structure.
* **Returns:**
  output property
* **Return type:**
  output (torch.tensor)

## matgl.models._wrappers module

Implementations of pseudo-models that wrap other models.

### *class* matgl.models._wrappers.TransformedTargetModel(model: nn.Module, target_transformer: Transformer)

Bases: `Module`, `IOMixIn`

A model where the target is first transformed prior to training and the reverse transformation is performed for
predictions. This is modelled after scikit-learn’s TransformedTargetRegressor. It should be noted that this model
is almost never used for training since the general idea is to use the transformed target for loss computation.
Instead, it is created after a model has been fitted for serialization for end user to call the model to perform
predictions without having to worry about what target transformations have been performed.

* **Parameters:**
  * **model** (*nn.Module*) – Model to wrap.
  * **target_transformer** (*Transformer*) – Transformer to use for target transformation.

#### forward(\*args, \*\*kwargs)

* **Parameters:**
  * **\*args** – Passthrough to parent model.forward method.
  * **\*\*kwargs** – Passthrough to parent model.forward method.
* **Returns:**
  Inverse transformed output.

#### predict_structure(\*args, \*\*kwargs)

Pass through to parent model.predict_structure with inverse transform.

* **Parameters:**
  * **\*args** – Pass-through to self.model.predict_structure.
  * **\*\*kwargs** – Pass-through to self.model.predict_structure.
* **Returns:**
  Transformed answer.