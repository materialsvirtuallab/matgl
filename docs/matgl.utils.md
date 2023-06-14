---
layout: default
title: matgl.utils.md
nav_exclude: true
---
# matgl.utils package

Implementation of various utility methods and classes.


## matgl.utils.cutoff module

Cutoff functions for constructing M3GNet potentials.


### matgl.utils.cutoff.cosine_cutoff(r: Tensor, cutoff: float)
Cosine cutoff function
:param r: radius distance tensor
:type r: torch.tensor
:param cutoff: cutoff distance.
:type cutoff: float

Returns: cosine cutoff functions


### matgl.utils.cutoff.polynomial_cutoff(r, cutoff: float)
Polynomial cutoff function
:param r: radius distance tensor
:type r: torch.tensor
:param cutoff: cutoff distance.
:type cutoff: float

Returns: polynomial cutoff functions

## matgl.utils.io module

Provides utilities for managing models and data.


### _class_ matgl.utils.io.IOMixIn()
Bases: `object`

Mixin class for model saving and loading.

For proper usage, models should subclass nn.Module and IOMix and the save_args method should be called
immediately after the super().__init__() call:

```default
super().__init__()
self.save_args(locals(), kwargs)
```


#### _classmethod_ load(path: str | Path | dict, \*\*kwargs)
Load the model weights from a directory.


* **Parameters**

    
    * **path** (*str**|**path**|**dict*) – Path to saved model or name of pre-trained model. If it is a dict, it is assumed to
    be of the form:

    ```default
    {
        "model.pt": path to model.pt file,
        "state.pt": path to state file,
        "model.json": path to model.json file
    }
    ```

    Otherwise, the search order is path, followed by download from PRETRAINED_MODELS_BASE_URL
    (with caching).



    * **\*\*kwargs** – Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
    want to update the model.


Returns: model_object.


#### save(path: str | Path = '.', metadata: dict | None = None, makedirs: bool = True)
Save model to a directory.

Three files will be saved.
- path/model.pt, which contains the torch serialized model args.
- path/state.pt, which contains the saved state_dict from the model.
- path/model.json, a txt version of model.pt that is purely meant for ease of reference.


* **Parameters**

    
    * **path** – String or Path object to directory for model saving. Defaults to current working directory (“.”).


    * **metadata** – Any additional metadata to be saved into the model.json file. For example, a good use would be
    a description of model purpose, the training set used, etc.


    * **makedirs** – Whether to create the directory using os.makedirs(exist_ok=True). Note that if the directory
    already exists, makedirs will not do anything.



#### save_args(locals: dict, kwargs: dict | None = None)
Method to save args into a private _init_args variable.

This should be called after super in the __init__ method, e.g., self.save_args(locals(), kwargs).


* **Parameters**

    
    * **locals** – The result of locals().


    * **kwargs** – kwargs passed to the class.



### _class_ matgl.utils.io.RemoteFile(uri: str, cache_location: str | Path = PosixPath('/Users/shyue/.cache/matgl'), force_download: bool = False)
Bases: `object`

Handling of download of remote files to a local cache.

Args:
uri: Uniform resource identifier.
cache_location: Directory to cache downloaded RemoteFile. By default, downloaded models are saved at
$HOME/.matgl.
force_download: To speed up access, a model with the same name in the cache location will be used if
present. If you want to force a re-download, set this to True.


### matgl.utils.io.get_available_pretrained_models()
Checks Github for available pretrained_models for download. These can be used with load_model.


* **Returns**

    List of available models.



### matgl.utils.io.load_model(path: Path, \*\*kwargs)
Convenience method to load a model from a directory or name.


* **Parameters**

    
    * **path** (*str**|**path*) – Path to saved model or name of pre-trained model. The search order is path, followed by
    download from PRETRAINED_MODELS_BASE_URL (with caching).


    * **\*\*kwargs** – Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
    want to update the model.



* **Returns**

    model_object if include_json is false. (model_object, dict) if include_json is True.



* **Return type**

    Returns


## matgl.utils.maths module

Implementations of math functions.


### matgl.utils.maths.CWD(_ = '/Users/shyue/repos/matgl/matgl/utils_ )
Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index) root of
order l Spherical Bessel function is the [l, n] entry.


### _class_ matgl.utils.maths.GaussianExpansion(initial: float = 0.0, final: float = 4.0, num_centers: int = 20, width: None | float = 0.5)
Bases: `Module`

Gaussian Radial Expansion.
The bond distance is expanded to a vector of shape [m],
where m is the number of Gaussian basis centers.


* **Parameters**

    
    * **initial** – Location of initial Gaussian basis center.


    * **final** – Location of final Gaussian basis center


    * **num_centers** – Number of Gaussian Basis functions


    * **width** – Width of Gaussian Basis functions.



#### forward(bond_dists)
Expand distances.


* **Parameters**

    **bond_dists** – Bond (edge) distances between two atoms (nodes)



* **Returns**

    A vector of expanded distance with shape [num_centers]



#### reset_parameters()
Reinitialize model parameters.


#### training(_: boo_ )

### _class_ matgl.utils.maths.SphericalBesselFunction(max_l: int, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False)
Bases: `object`

Calculate the spherical Bessel function based on sympy + pytorch implementations.

Args:
max_l: int, max order (excluding l)
max_n: int, max number of roots used in each l
cutoff: float, cutoff radius
smooth: Whether to smooth the function.


#### _static_ rbf_j0(r, cutoff: float = 5.0, max_n: int = 3)
Spherical Bessel function of order 0, ensuring the function value
vanishes at cutoff.


* **Parameters**

    
    * **r** – torch.tensor pytorch tensors


    * **cutoff** – float, the cutoff radius


    * **max_n** – int max number of basis


Returns: basis function expansion using first spherical Bessel function


### _class_ matgl.utils.maths.SphericalHarmonicsFunction(max_l: int, use_phi: bool = True)
Bases: `object`

Spherical Harmonics function.

Args:
max_l: int, max l (excluding l)
use_phi: bool, whether to use the polar angle. If not,
the function will compute Y_l^0.


### matgl.utils.maths.broadcast(input_tensor: tensor, target_tensor: tensor, dim: int)
Broadcast input tensor along a given dimension to match the shape of the target tensor.
Modified from torch_scatter library ([https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)).


* **Parameters**

    
    * **input_tensor** – The tensor to broadcast.


    * **target_tensor** – The tensor whose shape to match.


    * **dim** – The dimension along which to broadcast.



* **Returns**

    resulting input tensor after broadcasting



### matgl.utils.maths.broadcast_states_to_atoms(g, state_feat)
Broadcast state attributes of shape [Ns, Nstate] to
bond attributes shape [Nb, Nstate].


* **Parameters**

    
    * **g** – DGL graph


    * **state_feat** – state_feature


Returns: broadcasted state attributes


### matgl.utils.maths.broadcast_states_to_bonds(g, state_feat)
Broadcast state attributes of shape [Ns, Nstate] to
bond attributes shape [Nb, Nstate].


* **Parameters**

    
    * **g** – DGL graph


    * **state_feat** – state_feature


Returns: broadcasted state attributes


### matgl.utils.maths.combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool)
Combine the spherical Bessel function and the spherical Harmonics function.

For the spherical Bessel function, the column is ordered by

    [n=[0, …, max_n-1], n=[0, …, max_n-1], …], max_l blocks,

For the spherical Harmonics function, the column is ordered by

    [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], …] max_l blocks, and each
    block has 2\*l + 1
    if use_phi is False, then the columns become
    [m=[0], m=[0], …] max_l columns


* **Parameters**

    
    * **sbf** – torch.tensor spherical bessel function results


    * **shf** – torch.tensor spherical harmonics function results


    * **max_n** – int, max number of n


    * **max_l** – int, max number of l


    * **use_phi** – whether to use phi


Returns:


### matgl.utils.maths.get_range_indices_from_n(ns)
Give ns = [2, 3], return [0, 1, 0, 1, 2].


* **Parameters**

    **ns** – torch.tensor, the number of atoms/bonds array


Returns: range indices


### matgl.utils.maths.get_segment_indices_from_n(ns)
Get segment indices from number array. For example if
ns = [2, 3], then the function will return [0, 0, 1, 1, 1].


* **Parameters**

    **ns** – torch.tensor, the number of atoms/bonds array



* **Return type**

    object


Returns: segment indices tensor


### matgl.utils.maths.repeat_with_n(ns, n)
Repeat the first dimension according to n array.


* **Parameters**

    
    * **ns** (*torch.tensor*) – tensor


    * **n** (*torch.tensor*) – a list of replications


Returns: repeated tensor


### matgl.utils.maths.scatter_sum(input_tensor: tensor, segment_ids: tensor, num_segments: int, dim: int)
Scatter sum operation along the specified dimension. Modified from the
torch_scatter library ([https://github.com/rusty1s/pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)).


* **Parameters**

    
    * **input_tensor** (*torch.Tensor*) – The input tensor to be scattered.


    * **segment_ids** (*torch.Tensor*) – Segment ID for each element in the input tensor.


    * **num_segments** (*int*) – The number of segments.


    * **dim** (*int*) – The dimension along which the scatter sum operation is performed (default: -1).



* **Returns**

    resulting tensor



### matgl.utils.maths.spherical_bessel_roots(max_l: int, max_n: int)
Calculate the spherical Bessel roots. The n-th root of the l-th
spherical bessel function is the [l, n] entry of the return matrix.
The calculation is based on the fact that the n-root for l-th
spherical Bessel function j_l, i.e., z_{j, n} is in the range
[z_{j-1,n}, z_{j-1, n+1}]. On the other hand we know precisely the
roots for j0, i.e., sinc(x).


* **Parameters**

    
    * **max_l** – max order of spherical bessel function


    * **max_n** – max number of roots


Returns: root matrix of size [max_l, max_n]


### matgl.utils.maths.spherical_bessel_smooth(r, cutoff: float = 5.0, max_n: int = 10)
This is an orthogonal basis with first
and second derivative at the cutoff
equals to zero. The function was derived from the order 0 spherical Bessel
function, and was expanded by the different zero roots.

Ref:

    [https://arxiv.org/pdf/1907.02374.pdf](https://arxiv.org/pdf/1907.02374.pdf)


* **Parameters**

    
    * **r** – torch.tensor distance tensor


    * **cutoff** – float, cutoff radius


    * **max_n** – int, max number of basis, expanded by the zero roots


Returns: expanded spherical harmonics with derivatives smooth at boundary


### matgl.utils.maths.unsorted_segment_fraction(data: tensor, segment_ids: tensor, num_segments: tensor)
Segment fraction
:param data: original data
:type data: torch.tensor
:param segment_ids: segment ids
:type segment_ids: torch.tensor
:param num_segments: number of segments
:type num_segments: torch.tensor


* **Returns**

    data after fraction.



* **Return type**

    data (torch.tensor)


## matgl.utils.training module

Utils for training MatGL models.


### _class_ matgl.utils.training.ModelTrainer(model, data_mean=None, data_std=None, loss: str = 'mse_loss', optimizer: Optimizer | None = None, scheduler: lr_scheduler | None = None, lr: float = 0.001, decay_steps: int = 1000, decay_alpha: float = 0.01)
Bases: `TrainerMixin`, `LightningModule`

Trainer for MEGNet and M3GNet models.

Args:
model: Which type of the model for training
data_mean: average of training data
data_std: standard deviation of training data
loss: loss function used for training
optimizer: optimizer for training
scheduler: scheduler for training
lr: learning rate for training
decay_steps: number of steps for decaying learning rate
decay_alpha: parameter determines the minimum learning rate.


#### forward(g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.tensor | None = None)

* **Parameters**

    
    * **g** – dgl Graph


    * **l_g** – Line graph


    * **state_attr** – State attribute.



* **Returns**

    Model prediction.



#### loss_fn(loss: Module, labels: tuple, preds: tuple)

* **Parameters**

    
    * **loss** – Loss function.


    * **labels** – Labels to compute the loss.


    * **preds** – Predictions.



* **Returns**

    total_loss, “MAE”: mae, “RMSE”: rmse}



* **Return type**

    {“Total_Loss”



#### step(batch: tuple)

* **Parameters**

    **batch** – Batch of training data.



* **Returns**

    results, batch_size



### _class_ matgl.utils.training.PotentialTrainer(model, element_refs: np.darray | None = None, energy_weight: float = 1.0, force_weight: float = 1.0, stress_weight: float | None = None, data_mean=None, data_std=None, calc_stress: bool = False, loss: str = 'mse_loss', optimizer: Optimizer | None = None, scheduler: lr_scheduler | None = None, lr: float = 0.001, decay_steps: int = 1000, decay_alpha: float = 0.01)
Bases: `TrainerMixin`, `LightningModule`

Trainer for MatGL potentials.

Init PotentialTrainer with key parameters.


* **Parameters**

    
    * **model** – Which type of the model for training


    * **element_refs** – element offset for PES


    * **energy_weight** – relative importance of energy


    * **force_weight** – relative importance of force


    * **stress_weight** – relative importance of stress


    * **data_mean** – average of training data


    * **data_std** – standard deviation of training data


    * **calc_stress** – whether stress calculation is required


    * **loss** – loss function used for training


    * **optimizer** – optimizer for training


    * **scheduler** – scheduler for training


    * **lr** – learning rate for training


    * **decay_steps** – number of steps for decaying learning rate


    * **decay_alpha** – parameter determines the minimum learning rate.



#### forward(g: dgl.DGLGraph, l_g: dgl.DGLGraph | None = None, state_attr: torch.tensor | None = None)

* **Parameters**

    
    * **g** – dgl Graph


    * **l_g** – Line graph


    * **state_attr** – State attr.



* **Returns**

    energy, force, stress, h



#### loss_fn(loss: nn.Module, labels: tuple, preds: tuple, energy_weight: float | None = None, force_weight: float | None = None, stress_weight: float | None = None, num_atoms: int | None = None)
Compute losses for EFS.


* **Parameters**

    
    * **loss** – Loss function.


    * **labels** – Labels.


    * **preds** – Predictions


    * **energy_weight** – Weight for energy loss.


    * **force_weight** – Weight for force loss.


    * **stress_weight** – Weight for stress loss.


    * **num_atoms** – Number of atoms.


Returns:

```default
{
    "Total_Loss": total_loss,
    "Energy_MAE": e_mae,
    "Force_MAE": f_mae,
    "Stress_MAE": s_mae,
    "Energy_RMSE": e_rmse,
    "Force_RMSE": f_rmse,
    "Stress_RMSE": s_rmse,
}
```


#### step(batch: tuple)

* **Parameters**

    **batch** – Batch of training data.



* **Returns**

    results, batch_size



### _class_ matgl.utils.training.TrainerMixin()
Bases: `object`

Mix-in class implementing common functions for training.


#### configure_optimizers()
Configure optimizers.


#### on_test_model_eval(\*args, \*\*kwargs)
Executed on model testing.


* **Parameters**

    
    * **\*args** – Pass-through


    * **\*\*kwargs** – Pass-through.



#### on_train_epoch_end()
Step scheduler every epoch.


#### predict_step(batch, batch_idx, dataloader_idx=0)
Prediction step.


* **Parameters**

    
    * **batch** – Data batch.


    * **batch_idx** – Batch index.


    * **dataloader_idx** – Data loader index.



* **Returns**

    Prediction



#### test_step(batch: tuple, batch_idx: int)
Test step.


* **Parameters**

    
    * **batch** – Data batch.


    * **batch_idx** – Batch index.



#### training_step(batch: tuple, batch_idx: int)
Training step.


* **Parameters**

    
    * **batch** – Data batch.


    * **batch_idx** – Batch index.



* **Returns**

    Total loss.



#### validation_step(batch: tuple, batch_idx: int)
Validation step.


* **Parameters**

    
    * **batch** – Data batch.


    * **batch_idx** – Batch index.



### matgl.utils.training.xavier_init(model: Module)
Xavier initialization scheme for the model.


* **Parameters**

    **model** (*nn.Module*) – The model to be Xavier-initialized.
