---
layout: default
title: API Documentation
nav_order: 5
---
# matgl package

MatGL (Materials Graph Library) is a graph deep learning library for materials science.

## Subpackages


* [matgl.apps package](matgl.apps.md)




    * [matgl.apps.pes module](matgl.apps.md#module-matgl.apps.pes)


        * [`Potential`](matgl.apps.md#matgl.apps.pes.Potential)


            * [`Potential.forward()`](matgl.apps.md#matgl.apps.pes.Potential.forward)


            * [`Potential.training`](matgl.apps.md#matgl.apps.pes.Potential.training)


* [matgl.data package](matgl.data.md)




    * [matgl.data.transformer module](matgl.data.md#module-matgl.data.transformer)


        * [`LogTransformer`](matgl.data.md#matgl.data.transformer.LogTransformer)


            * [`LogTransformer.inverse_transform()`](matgl.data.md#matgl.data.transformer.LogTransformer.inverse_transform)


            * [`LogTransformer.transform()`](matgl.data.md#matgl.data.transformer.LogTransformer.transform)


        * [`Normalizer`](matgl.data.md#matgl.data.transformer.Normalizer)


            * [`Normalizer.from_data()`](matgl.data.md#matgl.data.transformer.Normalizer.from_data)


            * [`Normalizer.inverse_transform()`](matgl.data.md#matgl.data.transformer.Normalizer.inverse_transform)


            * [`Normalizer.transform()`](matgl.data.md#matgl.data.transformer.Normalizer.transform)


        * [`Transformer`](matgl.data.md#matgl.data.transformer.Transformer)


            * [`Transformer.inverse_transform()`](matgl.data.md#matgl.data.transformer.Transformer.inverse_transform)


            * [`Transformer.transform()`](matgl.data.md#matgl.data.transformer.Transformer.transform)


* [matgl.ext package](matgl.ext.md)




    * [matgl.ext.ase module](matgl.ext.md#module-matgl.ext.ase)


        * [`Atoms2Graph`](matgl.ext.md#matgl.ext.ase.Atoms2Graph)


            * [`Atoms2Graph.get_graph()`](matgl.ext.md#matgl.ext.ase.Atoms2Graph.get_graph)


        * [`M3GNetCalculator`](matgl.ext.md#matgl.ext.ase.M3GNetCalculator)


            * [`M3GNetCalculator.calculate()`](matgl.ext.md#matgl.ext.ase.M3GNetCalculator.calculate)


            * [`M3GNetCalculator.implemented_properties`](matgl.ext.md#matgl.ext.ase.M3GNetCalculator.implemented_properties)


        * [`MolecularDynamics`](matgl.ext.md#matgl.ext.ase.MolecularDynamics)


            * [`MolecularDynamics.run()`](matgl.ext.md#matgl.ext.ase.MolecularDynamics.run)


            * [`MolecularDynamics.set_atoms()`](matgl.ext.md#matgl.ext.ase.MolecularDynamics.set_atoms)


        * [`Relaxer`](matgl.ext.md#matgl.ext.ase.Relaxer)


            * [`Relaxer.relax()`](matgl.ext.md#matgl.ext.ase.Relaxer.relax)


        * [`TrajectoryObserver`](matgl.ext.md#matgl.ext.ase.TrajectoryObserver)


            * [`TrajectoryObserver.compute_energy()`](matgl.ext.md#matgl.ext.ase.TrajectoryObserver.compute_energy)


            * [`TrajectoryObserver.save()`](matgl.ext.md#matgl.ext.ase.TrajectoryObserver.save)


    * [matgl.ext.pymatgen module](matgl.ext.md#module-matgl.ext.pymatgen)


        * [`Molecule2Graph`](matgl.ext.md#matgl.ext.pymatgen.Molecule2Graph)


            * [`Molecule2Graph.get_graph()`](matgl.ext.md#matgl.ext.pymatgen.Molecule2Graph.get_graph)


        * [`Structure2Graph`](matgl.ext.md#matgl.ext.pymatgen.Structure2Graph)


            * [`Structure2Graph.get_graph()`](matgl.ext.md#matgl.ext.pymatgen.Structure2Graph.get_graph)


        * [`get_element_list()`](matgl.ext.md#matgl.ext.pymatgen.get_element_list)


* [matgl.graph package](matgl.graph.md)




    * [matgl.graph.compute module](matgl.graph.md#module-matgl.graph.compute)


        * [`compute_3body()`](matgl.graph.md#matgl.graph.compute.compute_3body)


        * [`compute_pair_vector_and_distance()`](matgl.graph.md#matgl.graph.compute.compute_pair_vector_and_distance)


        * [`compute_theta_and_phi()`](matgl.graph.md#matgl.graph.compute.compute_theta_and_phi)


        * [`create_line_graph()`](matgl.graph.md#matgl.graph.compute.create_line_graph)


    * [matgl.graph.converters module](matgl.graph.md#module-matgl.graph.converters)


        * [`GraphConverter`](matgl.graph.md#matgl.graph.converters.GraphConverter)


            * [`GraphConverter.get_graph()`](matgl.graph.md#matgl.graph.converters.GraphConverter.get_graph)


    * [matgl.graph.data module](matgl.graph.md#module-matgl.graph.data)


        * [`M3GNetDataset`](matgl.graph.md#matgl.graph.data.M3GNetDataset)


            * [`M3GNetDataset.has_cache()`](matgl.graph.md#matgl.graph.data.M3GNetDataset.has_cache)


            * [`M3GNetDataset.load()`](matgl.graph.md#matgl.graph.data.M3GNetDataset.load)


            * [`M3GNetDataset.process()`](matgl.graph.md#matgl.graph.data.M3GNetDataset.process)


            * [`M3GNetDataset.save()`](matgl.graph.md#matgl.graph.data.M3GNetDataset.save)


        * [`MEGNetDataset`](matgl.graph.md#matgl.graph.data.MEGNetDataset)


            * [`MEGNetDataset.has_cache()`](matgl.graph.md#matgl.graph.data.MEGNetDataset.has_cache)


            * [`MEGNetDataset.load()`](matgl.graph.md#matgl.graph.data.MEGNetDataset.load)


            * [`MEGNetDataset.process()`](matgl.graph.md#matgl.graph.data.MEGNetDataset.process)


            * [`MEGNetDataset.save()`](matgl.graph.md#matgl.graph.data.MEGNetDataset.save)


        * [`MGLDataLoader()`](matgl.graph.md#matgl.graph.data.MGLDataLoader)


        * [`collate_fn()`](matgl.graph.md#matgl.graph.data.collate_fn)


        * [`collate_fn_efs()`](matgl.graph.md#matgl.graph.data.collate_fn_efs)


* [matgl.layers package](matgl.layers.md)




    * [matgl.layers._activations module](matgl.layers.md#module-matgl.layers._activations)


        * [`SoftExponential`](matgl.layers.md#matgl.layers._activations.SoftExponential)


            * [`SoftExponential.forward()`](matgl.layers.md#matgl.layers._activations.SoftExponential.forward)


            * [`SoftExponential.training`](matgl.layers.md#matgl.layers._activations.SoftExponential.training)


        * [`SoftPlus2`](matgl.layers.md#matgl.layers._activations.SoftPlus2)


            * [`SoftPlus2.forward()`](matgl.layers.md#matgl.layers._activations.SoftPlus2.forward)


            * [`SoftPlus2.training`](matgl.layers.md#matgl.layers._activations.SoftPlus2.training)


    * [matgl.layers._atom_ref module](matgl.layers.md#module-matgl.layers._atom_ref)


        * [`AtomRef`](matgl.layers.md#matgl.layers._atom_ref.AtomRef)


            * [`AtomRef.fit()`](matgl.layers.md#matgl.layers._atom_ref.AtomRef.fit)


            * [`AtomRef.forward()`](matgl.layers.md#matgl.layers._atom_ref.AtomRef.forward)


            * [`AtomRef.get_feature_matrix()`](matgl.layers.md#matgl.layers._atom_ref.AtomRef.get_feature_matrix)


            * [`AtomRef.training`](matgl.layers.md#matgl.layers._atom_ref.AtomRef.training)


    * [matgl.layers._basis module](matgl.layers.md#module-matgl.layers._basis)


        * [`GaussianExpansion`](matgl.layers.md#matgl.layers._basis.GaussianExpansion)


            * [`GaussianExpansion.forward()`](matgl.layers.md#matgl.layers._basis.GaussianExpansion.forward)


            * [`GaussianExpansion.reset_parameters()`](matgl.layers.md#matgl.layers._basis.GaussianExpansion.reset_parameters)


            * [`GaussianExpansion.training`](matgl.layers.md#matgl.layers._basis.GaussianExpansion.training)


        * [`SphericalBesselFunction`](matgl.layers.md#matgl.layers._basis.SphericalBesselFunction)


            * [`SphericalBesselFunction.rbf_j0()`](matgl.layers.md#matgl.layers._basis.SphericalBesselFunction.rbf_j0)


        * [`SphericalBesselWithHarmonics`](matgl.layers.md#matgl.layers._basis.SphericalBesselWithHarmonics)


            * [`SphericalBesselWithHarmonics.forward()`](matgl.layers.md#matgl.layers._basis.SphericalBesselWithHarmonics.forward)


            * [`SphericalBesselWithHarmonics.training`](matgl.layers.md#matgl.layers._basis.SphericalBesselWithHarmonics.training)


        * [`SphericalHarmonicsFunction`](matgl.layers.md#matgl.layers._basis.SphericalHarmonicsFunction)


        * [`spherical_bessel_smooth()`](matgl.layers.md#matgl.layers._basis.spherical_bessel_smooth)


    * [matgl.layers._bond module](matgl.layers.md#module-matgl.layers._bond)


        * [`BondExpansion`](matgl.layers.md#matgl.layers._bond.BondExpansion)


            * [`BondExpansion.forward()`](matgl.layers.md#matgl.layers._bond.BondExpansion.forward)


            * [`BondExpansion.training`](matgl.layers.md#matgl.layers._bond.BondExpansion.training)


    * [matgl.layers._core module](matgl.layers.md#module-matgl.layers._core)


        * [`EdgeSet2Set`](matgl.layers.md#matgl.layers._core.EdgeSet2Set)


            * [`EdgeSet2Set.forward()`](matgl.layers.md#matgl.layers._core.EdgeSet2Set.forward)


            * [`EdgeSet2Set.reset_parameters()`](matgl.layers.md#matgl.layers._core.EdgeSet2Set.reset_parameters)


            * [`EdgeSet2Set.training`](matgl.layers.md#matgl.layers._core.EdgeSet2Set.training)


        * [`GatedMLP`](matgl.layers.md#matgl.layers._core.GatedMLP)


            * [`GatedMLP.forward()`](matgl.layers.md#matgl.layers._core.GatedMLP.forward)


            * [`GatedMLP.training`](matgl.layers.md#matgl.layers._core.GatedMLP.training)


        * [`MLP`](matgl.layers.md#matgl.layers._core.MLP)


            * [`MLP.depth`](matgl.layers.md#matgl.layers._core.MLP.depth)


            * [`MLP.forward()`](matgl.layers.md#matgl.layers._core.MLP.forward)


            * [`MLP.in_features`](matgl.layers.md#matgl.layers._core.MLP.in_features)


            * [`MLP.last_linear`](matgl.layers.md#matgl.layers._core.MLP.last_linear)


            * [`MLP.out_features`](matgl.layers.md#matgl.layers._core.MLP.out_features)


            * [`MLP.training`](matgl.layers.md#matgl.layers._core.MLP.training)


    * [matgl.layers._embedding module](matgl.layers.md#module-matgl.layers._embedding)


        * [`EmbeddingBlock`](matgl.layers.md#matgl.layers._embedding.EmbeddingBlock)


            * [`EmbeddingBlock.forward()`](matgl.layers.md#matgl.layers._embedding.EmbeddingBlock.forward)


            * [`EmbeddingBlock.training`](matgl.layers.md#matgl.layers._embedding.EmbeddingBlock.training)


    * [matgl.layers._graph_convolution module](matgl.layers.md#module-matgl.layers._graph_convolution)


        * [`M3GNetBlock`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetBlock)


            * [`M3GNetBlock.forward()`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetBlock.forward)


            * [`M3GNetBlock.training`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetBlock.training)


        * [`M3GNetGraphConv`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv)


            * [`M3GNetGraphConv.edge_update_()`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv.edge_update_)


            * [`M3GNetGraphConv.forward()`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv.forward)


            * [`M3GNetGraphConv.from_dims()`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv.from_dims)


            * [`M3GNetGraphConv.node_update_()`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv.node_update_)


            * [`M3GNetGraphConv.state_update_()`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv.state_update_)


            * [`M3GNetGraphConv.training`](matgl.layers.md#matgl.layers._graph_convolution.M3GNetGraphConv.training)


        * [`MEGNetBlock`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetBlock)


            * [`MEGNetBlock.forward()`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetBlock.forward)


            * [`MEGNetBlock.training`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetBlock.training)


        * [`MEGNetGraphConv`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv)


            * [`MEGNetGraphConv.edge_update_()`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv.edge_update_)


            * [`MEGNetGraphConv.forward()`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv.forward)


            * [`MEGNetGraphConv.from_dims()`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv.from_dims)


            * [`MEGNetGraphConv.node_update_()`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv.node_update_)


            * [`MEGNetGraphConv.state_update_()`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv.state_update_)


            * [`MEGNetGraphConv.training`](matgl.layers.md#matgl.layers._graph_convolution.MEGNetGraphConv.training)


    * [matgl.layers._readout module](matgl.layers.md#module-matgl.layers._readout)


        * [`ReduceReadOut`](matgl.layers.md#matgl.layers._readout.ReduceReadOut)


            * [`ReduceReadOut.forward()`](matgl.layers.md#matgl.layers._readout.ReduceReadOut.forward)


            * [`ReduceReadOut.training`](matgl.layers.md#matgl.layers._readout.ReduceReadOut.training)


        * [`Set2SetReadOut`](matgl.layers.md#matgl.layers._readout.Set2SetReadOut)


            * [`Set2SetReadOut.forward()`](matgl.layers.md#matgl.layers._readout.Set2SetReadOut.forward)


            * [`Set2SetReadOut.training`](matgl.layers.md#matgl.layers._readout.Set2SetReadOut.training)


        * [`WeightedReadOut`](matgl.layers.md#matgl.layers._readout.WeightedReadOut)


            * [`WeightedReadOut.forward()`](matgl.layers.md#matgl.layers._readout.WeightedReadOut.forward)


            * [`WeightedReadOut.training`](matgl.layers.md#matgl.layers._readout.WeightedReadOut.training)


        * [`WeightedReadOutPair`](matgl.layers.md#matgl.layers._readout.WeightedReadOutPair)


            * [`WeightedReadOutPair.forward()`](matgl.layers.md#matgl.layers._readout.WeightedReadOutPair.forward)


            * [`WeightedReadOutPair.training`](matgl.layers.md#matgl.layers._readout.WeightedReadOutPair.training)


    * [matgl.layers._three_body module](matgl.layers.md#module-matgl.layers._three_body)


        * [`ThreeBodyInteractions`](matgl.layers.md#matgl.layers._three_body.ThreeBodyInteractions)


            * [`ThreeBodyInteractions.forward()`](matgl.layers.md#matgl.layers._three_body.ThreeBodyInteractions.forward)


            * [`ThreeBodyInteractions.training`](matgl.layers.md#matgl.layers._three_body.ThreeBodyInteractions.training)


        * [`combine_sbf_shf()`](matgl.layers.md#matgl.layers._three_body.combine_sbf_shf)


* [matgl.models package](matgl.models.md)




    * [matgl.models._m3gnet module](matgl.models.md#module-matgl.models._m3gnet)


        * [`M3GNet`](matgl.models.md#matgl.models._m3gnet.M3GNet)


            * [`M3GNet.forward()`](matgl.models.md#matgl.models._m3gnet.M3GNet.forward)


            * [`M3GNet.predict_structure()`](matgl.models.md#matgl.models._m3gnet.M3GNet.predict_structure)


            * [`M3GNet.training`](matgl.models.md#matgl.models._m3gnet.M3GNet.training)


    * [matgl.models._megnet module](matgl.models.md#module-matgl.models._megnet)


        * [`MEGNet`](matgl.models.md#matgl.models._megnet.MEGNet)


            * [`MEGNet.forward()`](matgl.models.md#matgl.models._megnet.MEGNet.forward)


            * [`MEGNet.predict_structure()`](matgl.models.md#matgl.models._megnet.MEGNet.predict_structure)


            * [`MEGNet.training`](matgl.models.md#matgl.models._megnet.MEGNet.training)


    * [matgl.models._wrappers module](matgl.models.md#module-matgl.models._wrappers)


        * [`TransformedTargetModel`](matgl.models.md#matgl.models._wrappers.TransformedTargetModel)


            * [`TransformedTargetModel.forward()`](matgl.models.md#matgl.models._wrappers.TransformedTargetModel.forward)


            * [`TransformedTargetModel.predict_structure()`](matgl.models.md#matgl.models._wrappers.TransformedTargetModel.predict_structure)


            * [`TransformedTargetModel.training`](matgl.models.md#matgl.models._wrappers.TransformedTargetModel.training)


* [matgl.utils package](matgl.utils.md)




    * [matgl.utils.cutoff module](matgl.utils.md#module-matgl.utils.cutoff)


        * [`cosine_cutoff()`](matgl.utils.md#matgl.utils.cutoff.cosine_cutoff)


        * [`polynomial_cutoff()`](matgl.utils.md#matgl.utils.cutoff.polynomial_cutoff)


    * [matgl.utils.io module](matgl.utils.md#module-matgl.utils.io)


        * [`IOMixIn`](matgl.utils.md#matgl.utils.io.IOMixIn)


            * [`IOMixIn.load()`](matgl.utils.md#matgl.utils.io.IOMixIn.load)


            * [`IOMixIn.save()`](matgl.utils.md#matgl.utils.io.IOMixIn.save)


            * [`IOMixIn.save_args()`](matgl.utils.md#matgl.utils.io.IOMixIn.save_args)


        * [`RemoteFile`](matgl.utils.md#matgl.utils.io.RemoteFile)


        * [`get_available_pretrained_models()`](matgl.utils.md#matgl.utils.io.get_available_pretrained_models)


        * [`load_model()`](matgl.utils.md#matgl.utils.io.load_model)


    * [matgl.utils.maths module](matgl.utils.md#module-matgl.utils.maths)


        * [`broadcast()`](matgl.utils.md#matgl.utils.maths.broadcast)


        * [`broadcast_states_to_atoms()`](matgl.utils.md#matgl.utils.maths.broadcast_states_to_atoms)


        * [`broadcast_states_to_bonds()`](matgl.utils.md#matgl.utils.maths.broadcast_states_to_bonds)


        * [`get_range_indices_from_n()`](matgl.utils.md#matgl.utils.maths.get_range_indices_from_n)


        * [`get_segment_indices_from_n()`](matgl.utils.md#matgl.utils.maths.get_segment_indices_from_n)


        * [`repeat_with_n()`](matgl.utils.md#matgl.utils.maths.repeat_with_n)


        * [`scatter_sum()`](matgl.utils.md#matgl.utils.maths.scatter_sum)


        * [`spherical_bessel_roots()`](matgl.utils.md#matgl.utils.maths.spherical_bessel_roots)


        * [`unsorted_segment_fraction()`](matgl.utils.md#matgl.utils.maths.unsorted_segment_fraction)


    * [matgl.utils.training module](matgl.utils.md#module-matgl.utils.training)


        * [`ModelTrainer`](matgl.utils.md#matgl.utils.training.ModelTrainer)


            * [`ModelTrainer.forward()`](matgl.utils.md#matgl.utils.training.ModelTrainer.forward)


            * [`ModelTrainer.loss_fn()`](matgl.utils.md#matgl.utils.training.ModelTrainer.loss_fn)


            * [`ModelTrainer.step()`](matgl.utils.md#matgl.utils.training.ModelTrainer.step)


        * [`PotentialTrainer`](matgl.utils.md#matgl.utils.training.PotentialTrainer)


            * [`PotentialTrainer.forward()`](matgl.utils.md#matgl.utils.training.PotentialTrainer.forward)


            * [`PotentialTrainer.loss_fn()`](matgl.utils.md#matgl.utils.training.PotentialTrainer.loss_fn)


            * [`PotentialTrainer.step()`](matgl.utils.md#matgl.utils.training.PotentialTrainer.step)


        * [`TrainerMixin`](matgl.utils.md#matgl.utils.training.TrainerMixin)


            * [`TrainerMixin.configure_optimizers()`](matgl.utils.md#matgl.utils.training.TrainerMixin.configure_optimizers)


            * [`TrainerMixin.on_test_model_eval()`](matgl.utils.md#matgl.utils.training.TrainerMixin.on_test_model_eval)


            * [`TrainerMixin.on_train_epoch_end()`](matgl.utils.md#matgl.utils.training.TrainerMixin.on_train_epoch_end)


            * [`TrainerMixin.predict_step()`](matgl.utils.md#matgl.utils.training.TrainerMixin.predict_step)


            * [`TrainerMixin.test_step()`](matgl.utils.md#matgl.utils.training.TrainerMixin.test_step)


            * [`TrainerMixin.training_step()`](matgl.utils.md#matgl.utils.training.TrainerMixin.training_step)


            * [`TrainerMixin.validation_step()`](matgl.utils.md#matgl.utils.training.TrainerMixin.validation_step)


        * [`xavier_init()`](matgl.utils.md#matgl.utils.training.xavier_init)



## matgl.config module

Global configuration variables for matgl.


### matgl.config.clear_cache(confirm: bool = True)
Deletes all files in the matgl.cache. This is used to clean out downloaded models.


* **Parameters**

    **confirm** – Whether to ask for confirmation. Default is True.
