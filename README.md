# OMatG: Open Materials Generation

![csp movie](assets/csp_movie.gif)

A state-of-the-art generative model for crystal structure prediction and *de novo* generation of inorganic crystals. 

This open-source framework accompanies the paper available on [arXiv](https://arxiv.org/abs/2502.02582),
which should be cited when using it.

> **NOTE:** 🚧 This repository is currently work in progress. 🚧

## Table of Contents

- [Dependencies](#dependencies)
- [Examples](#examples)
- [Training](#training)
- [Sampling](#sampling)
- [Visualize](#visualize)
- [Match Rate Computation](#match-rate-csp)
- [Datasets](#datasets)
- [OMatG Data Format](#omatg-data-format)
- [Citing OMatG](#citing-omatg)

## Dependencies

You can use any Python version between 3.10 and 3.12.

Install the dependencies and the `omg` package itself via pip. 

To install `omg` in editable mode (recommended for developers), use `pip install -e .` within the base directory of this 
repository once. Any changes in code are directly available in the installed `omg` package. 

> **NOTE**: Installing `torch=2.7.1` based on your version of `CUDA` in addition to `torch_geometric`, and `torch-scatter` before installing `omg` can help minimize sources of errors

To install `omg` as a package, use `pip install .` instead. If the code in this repository changes, this command has to 
executed again to also change the code of the installed package.

## Overview

OMatG leverages [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for model specification, training, and generation. To implement an OMatG model, one must specify a `config.yaml` file. See the [```omg/conf_examples```](omg/conf_examples) directory for some functional examples. Elements of model training and inference such as checkpointing and learning hyperparameters are handled specified in the configuration file and are handled by the [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html?utm_term=&utm_campaign=PMax+Q1+FY26&utm_source=adwords&utm_medium=ppc&hsa_acc=1332059986&hsa_cam=22691136685&hsa_grp=&hsa_ad=&hsa_src=x&hsa_tgt=&hsa_kw=&hsa_mt=&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=22681463925&gbraid=0AAAAA9Wu012kR5bgEIP9NsR-6TV6-wOhC&gclid=CjwKCAjw7MLDBhAuEiwAIeXGIfbIFwl3BTbZ6uOAzPJyLtOYqBBOx9YwvtIwLXCN0bulE2jVBTrDyBoCJ6kQAvD_BwE)

In a `config.yaml` file, one must specify a `SingleStochasticInterpolant` for each element of `(fractional_coordinates, lattice_vectors, atom_types)` by providing an [```Interpolant```](omg/si/interpolants.py) and a [```Gamma```](omg/si/gamma.py) plus additional parameters for training and sampling of the generative model (see [our paper](https://arxiv.org/abs/2502.02582) for more complete details). For `atom_types`, one must specify a [StochasticInterpolantSpecies](omg/si/abstracts.py) which has been implemented based on [discrete flow matching (DFM)](https://arxiv.org/abs/2402.04997) and is well-tailored to the discrete nature of generative modeling for chemical composition. [Four datasets](omg/data) have been provided, one of which must also be specified in this configuration file. 

The generative process itself relies almost exclusively on classes found in the [```omg/si```](omg/si) directory. At a high level, one should focus on stringing together instantiations of the [```SingleStochasticInterpolant```](omg/si/single_stochastic_interpolant.py) class into one [```StochasticInterpolants```](omg/si/stochastic_interpolants.py) object. Each ```SingleStochasticInterpolant``` object provides an `interpolate` method which maps to an intermediate point, $x_t$, between the point, $x_0$, sampled from some base distribution, and some $x_1$ sampled from the training set. During training, a neural network, $b^{\theta}(t, x)$, is regressed onto the time derivative of $x(t, x_0, x_1, z)$ batchwise. Optionally, one can also regress another neural network, $z^{\phi}(t, x)$, onto the added noise, $z$. Each `StochasticInterpolant` provides a way to generate new data from some initial $x_0$ through the `integrate` method. This can be done via an ODE or an SDE once training is complete. See [Albergo et al.](https://arxiv.org/abs/2303.08797) for more details on the stochastic interpolants framework. 

## Examples

Tutorial notebooks for using OMatG are available on [Kaggle](https://www.kaggle.com/philipphoellmer/code). In particular the [Crystals](https://www.kaggle.com/code/philipphoellmer/generative-modeling-workshop-session-crystals-sol) and [Crystals Sol](https://www.kaggle.com/code/philipphoellmer/generative-modeling-workshop-session-crystals-sol) notebooks are relevant.

## Training


Run the following command in any directory to train from scratch based on the configuration file `config.yaml`:

```bash
omg fit --config config.yaml --trainer.accelerator=gpu
```

This command will create checkpoints, log files, and cache files in the working directory.

If you want to include a Wandb logger with a name, add the `--trainer.logger=WandbLogger --trainer.logger.name=<name>` 
argument. Other loggers can be found [here](https://lightning.ai/docs/pytorch/stable/extensions/logging.html).

In order to restart training from a checkpoint, add the `--ckpt_path=<checkpoint_file.ckpt>` argument. 

In order to seed the random number generators before training, use `--seed_everything=<seed>`.

Exemplary configuration files can be found in the `omg/conf_examples` directory.

The training command can be executed in any directory. The configuration files contain paths to lmbd data files that are 
used, e.g., for training. The path to these data files can either be relative to the working directory, or relative to 
the `omg` directory (that is, use `"data/mp_20/val.lmdb"` for `lmdb_paths` in order to use the `mp_20` data set as in 
exemplary configuration files).

> **NOTE**: Model checkpoints used in the paper can be found at the [HuggingFace link](https://huggingface.co/OMatG) associated with OMatG. See details on [generating materials](#sampling) with checkpoint files below.


## Sampling

For generating new structures in an xyz file, run the following command:

```bash
omg predict --config {config_file} --ckpt_path=<checkpoint_file.ckpt> --model.generation_xyz_filename=<xyz_file> --data.batch_size=1024 --seed_everything=42 --trainer.max_epochs=1
```

For an xyz filename `filename.xyz`, this command will also create a file `filename_init.xyz` that contains the initial
structures that were integrated to yield the structures in `filename.xyz`. This file is required for the visualization
below.

## Visualize

Run the following command to compare distributions over the generated structures in an xyz file to distributions over 
training dataset:

```bash
omg visualize --config {config_file} --xyz_file {xyz_file} --plot_name {plot_name}
```

## Match Rate (CSP)

Run the following command to compute the match rate between the generated structures in an xyz file and the structures 
in the prediction dataset, as well as the rate of unique structures in the generated structures:

```bash
omg match --config {config_file} --xyz_file {xyz_file}
```

Computing the match rate as in DiffCSP or FlowMM requires to validate every structure which is quite slow. Also, 
computing the unique rate is slow. One can use the `--skip_validation=true` and `--skip_unique=true` arguments to skip
these computations. 

The validations, and the computations of the match rate and unique rate are parallelized. The number of processes is 
determined by `os.cpu_count()`. This can be changed by setting the `--number_cpus` argument (which is probably most 
useful in cluster environments).

## Datasets

All datasets can be found in [```omg/data```](omg/data). They are each described briefly below:

- MP-20 - 45,231 structures from the [Materials Project](https://pubs.aip.org/aip/apm/article/1/1/011002/119685/Commentary-The-Materials-Project-A-materials) with a maximum of 20 atoms per structure.
- MPTS-52 - [Chronologically split data from the Materials Project](https://joss.theoj.org/papers/10.21105/joss.05618) with 40,476 structures and up to 52 atoms per structure.
- Perov-5 - A [perovskite dataset](https://pubs.rsc.org/en/content/articlelanding/2012/ee/c2ee22341d) containing 18,928 structures each with five atoms per structure.
- Carbon-24 - A [dataset](https://arxiv.org/abs/2110.06197) of 10,153 structures consisting only of carbon with up to 24 atoms per structure. We did NOT include this as a benchmark in our paper.
- Alex-MP-20 - A [consolidated dataset](https://www.nature.com/articles/s41586-025-08628-5) of 675,204 structures of [Alexandria](https://arxiv.org/abs/2210.00579) and [MP-20](https://pubs.aip.org/aip/apm/article/1/1/011002/119685/Commentary-The-Materials-Project-A-materials) structures. The Alex-MP20 dataset is too large to be stored in this repository. We have made it available via the [HuggingFace link](https://huggingface.co/OMatG) associated with this project.

<!---
## Curriculum Learning

Run the following command to use a given configuration file with stochastic interpolants for all datafields 'pos', 
'species', and 'cell' to generate a new configuration file where only some of the stochastic interpolants are used, 
while the others are replaced by identity interpolants:

```bash
omg curriculum --config {config_file} --lessons {lessons}
```

Here, lessons should be a list of strings so, e.g., [pos] or [pos,species] (spaces can be included when the brackets are
surrounded by quotes).
--->
## OMatG Data Format

For a batch size of batch_size, the `torch_geometric.data.Data` instances contain the following attributes:
- `n_atoms`: `torch.Tensor` of shape `(batch_size, )` containing the number of atoms in each configuration.
- `batch`: `torch.Tensor` of shape `(sum(n_atoms),)` containing the index of the configuration to which each atom 
belongs.
- `species`: `torch.Tensor` of shape `(sum(n_atoms),)` containing the atomic numbers of the atoms in the configurations.
- `pos`: `torch.Tensor` of shape `(sum(n_atoms), 3)` containing the atomic positions of the atoms in the configurations.
- `cell`: `torch.Tensor` of shape `(batch_size, 3, 3)` containing the cell vectors of the configurations.
- `ptr`: `torch.Tensor` of shape `(batch_size + 1,)` containing the indices of the first atom of each configuration in 
the `species` and `pos` tensors.
- `property`: dict containing the properties of the configurations.

## Citing OMatG
```bibtex
@inproceedings{
    hoellmer_open_2024,
    title={Open Materials Generation with Stochastic Interpolants},
    author={Philipp Höllmer and Thomas Egg and Maya M. Martirossyan and Eric Fuemmeler and Amit Gupta and Zeren Shui and Pawan Prakash and Adrian Roitberg and Mingjie Liu and George Karypis and Mark Transtrum and Richard G. Hennig and Ellad B. Tadmor and Stefano Martiniani},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=gHGrzxFujU},
    note={Also at {\ttfamily arXiv:2502.02582} (\url{https://arxiv.org/abs/2502.02582})},
}
```