# Federated Learning under Distributed Concept Drift (FedDrift)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository is the source code for our paper: [Federated Learning under Distributed Concept Drift (AISTATS'23)](https://proceedings.mlr.press/v206/jothimurugesan23a/jothimurugesan23a.pdf).

Federated Learning (FL) under distributed concept drift is a largely unexplored area. Although concept drift is itself a well-studied phenomenon, it poses particular challenges for FL, because drifts arise staggered in time and space (across clients).
We first demonstrate that prior solutions to drift adaptation that use a single global model are ill-suited to staggered drifts, necessitating multiple-model solutions. 
We identify the problem of drift adaptation as a time-varying clustering problem, and we propose two new clustering algorithms for reacting to drifts based on local drift detection and hierarchical clustering.
Empirical evaluation shows that our solutions achieve significantly higher accuracy than existing baselines, and are comparable to an idealized algorithm with oracle knowledge of the ground-truth clustering of clients to concepts at each time step.

This repository is built on top of a federated learning research platform, [FedML](https://github.com/FedML-AI/FedML). 

# Setup Environment

Our installation script is based on [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Please modify the script according to your package manager.

```
$ ./CI-install.sh
```

Our expeirments are tested with Python 3.7.4 and PyTorch 10.2. You can also install the GPU-enabled PyTorch by modifying the script above.

The above script will create a conda environment named `feddrift` and install all the required packages. Please activate the environment before running any experiments.

```
$ conda activate feddrift
```

# Running Experiments

We use [Weights & Biases](https://wandb.ai/site) to log experiments. Please create an account and log in before running an experiment.

```
$ wandb login
```

To run an experiment, please use the following command:

```
$ cd fedml_experiments/distributed/fedavg_cont_ens
$ ./run_fedavg_distributed_pytorch.sh ${CLIENT_NUM} ${WORKER_NUM} ${SERVER_NUM} ${GPU_PER_SERVER} ${MODEL} ${DATA_DIST} ${ROUND} ${EPOCH} ${BATCH_SIZE} ${LR} ${DATASET} ${DATA_DIR} ${SAMPLE_NUM} ${NOISE_PROB} ${ONLY_TEST_ONE_CLIENT} ${TOTAL_ITER} ${CONCEPT_NUM} ${RESET_MODEL} ${DRIFT_TOGETHER} ${DRIFT_ALGO} ${DRIFT_ALGO_ARG} ${TIME_STRETH} ${DUMMY_ARG} ${CHANGE_POINT}
```

For example, running FedDrift for SEA-4 dataset and change point **A**:

```
$ ./run_fedavg_distributed_pytorch.sh 10 10 1 4 fnn homo 200 5 500 0.01 sea "./../../../data/" 100 0 0 10 4 0 0 softcluster H_A_C_1_10_0 1 0 A
```

The major algorithms we implemented are listed here (in the form of `${DRIFT_ALGO}` + optionally `${DRIFT_ALGO_ARG}`):

* `win-1`: Window method (one time step)
* `win-2`: Window method (two time step)
* `all`: Oblivous method (using all training data)
* `lin`: Weighted (linear decay)
* `exp`: Weighted (exponential decay)
* `dsurf`: [DriftSurf](https://proceedings.mlr.press/v139/tahmasbi21a/tahmasbi21a.pdf)
* `aue`: [Accuracy Updated Ensemble](https://link.springer.com/chapter/10.1007/978-3-642-21222-2_19)
* `kue`: [Kappa Updated Ensemble](https://link.springer.com/article/10.1007/s10994-019-05840-z)
* `softcluster` + `cfl_0.1_win-1`: Clustered federated learning
* `softclusterwin-1` + `hard-r`: [IFCA](https://arxiv.org/abs/2006.04088)
* `ada` + `win-1_iter`: [Adaptive-FedAvg](https://ieeexplore.ieee.org/document/9533710)
* `softcluster` + `H_A_F_1_06_0`: FedDrift (ours)
* `softcluster` + `mmacc_06`: FedDrift-Eagar (ours)

# Reference Papers

If you use our code in your work, we would appreciate a reference to the following papers

Ellango Jothimurugesan, Kevin Hsieh, Jianyu Wang, Gauri Joshi, Phillip B. Gibbons. [Federated Learning under Distributed Concept Drift](https://proceedings.mlr.press/v206/jothimurugesan23a/jothimurugesan23a.pdf). Proceedings of The 26th International Conference on Artificial Intelligence and Statistics (AISTATS), 2023.


# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
[https://cla.microsoft.com](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

