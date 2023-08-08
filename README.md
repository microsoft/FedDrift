# Federated Learning under Distributed Concept Drift (FedDrift)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository is the source code for our paper: [Federated Learning under Distributed Concept Drift (AISTATS'23)](https://proceedings.mlr.press/v206/jothimurugesan23a/jothimurugesan23a.pdf).

Federated Learning (FL) under distributed concept drift is a largely unexplored area. Although concept drift is itself a well-studied phenomenon, it poses particular challenges for FL, because drifts arise staggered in time and space (across clients).
We first demonstrate that prior solutions to drift adaptation that use a single global model are ill-suited to staggered drifts, necessitating multiple-model solutions. 
We identify the problem of drift adaptation as a time-varying clustering problem, and we propose two new clustering algorithms for reacting to drifts based on local drift detection and hierarchical clustering.
Empirical evaluation shows that our solutions achieve significantly higher accuracy than existing baselines, and are comparable to an idealized algorithm with oracle knowledge of the ground-truth clustering of clients to concepts at each time step.

This repository is built on top of a federated learning research platform, [FedML](https://github.com/FedML-AI/FedML). 

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

