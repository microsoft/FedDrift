# Federated Learning in Non-Staionary Environment

Federated learning is a promising machine learning paradigm that enables collaborative learning while preserving data privacy. Prior work has been focused on efficient and robust federated learning algorithm based on a fixed amount of training data. However, real-world data is generated continuously and training data can drift over time. Using obsolete training data will generate ML models that perform poorly on future data. This problem is particularly challenging for federated learning as different clients may have different concepts at different time, and there may not be a one-size-fit-all ML model that works for all clients. In this project, we aim to understand the problem of federated learning under concept drift over time. Our goals are (1) making further theoretical understanding of this problem; and (2) designing effective algorithms to address this challenge.

This repository is built on top of a federated learning research platform, [FedML](https://github.com/FedML-AI/FedML). 

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

