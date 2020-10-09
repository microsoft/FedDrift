#!/bin/bash

set -ex

# code checking
pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb off

# 1. MNIST standalone FedAvg
cd ./fedml_experiments/standalone/fedavg
sh run_fedavg_standalone_pytorch.sh 0 4 4 mnist ./../../../data/mnist lr hetero 1 1 0.03 sgd
sh run_fedavg_standalone_pytorch.sh 0 4 4 shakespeare ./../../../data/shakespeare rnn hetero 1 1 0.8 sgd
sh run_fedavg_standalone_pytorch.sh 0 4 4 femnist ./../../../data/FederatedEMNIST cnn hetero 1 1 0.03 sgd
cd ./../../../


# 2. MNIST distributed FedAvg
cd ./fedml_experiments/distributed/fedavg
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 lr hetero 1 1 2 0.03 mnist "./../../../data/mnist"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 cnn hetero 1 1 10 0.8 femnist "./../../../data/FederatedEMNIST"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 rnn hetero 1 1 10 0.8 shakespeare "./../../../data/shakespeare"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 resnet56 homo 1 1 64 0.001 cifar10 "./../../../data/cifar10"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 resnet56 hetero 1 1 64 0.001 cifar10 "./../../../data/cifar10"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 resnet56 homo 1 1 64 0.001 cifar100 "./../../../data/cifar100"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 resnet56 hetero 1 1 64 0.001 cifar100 "./../../../data/cifar100"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 resnet56 homo 1 1 64 0.001 cinic10 "./../../../data/cinic10"
sh run_fedavg_distributed_pytorch.sh 4 4 1 4 resnet56 hetero 1 1 64 0.001 cinic10 "./../../../data/cinic10"
cd ./../../../

# 3. MNIST mobile FedAvg
cd ./fedml_mobile/server/executor/
python3 app.py &
bg_pid_server=$!
echo "pid="$bg_pid_server

sleep 30
python3 ./mobile_client_simulator.py --client_uuid '0' &
bg_pid_client0=$!
echo $bg_pid_client0

python3 ./mobile_client_simulator.py --client_uuid '1' &
bg_pid_client1=$!
echo $bg_pid_client1

sleep 80
kill $bg_pid_server
kill $bg_pid_client0
kill $bg_pid_client1

cd ./../../../