## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments

## MNIST experiments
```
sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0

##run on background
nohup sh run_fedavg_distributed_pytorch.sh 10 10 1 4 lr hetero 200 20 10 0.03 mnist "./../../../data/mnist" 0 > ./fedavg-lr-mnist.txt 2>&1 &
```
