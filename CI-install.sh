#!/bin/bash
set -ex

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda config --set always_yes yes #--set changeps1 no
conda update -q conda
conda info -a

echo "conda create -n feddrift python=3.7.4"
conda create -n feddrift python=3.7.4

echo "conda activate feddrift"
conda activate feddrift

# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install MPI
conda install -c anaconda mpi4py

# Install Wandb
pip install --upgrade wandb

# Install other required package
conda install scikit-learn
conda install numpy
conda install h5py
conda install setproctitle
conda install networkx

cd ./fedml_mobile/server/executor
pip install -r requirements.txt
cd ./../../../

# Install wilds package (still have to download the dataset separately)
pip install wilds
