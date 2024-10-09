#!/bin/bash

mamba install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y

python -m pip install albumentations

python -m pip install pre-commit

# Update mamba and conda
mamba update mamba conda -y

# Install libraries
mamba install timm -c conda-forge -y
mamba install numpy -c conda-forge -y
mamba install pandas -c conda-forge -y
mamba install scipy -c conda-forge -y
mamba install matplotlib -c conda-forge -y
# Install OpenCV and scikit-image
mamba install opencv -c conda-forge -y
mamba install scikit-image -c conda-forge -y
mamba install transformers -c conda-forge -y
mamba install pytorch-lightning -c conda-forge
mamba install ml-collections -c conda-forge
mamba install wandb -c conda-forge

