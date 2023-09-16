#!/bin/bash

# Create conda environment with python 3.9
conda create -n pygeo45 python=3.9 -y

# Activate the environment
conda activate pygeo45

# Install necessary packages
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge scipy scikit-learn -y
pip install torch-geometric ogb

echo "Environment 'pygeo39' setup and packages installed."
