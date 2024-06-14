#!/bin/bash

# Create a Python virtual environment called 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Update pip to its latest version
pip install --upgrade pip

# Install the required libraries
pip install torchvision torch tensorboardX scikit-learn imageio tqdm IPython seaborn scipy numpy einops timm psutil tensorflow

echo "Python virtual environment setup and libraries installed successfully."