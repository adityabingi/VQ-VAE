# VQ-VAE
This is a minimalistic PyTorch implementation of [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf) (VQ-VAE) for CIFAR10 dataset

# Usage 
Training with default hyperparameters for CIFAR10

`python main.py --train`

For testing on validation data of CIFAR10

`python main.py --test`

For reconstructing sample images from CIFAR10

`python main.py --generate`
