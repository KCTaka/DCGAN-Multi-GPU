# DCGAN training for Multi-GPU

This project implements a model based on the paper Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks and the original GAN paper by Ian Goodfellow et al. The code includes `torchrun` capabilities for distributed training on a single cluster.

## Prerequisites

Ensure you have the following libraries installed:

- `matplotlib`
- `tqdm`
- `gdown`
- `torchvision`
- `torch`

You can install these libraries using pip:

```bash
pip install matplotlib tqdm gdown torchvision torch
```

## Running the Code

To run the code, use the following command:

```bash
torchrun --standalone --nproc_per_node=gpu torchrun_main.py
```

## Hyperparameters

All hyperparameters and values must be edited directly in the code. Future updates will include a more flexible configuration system.

## References

- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- Generative Adversarial Networks by Ian Goodfellow et al.
