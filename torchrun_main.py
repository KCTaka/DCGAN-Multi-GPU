# main.py
# torchrun --standalone --nproc_per_node=gpu torchrun_main.py
# pip install matplotlib tqdm gdown --upgrade torchvision; torchrun --standalone --nproc_per_node=gpu torchrun_main.py

import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch.distributed as dist
import torch.nn as nn

from model_dcgan import Generator, Discriminator, weights_init
from torchrun_training import Trainer, setup_distributed

def main(dataset, image_size, channels):
    setup_distributed()
    
    # Create models
    D = Discriminator(image_size=image_size, channels=channels).apply(weights_init)
    G = Generator(latent_dim=128, image_size=image_size, channels=channels).apply(weights_init)
    
    # Create trainer
    trainer = Trainer(
        D=D,
        G=G,
        train_dataset=dataset,
        k_steps=1,
        m_steps=1,
        batch_size=128 // world_size,  # Split batch size across GPUs
        lr=2e-4,
        num_epochs=1000,
        save_period=10,
    )
    
    # Start training. If fail, print error and proceed
    try:
        trainer.train()
    except Exception as e:
        print(e)
        print('Failed to train')
    finally:
        # Clean up
        if world_size > 1:
            print('Destroying process group')
            dist.destroy_process_group()

if __name__ == "__main__":
    # Data loading and preprocessing code...
    image_size = 64
    mean = 0.5
    std = 0.5
    
    transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((mean, mean, mean), (std, std, std)),
    ])
    
    data_dir = './data'
    dataset = datasets.LFWPeople(root=data_dir, split='train', transform=transform, download=True)
    image_size = dataset[0][0].shape[-2:]
    channels = dataset[0][0].shape[-3]
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()

    main(dataset, image_size, channels)
