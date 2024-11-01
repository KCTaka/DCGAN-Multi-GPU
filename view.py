import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import torch

def view_tensor_image(tensor_image, black_white=False, normalized=False):
    if isinstance(tensor_image, torch.Tensor) and black_white:
        np_image = tensor_image.cpu().detach().numpy()
        np_image = np_image.squeeze()
        plt.imshow(np_image, cmap='gray')
        plt.show()
        return
    
    if isinstance(tensor_image, torch.Tensor) and normalized:
        mean = 0.5
        std = 0.5
        np_image = tensor_image.cpu().detach().numpy()
        np_image = np_image * std + mean
    
    elif isinstance(tensor_image, torch.Tensor):
        np_image = tensor_image.cpu().detach().numpy()
    elif isinstance(tensor_image, np.ndarray):
        np_image = tensor_image
    else:
        raise ValueError('Not supported type')
    
    np_image = np.transpose(np_image, (1, 2, 0))

    plt.imshow(np_image)
    plt.show()
    
def save_tensor_gif(tensor_video, output_dir):
    parent_dir = os.path.dirname(output_dir)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    # Convert tensor_video into list of numpy image matrix
    video = tensor_video.cpu().detach().numpy()
    video = video * 255
    video = video.astype(np.uint8)
    
    frames = [Image.fromarray(frame) for frame in video]
    frames[0].save(output_dir, save_all=True, append_images=frames[1:], loop=0, duration=1000/30)
    
# self.history = {'d_loss': [], 'g_loss': [], 'epoch': []}
def view_history(history):
    plt.plot(history['epoch'], history['d_loss'], label='Discriminator Loss')
    plt.plot(history['epoch'], history['g_loss'], label='Generator Loss')
    plt.legend()
    plt.show()
    