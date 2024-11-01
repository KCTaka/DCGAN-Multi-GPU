import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding = 0,dilation=1, groups=1, bias=True, conv_trans = False, leaky=0.2):
        super(Block, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias) if conv_trans else nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(leaky) if leaky else nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size=(128, 128), channels = 3):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.channels = channels
        
        num_blocks = torch.log2(torch.tensor(min(image_size))).int().item() - 3
        max_channel = 512
        layers = [
            nn.Conv2d(channels, max_channel//(2**num_blocks), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        ]
        for i in range(num_blocks):
            layers.append(Block(max_channel//2**(num_blocks-i), max_channel//2**(num_blocks-i-1), 4, 2, 1, bias=False, leaky=0.2))
            
        layers += [nn.Conv2d(max_channel, 1, 4, 1, 0, bias=False), nn.Sigmoid()]
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x.view(x.size(0), -1)
        return x
        
class Generator(nn.Module):
    def __init__(self, latent_dim=128, image_size=(128, 128), channels = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size    
        self.channels = channels
        
        num_blocks = torch.log2(torch.tensor(min(image_size))).int().item() - 3
        max_channel = 512
        layers = [Block(latent_dim, max_channel, 4, 1, 0, bias=False, conv_trans = True, leaky=False)]
        for i in range(num_blocks):
            layers.append(Block(max_channel//2**i, max_channel//2**(i+1), 4, 2, 1, output_padding=0, bias=False, conv_trans = True, leaky=False))
            
        layers += [nn.ConvTranspose2d(max_channel//(2**num_blocks), channels, 4, 2, 1, output_padding=0, bias=False), nn.Tanh()]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.model(x)
        return x

if __name__ == "__main__":
    D = Discriminator(image_size=(128, 128)).apply(weights_init)
    G = Generator(128, image_size=(128, 128)).apply(weights_init)
    
    random_image = torch.randn(5, 3, 128, 128)
    print(D(random_image).size())
    
    random_noise = torch.randn(5, 128)
    print(G(random_noise).size())
    
    
    
    

    
    