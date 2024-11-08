# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys
import os
# 
# Check if the code is running in an IPython environment
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def setup_distributed():
    dist.init_process_group(backend="nccl")

class Trainer(): 
    def __init__(self, D, G, train_dataset, k_steps, m_steps, batch_size, lr=0.0002, num_epochs=100, save_period=10, checkpoints_dir='checkpoints/latest.pth'):
        self.rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        
        # Move models to the correct device
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        self.D = D.to(self.device)
        self.G = G.to(self.device)
        
        # Wrap models with DDP
        if self.world_size > 1:
            self.D = DDP(self.D, device_ids=[self.rank])
            self.G = DDP(self.G, device_ids=[self.rank])
        
        self.k = k_steps
        self.m = m_steps
        self.num_epochs = num_epochs
        self.latent_dim = self.G.module.latent_dim if self.world_size > 1 else G.latent_dim
        
        # Set up distributed sampler and dataloader
        self.train_sampler = DistributedSampler(train_dataset) if self.world_size > 1 else None
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        
        beta1 = 0.5
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))
        
        self.criterion = nn.BCELoss()
        
        self.best_d_loss = float('inf')
        self.best_g_loss = float('inf')
        self.history = {'d_loss': [], 'g_loss': [], 'epoch': []}
        self.start_epoch = 0
        self.save_period = save_period
        
        if os.path.exists(checkpoints_dir):
            print(f'Loading checkpoint: {checkpoints_dir}')
            self.load_checkpoint(checkpoints_dir)
        
    def activate_model_params(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad
    
    def save_checkpoint(self, epoch, g_loss, d_loss, is_best=False, save_period=10):
        if self.rank != 0 and (epoch+1) % save_period != 0 and not is_best:
            return
            
        checkpoint = {
            'epoch': epoch,
            'g_state_dict': self.G.module.state_dict() if self.world_size > 1 else self.G.state_dict(),
            'd_state_dict': self.D.module.state_dict() if self.world_size > 1 else self.D.state_dict(),
            'g_loss': g_loss,
            'd_loss': d_loss,
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'history': self.history,
            'best_g_loss': self.best_g_loss,
            'best_d_loss': self.best_d_loss,
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        
        if (epoch+1) % save_period == 0:
            #torch.save(checkpoint, f'checkpoints/epoch_{epoch+1}.pth')
            torch.save(checkpoint, 'checkpoints/latest.pth')
            #torch.save(checkpoint, 'latest.pth')
        
        if is_best:
            torch.save(checkpoint, 'checkpoints/best.pth')
            
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint file not found: {checkpoint_path}')
            return False
        
        if self.rank == 0:
            print(f'Loading checkpoint: {checkpoint_path}')
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.world_size > 1:
            self.G.module.load_state_dict(checkpoint['g_state_dict'])
            self.D.module.load_state_dict(checkpoint['d_state_dict'])
        else:
            self.G.load_state_dict(checkpoint['g_state_dict'])
            self.D.load_state_dict(checkpoint['d_state_dict'])
        
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_g_loss = checkpoint['best_g_loss']
        self.best_d_loss = checkpoint['best_d_loss']
        self.history = checkpoint['history']
        
        if self.rank == 0:
            print(f'Checkpoint loaded. Resuming training from epoch {self.start_epoch}')
        return True
    
    def train(self):                
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), desc='Epochs', disable=self.rank != 0):
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
                
            total_d_loss = []
            total_g_loss = []
            
            k_count = 0
            
            for real_images, _ in tqdm(self.dataloader, desc='Batches', leave=False, disable=self.rank != 0):
                k_count = (k_count + 1) % self.k

                self.D.zero_grad()
                
                real_images = real_images.to(self.device)
                batch_len = real_images.size(0)
                noise = torch.randn(batch_len, self.latent_dim).to(self.device)
                fake_images = self.G(noise)
                    
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                fake_outputs = self.D(fake_images)
                real_outputs = self.D(real_images)
                
                fake_labels = torch.zeros_like(fake_outputs).to(self.device)
                real_labels = torch.ones_like(real_outputs).to(self.device)
                
                loss_fake = self.criterion(fake_outputs, fake_labels)
                loss_real = self.criterion(real_outputs, real_labels)
                d_loss = loss_fake + loss_real
                d_loss.backward()
                self.d_optimizer.step()
                
                # Synchronize loss across GPUs
                if self.world_size > 1:
                    dist.all_reduce(d_loss)
                    d_loss /= self.world_size
                
                total_d_loss.append(d_loss.item())
                    
                if k_count != 0:
                    continue
                
                for _ in range(self.m):
                    self.G.zero_grad()
                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()
                    noise = torch.randn(batch_len, self.latent_dim).to(self.device)
                    fake_images = self.G(noise)
                    outputs = self.D(fake_images)
                    g_loss = self.criterion(outputs, torch.ones_like(outputs))
                    
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    # Synchronize loss across GPUs
                    if self.world_size > 1:
                        dist.all_reduce(g_loss)
                        g_loss /= self.world_size
                    
                    total_g_loss.append(g_loss.item())
            
            epoch_d_loss = sum(total_d_loss)/len(total_d_loss)
            epoch_g_loss = sum(total_g_loss)/len(total_g_loss)
            
            if self.rank == 0:
                self.history['d_loss'].append(epoch_d_loss)
                self.history['g_loss'].append(epoch_g_loss)
                self.history['epoch'].append(epoch)
                
                is_best = False
                if epoch_g_loss < self.best_g_loss:
                    self.best_d_loss = epoch_d_loss
                    self.best_g_loss = epoch_g_loss
                    is_best = True
                    
                self.save_checkpoint(epoch, epoch_g_loss, epoch_d_loss, is_best=is_best, save_period=self.save_period)
                
                tqdm.write(f'Epoch {epoch+1}, Discriminator Loss: {epoch_d_loss}, Generator Loss: {epoch_g_loss}')
