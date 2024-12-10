import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

from dataclasses import dataclass
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Operating_modes_name = [
    "turbine_mode", "equilibrium_turbine_mode", "pump_mode", "equilibrium_pump_mode",
    "short_circuit_mode", "equilibrium_short_circuit_mode", "machine_on", "machine_off", 
    "dyn_only_on", "all"
]

DEFAULT_PARAMS = {
    # Data parameters
    'window_size': 50,
    'batch_size': 256,
    'val_split': 0.2,
    
    # Model architecture
    'kernel_size': 3,
    'use_batchnorm': False,
    
    # Training parameters
    'learning_rate': 5e-4,
    'n_epochs': 100,
    
    # Thresholding
    'n_sigma': 3  # Number of standard deviations for anomaly threshold
}

class SlidingWindowDataset(Dataset):
    def __init__(self, training_df, window=50, stride=1, horizon=1, device='cpu'):
        self.window = window
        self.stride = stride
        self.horizon = horizon
        
        # Store raw data without normalization
        self.X = training_df.to_numpy().astype(np.float32)
        #self.y = sensor_df.values.astype(np.float32)
        
        self.indices = self._get_indices()
        self.indices = torch.from_numpy(self.indices).to(device)

    def _get_indices(self):
        valid_length = len(self.X) - self.window - self.horizon + 1
        idx_list = [i for i in range(0, valid_length, self.stride)]
        return np.asarray([(idx, idx+self.window) for idx in idx_list])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        i_start, i_stop = self.indices[i]
        x = self.X[i_start:i_stop, :]
        #y = self.y[i_stop + self.horizon - 1]
        x = torch.from_numpy(x).transpose(0, 1)
        #y = torch.from_numpy(y)
        return x#, y
    

class CAE(nn.Module):
    def __init__(self, in_features, out_features, params=DEFAULT_PARAMS):
        super(CAE, self).__init__()
        
        # Encoder with increased capacity
        self.encoder = nn.Sequential(
            # First layer - increase channels more gradually
            nn.Conv1d(in_features, 67, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(67),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Conv1d(67, 45, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(45),
            nn.LeakyReLU(0.2),
            
            # Third layer
            nn.Conv1d(45, 22, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(22),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder with matching architecture
        self.decoder = nn.Sequential(
            # First layer
            nn.ConvTranspose1d(22, 45, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(45),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.ConvTranspose1d(45, 67, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(67),
            nn.LeakyReLU(0.2),
            
            # Final layer
            nn.ConvTranspose1d(67, out_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Ensure output is bounded between 0 and 1
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CAETrainer:
    def __init__(self, model, optimizer, device='cpu', model_name='cae_model'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.MSELoss()
        self.model_name = model_name
        self.losses = {'train': [], 'val': []}
        self.best_val_loss = float('inf')
        self.reconstruction_errors = None
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        with tqdm(train_loader, desc='Training', leave=False) as pbar:
            for x in pbar:
                batch_size = x.shape[0]
                total_samples += batch_size
                x = x.to(self.device)
                
                self.optimizer.zero_grad()

                output = self.model(x)
                loss = self.criterion(output, x)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * batch_size  # Multiply by batch size
                pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})
        
        return total_loss / total_samples  # Divide by total number of samples

    @torch.no_grad()
    def validate(self, val_loader, save_reconstruction_errors=False, save_path='models'):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        reconstruction_errors = []
        
        with tqdm(val_loader, desc='Validating', leave=False) as pbar:
            for x in pbar:
                batch_size = x.shape[0]
                total_samples += batch_size
                x = x.to(self.device)
                
                output = self.model(x)
                loss = self.criterion(output, x)
                
                total_loss += loss.item() * batch_size  # Multiply by batch size
                
                errors = torch.mean((output - x) ** 2, dim=(1, 2))
                reconstruction_errors.extend(errors.cpu().numpy())
                
                pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

        return total_loss / total_samples, np.array(reconstruction_errors) 
    
    def save_model(self, path, is_best=False):
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'best_val_loss': self.best_val_loss,
            'params': getattr(self.model, 'params', DEFAULT_PARAMS),  # Use DEFAULT_PARAMS as fallback
            'model_config': {
                'in_features': next(self.model.encoder.parameters()).shape[1],
                'out_features': list(self.model.decoder.parameters())[-1].shape[0]
            },
            'reconstruction_stats': {
                'mean': np.mean(self.reconstruction_errors) if hasattr(self, 'reconstruction_errors') else None,
                'std': np.std(self.reconstruction_errors) if hasattr(self, 'reconstruction_errors') else None
            }
        }
        
        if is_best:
            torch.save(model_info, f"{path}/best_{self.model_name}.pth")
        else:
            torch.save(model_info, f"{path}/latest_{self.model_name}.pth")
    
    def train(self, train_loader, val_loader, n_epochs, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training CAE for {n_epochs} epochs...")
        with tqdm(range(n_epochs), desc='Epochs') as pbar:
            for epoch in pbar:
                train_loss = self.train_epoch(train_loader)
                val_loss, reconstruction_errors = self.validate(
                    val_loader,
                    save_reconstruction_errors=(epoch == n_epochs - 1),  # Save at the last epoch
                    save_path=save_dir
                )
                
                self.losses['train'].append(train_loss)
                self.losses['val'].append(val_loss)
                
                # Update progress bar with metrics
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}'
                })
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    # Store reconstruction errors from best validation performance
                    self.reconstruction_errors = reconstruction_errors
                    self.save_model(save_dir, is_best=True)
                    print("\nNew best model saved!")
                
                # Save latest model
                self.save_model(save_dir)
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
        # Print final reconstruction statistics
        print("\nFinal reconstruction statistics:")
        print(f"Mean: {np.mean(self.reconstruction_errors):.6f}")
        print(f"Std: {np.std(self.reconstruction_errors):.6f}")
        
        return self.reconstruction_errors
    
    def plot_training_curves(self, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses['train'], label='Train Loss')
        plt.plot(self.losses['val'], label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/training_curves.pdf")
        plt.close()

def compute_reconstruction_error(model, x, n_control_features):
    """Compute reconstruction error for sensor features only"""
    with torch.no_grad():
        # Get sensor part of input
        sensor_input = x[:, n_control_features:, :]
        # Get reconstruction
        reconstruction = model(x)
        # Compute MSE
        error = torch.mean((reconstruction - sensor_input) ** 2, dim=(1, 2))
    return error

def calculate_threshold(reconstruction_errors, n_sigma=3):
    """Calculate threshold for anomaly detection"""
    mean = torch.mean(reconstruction_errors)
    std = torch.std(reconstruction_errors)
    return mean + n_sigma * std

def detect_anomalies(reconstruction_errors, threshold):
    """Detect anomalies based on reconstruction error and threshold"""
    return (reconstruction_errors > threshold).float()


def create_data_loaders_with_indices(dataset, batch_size=256, val_split=0.2):
    """Modified version that returns indices for proper normalization"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    print("\nDiagnostic Information:")
    print(f"Total dataset size: {dataset_size}")
    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    
    return train_loader, val_loader, train_indices, val_indices


def main(unit="VG5",mode='turbine'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n=== Loading Training dataframe ===")
    training_df = pd.read_parquet(f"./data/training_{unit}_{mode}.parquet")

    print("\n=== Creating Initial Dataset for Split ===")
    initial_dataset = SlidingWindowDataset(
        training_df, 
        window=DEFAULT_PARAMS['window_size']
    )
    
    print("\n=== Splitting data ===")
    # Create final data loaders
    train_loader, val_loader = create_data_loaders_with_indices(
        initial_dataset, 
        batch_size=DEFAULT_PARAMS['batch_size'], 
        val_split=DEFAULT_PARAMS['val_split']
    )[:2]
    
    # Check actual data going into model
    print("\n=== Checking Model Input Data ===")
    x_batch = next(iter(train_loader))
    print(f"Sample batch shape: {x_batch.shape}")
    print(f"Sample batch range: [{x_batch.min().item():.6f}, {x_batch.max().item():.6f}]")
    
    print("\n=== Initializing Model ===")
    n_features_in = len(training_df.columns)
    n_features_out = len(training_df.columns)
    
    model = CAE(
        in_features=n_features_in,
        out_features=n_features_out,
        params=DEFAULT_PARAMS
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=DEFAULT_PARAMS['learning_rate']
    )

    trainer = CAETrainer(model, optimizer, device=device, model_name=f'cae_model_{unit}_{mode}')

    print("\n=== Starting Training ===")
    reconstruction_errors = trainer.train(
        train_loader, 
        val_loader, 
        n_epochs=DEFAULT_PARAMS['n_epochs']
    )
    
    # Save everything
    print("\n=== Saving Models and Statistics ===")
    os.makedirs('models', exist_ok=True)
    
    error_stats = {
        'mean': np.mean(reconstruction_errors),
        'std': np.std(reconstruction_errors)
    }
    
    with open('models/reconstruction_error_stats_cae.pkl', 'wb') as f:
        pickle.dump(error_stats, f)

    print("\nAll files saved in 'models' directory")

if __name__ == "__main__":
    main(unit="VG5", mode='turbine')