import torch
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm
from traincae import DEFAULT_PARAMS, CAE, SlidingWindowDataset


operating_conditions_set_turbine = {
    "dyn_only_on": False, "turbine_mode": True, "equilibrium_turbine_mode": True, "short_circuit_mode": False
}

operating_conditions_set_pump = {
    "dyn_only_on": False, "pump_mode": True, "equilibrium_pump_mode": True, "short_circuit_mode": False
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Your existing paths
base_path = Path("./models")
dataset_path = Path("../dataset")


def load_training_model(unit, mode, DEFAULT_PARAMS = DEFAULT_PARAMS):

    model_path = base_path / f"models_{unit}_{mode}/best_cae_model_{unit}_{mode}.pth"

    # Load model and parameters
    checkpoint = torch.load(model_path, map_location=device)
    DEFAULT_PARAMS = checkpoint['params']


    # Load parameters directly from checkpoint
    DEFAULT_PARAMS = checkpoint['params']  # This contains all the parameters used during training

    # Loading the training df
    training_df = pd.read_parquet(f'data/training_{unit}_{mode}.parquet')

    # Create dataset for evaluation
    train_dataset = SlidingWindowDataset(
        training_df,
        window=DEFAULT_PARAMS['window_size'],
        device=device
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFAULT_PARAMS['batch_size'],
        shuffle=False  # Important: keep order for anomaly detection
    )

    # Load your model
    model = CAE(
        in_features=len(training_df.columns),
        out_features=len(training_df.columns),
        params=DEFAULT_PARAMS
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return training_df, train_loader, model


def reconstruct_signal_from_windows(dataloader, df, model, device=device):
    """
    Reconstruct the original signals from overlapping windows using the average method.

    Parameters:
    - dataloader: DataLoader object that provides the windows
    - model: Trained PyTorch model for reconstruction
    - device: The device to run the model on (e.g., 'cpu' or 'cuda')

    Returns:
    - reconstructed_signal: The reconstructed signals of shape (num_timesteps, num_channels)
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device
    
    # Get the original data shape from the first batch
    for batch in dataloader:
        _, num_channels, window_size = batch.shape
        break
    
    # Initialize an array to store the reconstructed signals and a count array for averaging
    num_windows = len(dataloader.dataset)
    num_timesteps = (num_windows - 1) + window_size
    reconstructed_signal = np.zeros((num_timesteps, num_channels), dtype=np.float32)
    count = np.zeros((num_timesteps, num_channels), dtype=np.int32)
    
    start_idx = 0

    # Collect the reconstructed windows and accumulate them
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Reconstructing signals"):
            batch = batch.to(device)
            output = model(batch)
            
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            batch_size = output.shape[0]
            for i in range(batch_size):
                end_idx = start_idx + window_size
                reconstructed_signal[start_idx:end_idx, :] += output[i].T
                count[start_idx:end_idx, :] += 1
                start_idx += 1

    # Average the overlapping segments
    reconstructed_signal /= count
    
    return pd.DataFrame(reconstructed_signal,columns=df.columns, index=df.index[:-1])


def computing_threshold(training_df, reconstructed_df, save_path ,quantile=0.995):
    
    error = np.abs(training_df- reconstructed_df).dropna()
    threshold_df = pd.Series(np.quantile(error,quantile,axis=0), index=error.columns)

    threshold_df.to_csv(save_path)


def read_threshold(path):
    return pd.read_csv(path, index_col=0)["0"]

def main(unit, mode, quantile=0.995):

    print(f"\n=== Working on {unit} in {mode} mode ===")

    print(f"\n=== Loading the model and the dataset ===")
    training_df, train_loader, model = load_training_model(unit, mode, DEFAULT_PARAMS)

    print(f"\n=== Reconstructing the dataset ===")
    reconstructed_df = reconstruct_signal_from_windows(dataloader=train_loader, df=training_df, 
                                                     model=model)
    
    print(f"\n=== Reconstructing the dataset ===")
    save_path = base_path / f"models_{unit}_{mode}/threshold_{unit}_{mode}.csv"
    computing_threshold(training_df, reconstructed_df, save_path ,quantile=quantile)

if __name__ == "__main__":
    units = ['VG4','VG5','VG6']
    modes = ["pump", 'turbine']

    for unit in tqdm(units, desc='Units'):
        for mode in tqdm(modes, desc='Modes', leave=False):  # leave=False removes the progress bar after the inner loop
            main(unit=unit, mode=mode)