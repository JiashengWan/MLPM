import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
from traincae import DEFAULT_PARAMS, CAE, SlidingWindowDataset
import matplotlib.pyplot as plt
import seaborn as sns

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

def read_threshold(path):
    return pd.read_csv(path, index_col=0)["0"]

def load_model(unit, mode, DEFAULT_PARAMS = DEFAULT_PARAMS):

    model_path = base_path / f"models_{unit}_{mode}/best_cae_model_{unit}_{mode}.pth"
    threshold_path = base_path / f"models_{unit}_{mode}/threshold_{unit}_{mode}.csv"

    # Load scaler_control.pkl from the data folder
    with open(f'data/scaler_{unit}_{mode}.pkl', 'rb') as file:
        scaler = joblib.load(file)

    # Load threshold
    threshold_df = read_threshold(threshold_path)

    # Load model and parameters
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load parameters directly from checkpoint
    DEFAULT_PARAMS = checkpoint['params']  # This contains all the parameters used during training

    # Loading the training df
    training_df = pd.read_parquet(f'data/training_{unit}_{mode}.parquet')

    # Load your model
    model = CAE(
        in_features=len(training_df.columns),
        out_features=len(training_df.columns),
        params=DEFAULT_PARAMS
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return scaler, threshold_df, model

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

def compute_error(original_df, reconstructed_df):
    return np.abs(original_df- reconstructed_df).dropna()

def keep_consecutive_true(series, N):
    """
    Keeps only the consecutive True values in the series if they appear for at least N times.
    
    Parameters:
    - series: A Pandas boolean Series
    - N: The minimum length of consecutive True values to keep
    
    Returns:
    - A Pandas Series where only consecutive True values of length >= N are kept, others are set to False
    """
    # Create an empty series with the same index
    result = pd.Series(False, index=series.index)
    
    # Find the consecutive True values
    consecutive_count = 0
    for i in range(len(series)):
        if series.iloc[i]:  # If current value is True
            consecutive_count += 1
        else:
            consecutive_count = 0
        
        # Keep the True value if the consecutive count is >= N
        if consecutive_count >= N:
            result.iloc[i] = True
    
    return result

def filter_faulty_df(faulty_df,N=130):
    for column in tqdm(faulty_df.columns, desc="Processing columns"):
        faulty_df[column] = keep_consecutive_true(faulty_df[column],N=N)
    return faulty_df

def get_faulty_signals(faulty_df):
    return faulty_df.sum().sort_values(ascending=False)

def get_indexes_fault(faulty_df, column):
    i=0
    ind=[]
    for bool in faulty_df["stat_coil_ph03_05_tmp"].values:
        if bool==True:
            ind.append(i)
        i+=1
    return ind

def correlation_faulty_signals(faulty_df, filtered_columns, save_path=None):

    corr_map = faulty_df[filtered_columns].corr()

    f, ax = plt.subplots(1, 1, figsize=(8, 6)) # 8,6 - 16,14
    sns.heatmap(corr_map, 
                cmap="vlag")
    plt.title("Heatmap of the correlation of the faulty signals")

    if save_path:
        plt.savefig(save_path)

    return corr_map

def find_unique_correlations(dataframe, threshold):
    df = dataframe[dataframe>threshold]
    unique_correlations = set()
    for col in dataframe.columns:
        for idx in dataframe.index:
            if col != idx and not np.isnan(df.at[idx, col]):
                unique_correlations.add(idx)
                unique_correlations.add(col)
    return list(unique_correlations)
