import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from traincae import DEFAULT_PARAMS, SlidingWindowDataset
from findAnomalies import load_model, reconstruct_signal_from_windows
from random import randint
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_signal(original_df,reconstructed_df,unit,mode):

    # Picking a column randomly
    ind = randint(0,len(original_df.columns)-1)
    column = original_df.columns[ind]
    #column='charge'
    plt.figure(figsize=(10, 6))
    plt.plot(reconstructed_df[column][100:1000], label=f'Reconstructed Signal of {column}', color='b', alpha=0.6)
    plt.plot(original_df[column][100:1000], label="Original", color='r', linestyle='--', alpha=0.6)
    plt.xlabel('Time index')
    plt.ylabel('Value')
    plt.ylim(-0.05,1.05)
    plt.title(f'Reconstructed Signal of {column}')
    plt.grid()
    plt.legend()

    plt.savefig(f"../images/reconstruction_training_{unit}_{mode}.pdf")


def main(unit,mode):
    print(f"\n=== Loading Training {unit}-{mode} ===")
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
    print("\n=== Loading Model ===")
    _, _, model = load_model(unit = unit, mode = mode)

    print(f"\n=== Reconstructin {unit}-{mode} ===")
    reconstructed_test_df = reconstruct_signal_from_windows(train_loader, training_df, model, device)

    print("\n=== Creating the plot ===")
    plot_signal(original_df=training_df, 
                reconstructed_df=reconstructed_test_df,
                unit=unit,
                mode=mode)


if __name__ == "__main__":
    units = ['VG4',"VG5",'VG6']
    modes = ["pump", 'turbine']

    for unit in tqdm(units, desc='Units'):
        for mode in tqdm(modes, desc='Modes', leave=False):  # leave=False removes the progress bar after the inner loop
            main(unit,mode)