import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from traincae import DEFAULT_PARAMS, SlidingWindowDataset
from findAnomalies import load_model, reconstruct_signal_from_windows, compute_error
from loadData import RawDataset, preprocess_hydraulic_data

dataset_path = Path("../dataset")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

operating_conditions_set_turbine = {
    "dyn_only_on": False, "turbine_mode": True, "equilibrium_turbine_mode": True, 
    #"short_circuit_mode": False # Comment if VG4
}

operating_conditions_set_pump = {
    "dyn_only_on": False, "pump_mode": True, "equilibrium_pump_mode": True, 
    #"short_circuit_mode": False # Comment if VG4
}


def main(unit, mode):
    training_df = pd.read_parquet(f'data/training_{unit}_{mode}.parquet')

    scaler, _, model = load_model(unit = unit, mode = mode)


    rds = RawDataset(dataset_path, unit, load_synthetic=False, load_training=False) 

    VG4_bool=False
    if unit=="VG4":
        VG4_bool = True

    if mode == "pump":
        operating_conditions = operating_conditions_set_pump
    elif mode == "turbine":
        operating_conditions = operating_conditions_set_turbine

    test_df = preprocess_hydraulic_data(rds= rds, 
                                        set_type="test",
                                        operating_conditions = operating_conditions,
                                        scaler=scaler, fit=False, VG4=VG4_bool)

    # Create dataset for evaluation
    test_dataset = SlidingWindowDataset(
        test_df,
        window=DEFAULT_PARAMS['window_size'],
        device=device
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=DEFAULT_PARAMS['batch_size'],
        shuffle=False  # Important: keep order for anomaly detection
    )

    reconstructed_test_df = reconstruct_signal_from_windows(test_loader, test_df, model, device)

    error_test = compute_error(test_df,reconstructed_test_df)

    dir_path = Path(f"./reconstructed/test_{unit}_{mode}")

    os.makedirs(dir_path,exist_ok=True)
    test_df.to_parquet(dir_path/f'original_{unit}_{mode}.parquet')
    reconstructed_test_df.to_parquet(dir_path/f"reconstructed_{unit}_{mode}.parquet")
    error_test.to_parquet(dir_path/f"error_{unit}_{mode}.parquet")


if __name__ == "__main__":
    units = ['VG4']
    modes = ["pump", 'turbine']

    for unit in tqdm(units, desc='Units'):
        for mode in tqdm(modes, desc='Modes', leave=False):  # leave=False removes the progress bar after the inner loop
            main(unit=unit, mode=mode)    
