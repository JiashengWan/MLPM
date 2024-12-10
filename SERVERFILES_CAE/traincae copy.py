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

@dataclass
class Case():
    info: pd.DataFrame
    measurements: pd.DataFrame

class RawDataset():

    def __init__(self, root, unit = "VG4", load_training=False, load_synthetic=False) -> None:
        
        read_pq_file = lambda f: pq.read_table(root / f).to_pandas()
        
        cases = {
            "test": [f"{unit}_generator_data_testing_real_measurements.parquet", root / f"{unit}_generator_data_testing_real_info.csv" ], 
        }
        
        if load_training:
            cases = {
                **cases,
                "train": [f"{unit}_generator_data_training_measurements.parquet", root / f"{unit}_generator_data_training_info.csv" ], 
            }
        
        if load_synthetic:
            cases = {
                **cases,
                "test_s01": [f"{unit}_generator_data_testing_synthetic_01_measurements.parquet", root / f"{unit}_generator_data_testing_synthetic_01_info.csv"], 
                "test_s02": [f"{unit}_generator_data_testing_synthetic_02_measurements.parquet", root / f"{unit}_generator_data_testing_synthetic_02_info.csv"]
            }
        
        
        self.data_dict = dict()
        
        for id_c, c in cases.items():
            # if you need to verify the parquet header:
            # pq_rows = RawDataset.read_parquet_schema_df(root / c[0])
            info = pd.read_csv(c[1])
            measurements = read_pq_file(c[0])
            self.data_dict[id_c] = Case(info, measurements)
            
        
        
    @staticmethod
    def read_parquet_schema_df(uri: str) -> pd.DataFrame:
        """Return a Pandas dataframe corresponding to the schema of a local URI of a parquet file.

        The returned dataframe has the columns: column, pa_dtype
        """
        # Ref: https://stackoverflow.com/a/64288036/
        schema = pq.read_schema(uri, memory_map=True)
        schema = pd.DataFrame(({"column": name, "pa_dtype": str(pa_dtype)} for name, pa_dtype in zip(schema.names, schema.types)))
        schema = schema.reindex(columns=["column", "pa_dtype"], fill_value=pd.NA)  # Ensures columns in case the parquet file has an empty dataframe.
        return schema
    pass

class SlidingWindowDataset(Dataset):
    def __init__(self, control_df, sensor_df, window=50, stride=1, horizon=1, device='cpu'):
        self.window = window
        self.stride = stride
        self.horizon = horizon
        
        # Store raw data without normalization
        self.X_control = control_df.values.astype(np.float32)
        self.X_sensor = sensor_df.values.astype(np.float32)
        self.X = np.concatenate([self.X_control, self.X_sensor], axis=1)
        self.y = sensor_df.values.astype(np.float32)
        
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
        y = self.y[i_stop + self.horizon - 1]
        x = torch.from_numpy(x).transpose(0, 1)
        y = torch.from_numpy(y)
        return x, y
    
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

def extracting_name_groups_time_series(rds, set_type):
        df_name_control_sensors = rds.data_dict[set_type].info
        control_time_series_name = df_name_control_sensors[
            (df_name_control_sensors['control_signal'] == True) &
            (df_name_control_sensors["signal_type"] == "Measurement")
        ]
        sensors_time_series_name = df_name_control_sensors[
            (df_name_control_sensors['control_signal'] == False) &
            (df_name_control_sensors["signal_type"] == "Measurement")
        ]
        return control_time_series_name["attribute_name"], sensors_time_series_name["attribute_name"]

def extracting_groups_time_series(rds, set_type):
    control_time_series_name, sensors_time_series_name = extracting_name_groups_time_series(rds, set_type=set_type)
    control_df = rds.data_dict[set_type].measurements[control_time_series_name]
    sensors_df = rds.data_dict[set_type].measurements[sensors_time_series_name]
    operating_modes_df = rds.data_dict[set_type].measurements[Operating_modes_name]
    return control_df, sensors_df, operating_modes_df

def get_indexes_operating_mode(operating_modes_df, operating_conditions):
    mask = pd.Series([True] * len(operating_modes_df), index=operating_modes_df.index)
    for column, value in operating_conditions.items():
        mask &= (operating_modes_df[column] == value)
    return operating_modes_df[mask].index

def feature_extraction(df, feature_prefixes, new_feature_name):
    feature_columns = df.columns[df.columns.str.startswith(tuple(feature_prefixes))]
    df[new_feature_name] = df[feature_columns].sum(axis=1)
    df.drop(columns=feature_columns, inplace=True)
    return df

def normalize_features(df,scaler, fit = True):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns]) if fit else scaler.transform(df[numerical_columns])
    return df

def dynamic_group_features(df):
    feature_groups = {
        'voltage_current': df.columns[df.columns.str.startswith(("exc", "ph", "elec", "mid", "neutral"))].tolist(),
        'stator_temperature': df.columns[df.columns.str.startswith("stat_") & 
                                        df.columns.str.contains("coil|magn")].tolist(),
        'cooling_system': df.columns[df.columns.str.startswith("air_circ_")].tolist(),
        'heat_exchanger': df.columns[df.columns.str.startswith("water_circ_")].tolist()
    }

    # Flatten the list of all grouped features
    grouped_features = [feature for group in feature_groups.values() for feature in group]
    # Find complementary features
    complementary_features = [col for col in df.columns if col not in grouped_features]
    
    feature_groups['complementary'] = complementary_features

    return feature_groups

def preprocess_hydraulic_data(rds, set_type, operating_conditions, scaler_control, scaler_sensors, fit):
    """
    Preprocess the hydraulic site data to return control and sensor signals filtered by operating conditions.
    
    Parameters:
    - rds: RawDataset object containing the data dictionary
    - set_type: Type of dataset to use ('train', 'test', etc.)
    - operating_conditions: A dictionary specifying operating conditions to filter the data
    - scaler_control: Scaler for control signals (if None, no normalization is performed)
    - scaler_sensors: Scaler for sensor signals (if None, no normalization is performed)
    - fit: Whether to fit the scalers (ignored if scalers are None)
    """
    # Step 1: Extract control, sensor, and operating mode time series
    control_df, sensors_df, operating_modes_df = extracting_groups_time_series(rds, set_type=set_type)

    # Step 2: Filter based on operating conditions if specified
    if operating_conditions:
        indexes = get_indexes_operating_mode(operating_modes_df, operating_conditions)
        control_df = control_df.loc[indexes]
        sensors_df = sensors_df.loc[indexes]
        
    # Select features containing both 'water' and 'opening' in their name
    water_opening_features = [col for col in control_df.columns if "water" in col and "opening" in col]
    control_df = feature_extraction(control_df, water_opening_features, "water_opening_sum")

    # Select features containing both 'injector' and 'opening' in their name
    injector_opening_features = [col for col in control_df.columns if "injector" in col and "opening" in col]
    control_df = feature_extraction(control_df, injector_opening_features, "injector_opening_sum")

    # Step 4: Normalize only if scalers are provided
    if scaler_control is not None:
        control_df = normalize_features(control_df, scaler=scaler_control, fit=fit)
    if scaler_sensors is not None:
        sensors_df = normalize_features(sensors_df, scaler=scaler_sensors, fit=fit)

    # Step 5: Select only the sensor features corresponding to the specified feature group
    grouped_data = dynamic_group_features(sensors_df)
    selected_sensor_data = []
    for key in grouped_data.keys():
        selected_sensor_data.append(sensors_df[grouped_data[key]])

    return {
        'control_signals': control_df,
        'sensor_signals': selected_sensor_data
    }

class CAE(nn.Module):
    def __init__(self, in_features, out_features, params=DEFAULT_PARAMS):
        super(CAE, self).__init__()
        
        # Encoder with increased capacity
        self.encoder = nn.Sequential(
            # First layer - increase channels more gradually
            nn.Conv1d(in_features, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.Conv1d(48, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            # Third layer
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder with matching architecture
        self.decoder = nn.Sequential(
            # First layer
            nn.ConvTranspose1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            # Second layer
            nn.ConvTranspose1d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.2),
            
            # Final layer
            nn.ConvTranspose1d(48, out_features, kernel_size=3, stride=1, padding=1),
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
            for x, _ in pbar:
                batch_size = x.shape[0]
                total_samples += batch_size
                x = x.to(self.device)
                
                self.optimizer.zero_grad()
                
                n_control = x.shape[1] - _.shape[1]
                sensor_part = x[:, n_control:, :]
                
                output = self.model(x)
                loss = self.criterion(output, sensor_part)
                
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
            for x, _ in pbar:
                batch_size = x.shape[0]
                total_samples += batch_size
                x = x.to(self.device)
                
                n_control = x.shape[1] - _.shape[1]
                sensor_part = x[:, n_control:, :]
                
                output = self.model(x)
                loss = self.criterion(output, sensor_part)
                
                total_loss += loss.item() * batch_size  # Multiply by batch size
                
                errors = torch.mean((output - sensor_part) ** 2, dim=(1, 2))
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

def normalize_with_indices(df, train_indices, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(clip=True)  # Add clipping
    
    # Fit scaler only on training data
    scaler.fit(df.iloc[train_indices])
    
    # Transform all data
    normalized_data = scaler.transform(df)
    normalized_df = pd.DataFrame(
        normalized_data,
        columns=df.columns,
        index=df.index
    )
    
    return normalized_df, scaler

def validate_normalization(df, name=""):
    min_val = df.min().min()
    max_val = df.max().max()
    if min_val < 0 or max_val > 1:
        print(f"WARNING: {name} normalization outside [0,1] range!")
        print(f"Range: [{min_val:.3f}, {max_val:.3f}]")

def check_normalization(data_original, data_normalized, scaler, name=""):
    """Check normalization results"""
    print(f"\nChecking {name} normalization:")
    print(f"Original range: [{data_original.min().min():.3f}, {data_original.max().max():.3f}]")
    print(f"Normalized range: [{data_normalized.min().min():.3f}, {data_normalized.max().max():.3f}]")
    
    # Check inverse transform
    data_recovered = pd.DataFrame(
        scaler.inverse_transform(data_normalized),
        columns=data_original.columns,
        index=data_original.index
    )
    max_error = np.abs(data_original - data_recovered).max().max()
    print(f"Maximum reconstruction error: {max_error:.6f}")


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set paths
    dataset_root = Path("./data")
    
    print("\n=== Loading Raw Data ===")
    rds_train = RawDataset(dataset_root, "VG5", load_synthetic=False, load_training=True)
    
    # Operating conditions
    operating_conditions = {
        "dyn_only_on": False,
        "turbine_mode": True,
        "equilibrium_turbine_mode": True,
        "short_circuit_mode": False
    }
    
    # Get raw data without normalization
    processed_data = preprocess_hydraulic_data(
        rds_train, 
        set_type="train",
        operating_conditions=operating_conditions,
        scaler_control=None,
        scaler_sensors=None,
        fit=False
    )
    
    # Prepare sensor groups
    control_df = processed_data['control_signals']
    sensors_df = processed_data['sensor_signals']
    
    print("\n=== Raw Data Statistics ===")
    print("Control Data Statistics (Raw):")
    print(f"Shape: {control_df.shape}")
    print(f"Range: [{control_df.values.min():.3f}, {control_df.values.max():.3f}]")
    print("\nFeature-wise ranges:")
    for col in control_df.columns:
        print(f"{col}: [{control_df[col].min():.3f}, {control_df[col].max():.3f}]")
    
    # Define selected sensor groups
    selected_groups = ['stator_temperature', 'cooling_system', 'heat_exchanger']
    sensor_groups = {
        'stator_temperature': sensors_df[1],
        'cooling_system': sensors_df[2],
        'heat_exchanger': sensors_df[3]
    }
    
    # Combine selected sensors
    combined_sensors = pd.concat([sensor_groups[group] for group in selected_groups], axis=1)
    print("\nSensor Data Statistics (Raw):")
    print(f"Shape: {combined_sensors.shape}")
    print(f"Range: [{combined_sensors.values.min():.3f}, {combined_sensors.values.max():.3f}]")
    
    print(f"\nNumber of control features: {len(control_df.columns)}")
    print(f"Number of sensor features: {len(combined_sensors.columns)}")
    
    print("\n=== Creating Initial Dataset for Split ===")
    initial_dataset = SlidingWindowDataset(
        control_df, 
        combined_sensors, 
        window=DEFAULT_PARAMS['window_size']
    )
    
    # Get train/val split indices
    _, _, train_indices, val_indices = create_data_loaders_with_indices(
        initial_dataset, 
        batch_size=DEFAULT_PARAMS['batch_size'], 
        val_split=DEFAULT_PARAMS['val_split']
    )
    
    print("\n=== Normalizing Data ===")
    # Initialize scalers with explicit feature range and clipping
    scaler_control = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler_sensors = MinMaxScaler(feature_range=(0, 1), clip=True)
    
    # Fit scalers on training data only
    print("\nFitting scalers on training data...")
    scaler_control.fit(control_df.iloc[train_indices])
    scaler_sensors.fit(combined_sensors.iloc[train_indices])
    
    # Transform all data
    control_df_normalized = pd.DataFrame(
        scaler_control.transform(control_df),
        columns=control_df.columns,
        index=control_df.index
    )
    sensors_df_normalized = pd.DataFrame(
        scaler_sensors.transform(combined_sensors),
        columns=combined_sensors.columns,
        index=combined_sensors.index
    )
    
    print("\n=== Checking Normalization Results ===")
    # Check normalization ranges for entire dataset
    print("\nControl Data Normalization:")
    print(f"Full range: [{control_df_normalized.values.min():.6f}, {control_df_normalized.values.max():.6f}]")
    print("\nSensor Data Normalization:")
    print(f"Full range: [{sensors_df_normalized.values.min():.6f}, {sensors_df_normalized.values.max():.6f}]")
    
    # Check train/val separation
    print("\n=== Checking Train/Val Separation ===")
    print("\nTraining Data Statistics:")
    print(f"Control range: [{control_df_normalized.iloc[train_indices].values.min():.6f}, "
          f"{control_df_normalized.iloc[train_indices].values.max():.6f}]")
    print(f"Sensor range: [{sensors_df_normalized.iloc[train_indices].values.min():.6f}, "
          f"{sensors_df_normalized.iloc[train_indices].values.max():.6f}]")
    
    print("\nValidation Data Statistics:")
    print(f"Control range: [{control_df_normalized.iloc[val_indices].values.min():.6f}, "
          f"{control_df_normalized.iloc[val_indices].values.max():.6f}]")
    print(f"Sensor range: [{sensors_df_normalized.iloc[val_indices].values.min():.6f}, "
          f"{sensors_df_normalized.iloc[val_indices].values.max():.6f}]")
    
    # Verify all values are within [0, 1]
    def check_range_violations(df, name):
        min_val = df.values.min()
        max_val = df.values.max()
        if min_val < 0 or max_val > 1:
            print(f"\nWARNING: {name} contains values outside [0, 1] range!")
            print(f"Min: {min_val:.6f}, Max: {max_val:.6f}")
            # Find problematic columns
            for col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min < 0 or col_max > 1:
                    print(f"Column {col}: [{col_min:.6f}, {col_max:.6f}]")
    
    check_range_violations(control_df_normalized, "Control data")
    check_range_violations(sensors_df_normalized, "Sensor data")
    
    print("\n=== Creating Final Dataset ===")
    final_dataset = SlidingWindowDataset(
        control_df_normalized, 
        sensors_df_normalized, 
        window=DEFAULT_PARAMS['window_size']
    )
    
    # Create final data loaders
    train_loader, val_loader = create_data_loaders_with_indices(
        final_dataset, 
        batch_size=DEFAULT_PARAMS['batch_size'], 
        val_split=DEFAULT_PARAMS['val_split']
    )[:2]
    
    # Check actual data going into model
    print("\n=== Checking Model Input Data ===")
    x_batch, _ = next(iter(train_loader))
    print(f"Sample batch shape: {x_batch.shape}")
    print(f"Sample batch range: [{x_batch.min().item():.6f}, {x_batch.max().item():.6f}]")
    
    print("\n=== Initializing Model ===")
    n_features_in = len(control_df.columns) + len(combined_sensors.columns)
    n_features_out = len(combined_sensors.columns)
    
    model = CAE(
        in_features=n_features_in,
        out_features=n_features_out,
        params=DEFAULT_PARAMS
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=DEFAULT_PARAMS['learning_rate']
    )

    trainer = CAETrainer(model, optimizer, device=device, model_name='cae_model')

    print("\n=== Starting Training ===")
    reconstruction_errors = trainer.train(
        train_loader, 
        val_loader, 
        n_epochs=DEFAULT_PARAMS['n_epochs']
    )
    
    # Save everything
    print("\n=== Saving Models and Statistics ===")
    os.makedirs('models', exist_ok=True)
    
    with open('models/scaler_control_cae.pkl', 'wb') as f:
        pickle.dump(scaler_control, f)
    
    with open('models/scaler_sensors_cae.pkl', 'wb') as f:
        pickle.dump(scaler_sensors, f)
    
    # Save normalization statistics
    normalization_stats = {
        'control': {
            'original_range': (control_df.values.min(), control_df.values.max()),
            'normalized_range': (control_df_normalized.values.min(), control_df_normalized.values.max()),
            'train_range': (control_df_normalized.iloc[train_indices].values.min(), 
                          control_df_normalized.iloc[train_indices].values.max()),
            'val_range': (control_df_normalized.iloc[val_indices].values.min(), 
                        control_df_normalized.iloc[val_indices].values.max())
        },
        'sensors': {
            'original_range': (combined_sensors.values.min(), combined_sensors.values.max()),
            'normalized_range': (sensors_df_normalized.values.min(), sensors_df_normalized.values.max()),
            'train_range': (sensors_df_normalized.iloc[train_indices].values.min(), 
                          sensors_df_normalized.iloc[train_indices].values.max()),
            'val_range': (sensors_df_normalized.iloc[val_indices].values.min(), 
                        sensors_df_normalized.iloc[val_indices].values.max())
        }
    }
    
    with open('models/normalization_stats.pkl', 'wb') as f:
        pickle.dump(normalization_stats, f)
    
    error_stats = {
        'mean': np.mean(reconstruction_errors),
        'std': np.std(reconstruction_errors)
    }
    
    with open('models/reconstruction_error_stats_cae.pkl', 'wb') as f:
        pickle.dump(error_stats, f)

    print("\nAll files saved in 'models' directory")

if __name__ == "__main__":
    main()