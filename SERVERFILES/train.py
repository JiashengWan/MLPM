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
import time
from dataclasses import dataclass
import random

#####################
## DEFAULT PARAMS ###
#####################
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Operating_modes_name = [
    "turbine_mode", "equilibrium_turbine_mode", "pump_mode", "equilibrium_pump_mode",
    "short_circuit_mode", "equilibrium_short_circuit_mode", "machine_on", "machine_off", 
    "dyn_only_on", "all"
]

DEFAULT_PARAMS_TOP = {
    'window': 50,
    'n_ch': 64,
    'n_k': 7,
    'n_hidden': 128,
    'n_layers': 4,
    'dropout': 0.2,
    'padding': 'same',
    'use_batchnorm': True,
    'batch_size': 256,
    'base_lr': 5e-4,
    'weight_decay': 1e-5,
    'max_epochs': 50,
    'activation': 'relu',
    'leaky_relu_slope': 0.01 
}

# Your existing classes and functions here
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
        """Sliding window dataset for temperature prediction

        Args:
            control_df (pd.DataFrame): DataFrame containing control signals
            sensor_df (pd.DataFrame): DataFrame containing stator temperature readings
            window (int, optional): sequence window length. Defaults to 50.
            stride (int, optional): data stride length. Defaults to 1.
            horizon (int, optional): prediction forecasting length. Defaults to 1.
        """
        self.window = window
        self.stride = stride
        self.horizon = horizon
        
        # Combine control and sensor data for input
        self.X_control = control_df.values.astype(np.float32)
        self.X_sensor = sensor_df.values.astype(np.float32)
        self.X = np.concatenate([self.X_control, self.X_sensor], axis=1)
        
        self.y = sensor_df.values.astype(np.float32)          # Sensor measurements (Y_t)
        
        # Calculate valid indices for sliding windows
        self.indices = self._get_indices()
        self.indices = torch.from_numpy(self.indices).to(device)

    def _get_indices(self):
        """Calculate valid indices for sliding windows"""
        valid_length = len(self.X) - self.window - self.horizon + 1
        idx_list = [i for i in range(0, valid_length, self.stride)]
        return np.asarray([(idx, idx+self.window) for idx in idx_list])

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        i_start, i_stop = self.indices[i]
        x = self.X[i_start:i_stop, :]
        y = self.y[i_stop + self.horizon - 1]  # Target is horizon steps ahead
        x = torch.from_numpy(x).transpose(0, 1)  # Shape: (features, sequence_length)
        y = torch.from_numpy(y)
        return x, y

# def create_datasets(control_df, sensor_df, window_size, train_indices, test_indices, device='cpu'):
#     """Create train and test datasets"""
#     # Split data according to indices
#     control_train = control_df.iloc[train_indices]
#     sensor_train = sensor_df.iloc[train_indices]
    
#     control_test = control_df.iloc[test_indices]
#     sensor_test = sensor_df.iloc[test_indices]
    
#     # Create datasets
#     train_dataset = SlidingWindowDataset(control_train, sensor_train, window=window_size, device=device)
#     test_dataset = SlidingWindowDataset(control_test, sensor_test, window=window_size, device=device)
    
#     # Normalize features using training data
#     scaler = MinMaxScaler()
#     train_dataset.X = scaler.fit_transform(train_dataset.X)
#     test_dataset.X = scaler.transform(test_dataset.X)
    
#     return train_dataset, test_dataset


def create_data_loaders(dataset, batch_size=256, val_split=0.2):
    """Create train and validation data loaders from a single dataset"""
    # Set random seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    
    # Calculate split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=valid_sampler,
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    
    return train_loader, val_loader

def seed_everything(seed: int):
    """Sets the seed for generating random numbers in PyTorch, numpy and Python.
    
    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_weights(m):
    """Initialize neural network weights"""
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.weight.data = nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        n_epochs=20,
        criterion=nn.MSELoss(),
        model_name='best_model',
        seed=42,
        device='cpu'
    ):
        self.seed = seed
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.criterion = criterion
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Model saving
        time_stamp = time.strftime("%m%d%H%M%S")
        self.model_path = f'models/{model_name}_{time_stamp}.pt'
        
        self.losses = {split: [] for split in ['train', 'eval', 'test']}

    # Add these methods
    def save(self, model, model_path=None):
        """Save model state dict"""
        if model_path is None:
            model_path = self.model_path
        torch.save(model.state_dict(), model_path)
        
    def load(self, model, model_path=None):
        """Load model state dict"""
        if model_path is None:
            model_path = self.model_path
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")
        return model

    def compute_loss(self, x, y, model=None):
        y = y.view(-1)
        y_pred = self.model(x)
        y_pred = y_pred.view(-1)
        loss = self.criterion(y, y_pred)
        return loss, y_pred, y
    
    def train_epoch(self, loader):
        self.model.train()
        b_losses = []
        with tqdm(loader, desc="Training", unit="batch") as t_loader:
            for x, y in t_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                
                loss, pred, target = self.compute_loss(x, y)
                
                loss.backward()
                self.optimizer.step()
                b_losses.append(loss.detach().cpu().numpy())
                
                t_loader.set_postfix(loss=loss.detach().cpu().numpy())
        
        agg_loss = np.sqrt((np.asarray(b_losses) ** 2).mean())
        self.losses['train'].append(agg_loss)
        return agg_loss
    
    @torch.no_grad()
    def eval_epoch(self, loader, split='eval'):
        self.model.eval()
        b_losses = []
        with tqdm(loader, desc=f"Evaluating ({split})", unit="batch") as t_loader:
            for x, y in t_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                loss, pred, target = self.compute_loss(x, y)
                
                b_losses.append(loss.detach().cpu().numpy())
                t_loader.set_postfix(loss=loss.detach().cpu().numpy())
        
        agg_loss = np.sqrt((np.asarray(b_losses) ** 2).mean())
        self.losses[split].append(agg_loss)
        return agg_loss
    
    def fit(self, train_loader, val_loader):
        """Train the model"""
        print(f"Training model for {self.n_epochs} epochs...")
        train_start = time.time()  # Added total time tracking from first version
        best_val_loss = float('inf')
        
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.n_epochs}")
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.eval_epoch(val_loader, split='eval')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(self.model, self.model_path)  # Added model_path from first version
                print(f"Epoch {epoch + 1}: New best model saved!")
            
            # More structured logging from first version
            s = (
                f"[Epoch {epoch + 1}] "
                f"train_loss = {train_loss:.5f}, "
                f"val_loss = {val_loss:.5f}"
            )
            epoch_time = time.time() - epoch_start
            s += f" [{epoch_time:.1f}s]"
            print(s)
        
        # Added total time reporting from first version
        train_time = int(time.time() - train_start)
        print(f'Training completed in {train_time}s!')


    @torch.no_grad()
    def eval_prediction(self, test_loader):
        """Evaluate predictions for anomaly detection"""
        print("Evaluating predictions...")
        
        best_model = self.load(self.model)  # Load best model
        best_model.eval()
        
        preds = []
        trues = []
        
        for x, y in tqdm(test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            _, y_pred, y_target = self.compute_loss(x, y)  # Use compute_loss for consistency
            preds.append(y_pred.cpu().numpy())
            trues.append(y_target.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print(f"Predictions shape: {preds.shape}")
        print(f"True values shape: {trues.shape}")
        
        # Create sensor-wise DataFrames
        dfs = []
        for sensor_idx in range(preds.shape[1]):
            df = pd.DataFrame({
                'pred': preds[:, sensor_idx],
                'true': trues[:, sensor_idx],
                'error': preds[:, sensor_idx] - trues[:, sensor_idx],
                'abs_error': np.abs(preds[:, sensor_idx] - trues[:, sensor_idx]),
                'sensor': sensor_idx
            })
            dfs.append(df)
        
        # Combine all sensor DataFrames
        df_all = pd.concat(dfs, axis=0, ignore_index=True)
        
        # Calculate metrics
        overall_rmse = np.sqrt(np.mean((preds - trues) ** 2))
        sensor_rmse = np.sqrt(np.mean((preds - trues) ** 2, axis=0))
        
        # Create metrics DataFrame including both RMSE and seed
        df_metrics = pd.DataFrame({
            'sensor': list(range(len(sensor_rmse))) + ['overall'],
            'rmse': np.append(sensor_rmse, overall_rmse),
            'seed': self.seed
        })
        
        print("\nMetrics Summary:")
        print(f"Overall RMSE: {overall_rmse:.4f}")
        print("Sensor-wise RMSE:", sensor_rmse)
        
        return df_all, df_metrics

class CNN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 window=50, 
                 n_ch=10, 
                 n_k=10, 
                 n_hidden=50, 
                 n_layers=3,
                 dropout=0.0,
                 padding='same',
                 use_batchnorm=False,
                 activation='relu',
                 leaky_relu_slope=0.01):

        super().__init__()

        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create conv layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(in_channels, n_ch, kernel_size=n_k, padding=padding),
            nn.BatchNorm1d(n_ch) if use_batchnorm else nn.Identity(),
            self.activation,  # Use the selected activation
            nn.Dropout(dropout)
        ))
        
        # Middle layers
        for _ in range(n_layers - 2):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(n_ch, n_ch, kernel_size=n_k, padding=padding),
                nn.BatchNorm1d(n_ch) if use_batchnorm else nn.Identity(),
                self.activation,
                nn.Dropout(dropout)
            ))
        
        # Final conv layer
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(n_ch, n_ch, kernel_size=n_k, padding=padding),
            nn.BatchNorm1d(n_ch) if use_batchnorm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout)
        ))
        
        # Fully connected layers
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_ch * window, n_hidden),
            nn.BatchNorm1d(n_hidden) if use_batchnorm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(n_hidden, out_channels)
        )
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, x):
        # Pass through conv layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Pass through regressor
        x = self.regressor(x)
        return x
    
    pass

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

# Helper functions for feature extraction and normalization
def create_scaled_sum_feature(df, feature_prefixes, new_feature_name):
    feature_columns = df.columns[df.columns.str.startswith(tuple(feature_prefixes))]
    df[new_feature_name] = df[feature_columns].sum(axis=1)
    scaler = MinMaxScaler()
    df[new_feature_name] = scaler.fit_transform(df[[new_feature_name]])
    df.drop(columns=feature_columns, inplace=True)
    return df

def normalize_features(df):
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
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

def preprocess_hydraulic_data(rds, set_type, operating_conditions):
    """
    Preprocess the hydraulic site data to return normalized control and sensor signals filtered by operating conditions.

    Parameters:
    - rds: RawDataset object containing the data dictionary
    - set_type: Type of dataset to use ('train', 'test', etc.)
    - feature_group: The group of sensor features to select for preprocessing
    - operating_conditions: A dictionary specifying operating conditions to filter the data
    
    Returns:
    - Dictionary with two normalized and filtered DataFrames: control signals and sensor signals
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
    control_df = create_scaled_sum_feature(control_df, water_opening_features, "water_opening_sum")

    # Select features containing both 'injector' and 'opening' in their name
    injector_opening_features = [col for col in control_df.columns if "injector" in col and "opening" in col]
    control_df = create_scaled_sum_feature(control_df, injector_opening_features, "injector_opening_sum")

    # Step 4: Normalize the control and sensor DataFrames
    control_df = normalize_features(control_df)
    sensors_df = normalize_features(sensors_df)

    # Step 5: Select only the sensor features corresponding to the specified feature group
    grouped_data = dynamic_group_features(sensors_df)
    selected_sensor_data = []
    for key in grouped_data.keys():
        selected_sensor_data.append(sensors_df[grouped_data[key]])

    return {
        'control_signals': control_df,
        'sensor_signals': selected_sensor_data
        }

def plot_training_losses(trainer):
    plt.figure(figsize=(10, 6))
    
    linestyles = {
        'train': 'solid', 
        'eval': 'dashed'
    }
    
    colors = {
        'train': 'blue',
        'eval': 'red'
    }
    
    # Only iterate over train and eval losses
    for split in ['train', 'eval']:
        if split in trainer.losses:  # Ensure the split exists in the losses dictionary
            plt.plot(
                range(1, 1 + len(trainer.losses[split])), 
                trainer.losses[split], 
                label=f'{split} loss', 
                linestyle=linestyles[split],
                color=colors[split],
                marker='o',
                markersize=4
            )
    
    plt.title("Training/Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.yscale('log')
    
    # Add minor gridlines
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def init_model(params):
    """Initialize CNN model with given parameters."""
    model = CNN(
        in_channels=params['in_channels'],
        out_channels=params['out_channels'],
        window=params['window'],
        n_ch=params['n_ch'],
        n_k=params['n_k'],
        n_hidden=params['n_hidden'],
        n_layers=params['n_layers'],
        dropout=params['dropout'],
        padding=params['padding'],
        use_batchnorm=params['use_batchnorm'],
        activation=params.get('activation', 'relu'),
        leaky_relu_slope=params.get('leaky_relu_slope', 0.01)
    ).to(device)

    print(model)
    return model

def setup_training(selected_sensor_groups,params):
    """Setup training with selected sensor groups."""
    print(f"\nSetting up training for {selected_sensor_groups}")
    
    # Prepare input data
    control_norm = control_df_train
    sensor_norm = pd.concat([SENSOR_GROUPS[group] for group in selected_sensor_groups], axis=1)
    
    # Update params with combined input channels
    total_input_channels = len(control_norm.columns) + len(sensor_norm.columns)
    params = params.copy()
    params['in_channels'] = total_input_channels
    params['out_channels'] = len(sensor_norm.columns)
    
    print(f"Input channels (control + sensor): {params['in_channels']}")
    print(f"Output channels (future sensors): {params['out_channels']}")
    
    # Create dataset with horizon=1 for t+1 prediction
    dataset = SlidingWindowDataset(
        control_norm,
        sensor_norm,
        window=params['window'],
        stride=1,
        horizon=1
    )
    
    # Create loaders
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=params['batch_size'],
        val_split=0.2
    )
    
    return train_loader, val_loader

def main():

    # Set paths
    dataset_root = Path("./data")
    
    # Set seed
    SEED = 42
    seed_everything(SEED)

    # Load and preprocess data
    rds_u5_train = RawDataset(dataset_root, "VG5", load_synthetic=False, load_training=True)

    # Turbine Steady State Operation
    operating_conditions_set = {
        "dyn_only_on": False, "turbine_mode": True, 
        "equilibrium_turbine_mode": True, "short_circuit_mode": False
    }

    # Preprocess data
    rds_u5_turb_ss_volt = preprocess_hydraulic_data(
        rds_u5_train, set_type="train", operating_conditions=operating_conditions_set
    )

    del rds_u5_train  # To free memory

    global control_df_train, SENSOR_GROUPS
    control_df_train = rds_u5_turb_ss_volt['control_signals']
    sensors_df_train = rds_u5_turb_ss_volt['sensor_signals']

    # Split sensor into 5 different subgroups
    sensors_voltage_current = sensors_df_train[0]
    sensors_stator_temperature = sensors_df_train[1]
    sensors_cooling_system = sensors_df_train[2]
    sensors_heat_exchanger = sensors_df_train[3]
    sensors_other = sensors_df_train[4]

    # Define sensor groups
    SENSOR_GROUPS = {
        'stator_temperature': sensors_stator_temperature,
        'voltage_current': sensors_voltage_current,
        'cooling_system': sensors_cooling_system,
        'heat_exchanger': sensors_heat_exchanger,
        'other': sensors_other
    }

    SELECTED_SENSOR_GROUPS = ['stator_temperature', 'cooling_system', 'heat_exchanger']

    # Calculate channels
    total_output_channels = sum(len(SENSOR_GROUPS[group].columns) for group in SELECTED_SENSOR_GROUPS)
    total_input_channels = len(control_df_train.columns) + total_output_channels

    DEFAULT_PARAMS = {
        **DEFAULT_PARAMS_TOP,
        'in_channels': total_input_channels,
        'out_channels': total_output_channels
    }


    # Set up training
    train_loader, val_loader = setup_training(SELECTED_SENSOR_GROUPS, DEFAULT_PARAMS)

    # Initialize model
    model = init_model(DEFAULT_PARAMS)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEFAULT_PARAMS['base_lr'],
        weight_decay=DEFAULT_PARAMS['weight_decay']
    )

    # Create and run trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        n_epochs=DEFAULT_PARAMS['max_epochs'],
        device=device,
        seed=SEED,
    )

    # Train the model
    trainer.fit(train_loader, val_loader)

    # Save model and training results
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'params': DEFAULT_PARAMS,
        'trainer_losses': trainer.losses,
        'total_input_channels': total_input_channels,
        'total_output_channels': total_output_channels
    }
    torch.save(checkpoint, 'models/model_checkpoint.pth')

    # Save the training loss plot
    plot_training_losses(trainer)
    plt.savefig('models/training_loss.pdf', format='pdf')
    plt.close()

if __name__ == "__main__":
    main()