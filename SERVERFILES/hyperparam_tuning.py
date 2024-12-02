import os
import optuna
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import all necessary components from train.py
from train import (
    Trainer, init_model, seed_everything,
    DEFAULT_PARAMS_TOP, RawDataset, preprocess_hydraulic_data,
    Operating_modes_name, SlidingWindowDataset, create_data_loaders
)

# Define constants
SEED = 42
SELECTED_SENSOR_GROUPS = ['stator_temperature', 'cooling_system', 'heat_exchanger']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_data():
    """Setup global data needed for training"""
    # Load and preprocess data
    dataset_root = Path("./data")
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

    control_df = rds_u5_turb_ss_volt['control_signals']
    sensors_df_train = rds_u5_turb_ss_volt['sensor_signals']

    # Split sensor into 5 different subgroups
    sensor_groups = {
        'stator_temperature': sensors_df_train[1],
        'voltage_current': sensors_df_train[0],
        'cooling_system': sensors_df_train[2],
        'heat_exchanger': sensors_df_train[3],
        'other': sensors_df_train[4]
    }
    
    return control_df, sensor_groups

def setup_training_data(selected_sensor_groups, params, control_df, sensor_groups):
    """Setup training with selected sensor groups."""    
    # Prepare input data
    control_norm = control_df
    sensor_norm = pd.concat([sensor_groups[group] for group in selected_sensor_groups], axis=1)
    
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
        horizon=1,
        device=device
    )
    
    # Create loaders
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=params['batch_size'],
        val_split=0.2
    )
    
    return train_loader, val_loader, params

def evaluate_model(params, control_df, sensor_groups):
    """Evaluate model with multiple runs and storage management"""
    n_runs = 3
    df_all_val = pd.DataFrame()
    best_val_loss = float('inf')
    best_model_path = None

    for i in range(n_runs):
        seed = SEED + i
        seed_everything(seed)

        # Setup training data
        train_loader, val_loader, updated_params = setup_training_data(
            SELECTED_SENSOR_GROUPS, 
            params,
            control_df,
            sensor_groups
        )

        # Initialize model and optimizer
        model = init_model(updated_params)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=updated_params['base_lr'],
            weight_decay=updated_params['weight_decay']
        )

        # Train with storage management
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            n_epochs=updated_params['max_epochs'],
            device=device,
            seed=seed,
            model_name=f'trial_{study.trials[-1].number}_run_{i}'  # Unique name for each run
        )
        trainer.fit(train_loader, val_loader)

        # Collect validation loss
        val_rmse = trainer.losses['eval'][-1]
        
        # Only keep the best model across runs
        if val_rmse < best_val_loss:
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)  # Remove previous best
            best_val_loss = val_rmse
            best_model_path = trainer.model_path
        else:
            if trainer.model_path and os.path.exists(trainer.model_path):
                os.remove(trainer.model_path)  # Remove worse model
                
        df_all_val = pd.concat([df_all_val, pd.DataFrame({'rmse': [val_rmse]})], ignore_index=True)

    return df_all_val['rmse'].mean()

def objective(trial):
    """Optuna objective function"""
    # Start with default parameters
    params = DEFAULT_PARAMS_TOP.copy()
    
    # Add hyperparameters to tune
    params.update({
        # Architecture
        'n_layers': trial.suggest_int('n_layers', 2, 6),
        'n_ch': trial.suggest_int('n_ch', 32, 256, log=True),
        'n_k': trial.suggest_int('n_k', 3, 9, step=2),
        'n_hidden': trial.suggest_int('n_hidden', 64, 512, log=True),
        
        # Activation and regularization
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu']),
        'leaky_relu_slope': trial.suggest_float('leaky_relu_slope', 0.01, 0.3) 
            if params['activation'] == 'leaky_relu' else 0.01,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        
        # Optimization
        'base_lr': trial.suggest_float('base_lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    })
    
    try:
        # Evaluate the model
        rmse = evaluate_model(params, control_df, sensor_groups)
        return rmse
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        return float('inf')

if __name__ == "__main__":
    # Load data once
    print("Setting up data...")
    control_df, sensor_groups = setup_data()
    
    # Create directories for results
    os.makedirs('optuna_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Create unique study name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f'cnn_hyperopt_{timestamp}'
    
    # Clean up old files before starting
    print("\nCleaning up old files...")
    for f in os.listdir('models'):
        if f.endswith('.pt'):
            os.remove(os.path.join('models', f))
    
    print(f"\nCreating new study: {study_name}")
    # Create study with pruning
    study = optuna.create_study(
        storage=f'sqlite:///optuna_results/study_{timestamp}.db',
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=False
    )
    
    print("Starting optimization...")
    # Run optimization
    study.optimize(
        objective, 
        n_trials=10,  # Reduced to 5 trials
        timeout=3600*24  # 24 hour timeout
    )
    
    # Save results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save study statistics with timestamp
    results_file = f'optuna_results/study_results_{timestamp}.csv'
    df_results = study.trials_dataframe()
    df_results.to_csv(results_file)
    print(f"\nResults saved to: {results_file}")