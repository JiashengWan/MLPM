import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

Operating_modes_name = [
    "turbine_mode", "equilibrium_turbine_mode", "pump_mode", "equilibrium_pump_mode",
    "short_circuit_mode", "equilibrium_short_circuit_mode", "machine_on", "machine_off", 
    "dyn_only_on", "all"
]

# Define the preprocessing functions
@dataclass
class Case:
    info: pd.DataFrame
    measurements: pd.DataFrame

class RawDataset:
    def __init__(self, root, unit="VG4", load_training=False, load_synthetic=False) -> None:
        read_pq_file = lambda f: pq.read_table(root / f).to_pandas()
        cases = {
            "test": [f"{unit}_generator_data_testing_real_measurements.parquet", root / f"{unit}_generator_data_testing_real_info.csv"],
        }
        if load_training:
            cases["training"] = [f"{unit}_generator_data_training_measurements.parquet", root / f"{unit}_generator_data_training_info.csv"]
        if load_synthetic:
            cases["test_synthetic"] = [f"{unit}_generator_data_testing_synthetic_01_measurements.parquet", root / f"{unit}_generator_data_testing_synthetic_01_info.csv"]
        
        self.data_dict = {}
        for id_c, c in cases.items():
            info = pd.read_csv(c[1])
            measurements = read_pq_file(c[0])
            self.data_dict[id_c] = Case(info, measurements)

def extracting_name_groups_time_series(rds, set_type):
    df_name_control_sensors = rds.data_dict[set_type].info
    control_time_series_name = df_name_control_sensors[(df_name_control_sensors['control_signal'] == True) & (df_name_control_sensors["signal_type"] == "Measurement")]
    sensors_time_series_name = df_name_control_sensors[(df_name_control_sensors['control_signal'] == False) & (df_name_control_sensors["signal_type"] == "Measurement")]
    return control_time_series_name["attribute_name"], sensors_time_series_name["attribute_name"]

def extracting_groups_time_series(rds, set_type):
    # Extract names of control and sensor time series
    control_time_series_name, sensors_time_series_name = extracting_name_groups_time_series(rds, set_type=set_type)
    # Extract control and sensor data
    control_df = rds.data_dict[set_type].measurements[control_time_series_name]
    sensors_df = rds.data_dict[set_type].measurements[sensors_time_series_name]

    # Dynamically select available columns from Operating_modes_name
    available_columns = rds.data_dict[set_type].measurements.columns
    present_columns = [col for col in Operating_modes_name if col in available_columns]
    
    # Log missing columns if any
    missing_columns = [col for col in Operating_modes_name if col not in available_columns]
    if missing_columns:
        print(f"        Warning: Missing operating mode columns for {set_type}: {missing_columns}")
    
    # Extract only the present operating mode columns
    operating_modes_df = rds.data_dict[set_type].measurements[present_columns]

    return control_df, sensors_df, operating_modes_df


def get_indexes_operating_mode(operating_modes_df, operating_conditions):
    
    # Initialize a mask with all rows included
    mask = pd.Series([True] * len(operating_modes_df), index=operating_modes_df.index)
    
    for column, value in operating_conditions.items():
        if column in operating_modes_df.columns:
            # Apply the condition if the column exists
            mask &= (operating_modes_df[column] == value)
        else:
            # Log if the column is missing
            print(f"        Warning: Column '{column}' is missing from operating_modes_df and will be skipped.")
    
    # Return the filtered indexes
    filtered_indexes = operating_modes_df[mask].index
    return filtered_indexes


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
        'stator_temperature': df.columns[df.columns.str.startswith("stat_") & df.columns.str.contains("coil|magn")].tolist(),
        'cooling_system': df.columns[df.columns.str.startswith("air_circ_")].tolist(),
        'heat_exchanger': df.columns[df.columns.str.startswith("water_circ_")].tolist()
    }
    grouped_features = [feature for group in feature_groups.values() for feature in group]
    complementary_features = [col for col in df.columns if col not in grouped_features]
    feature_groups['complementary'] = complementary_features
    return feature_groups

def preprocess_hydraulic_data(rds, set_type, operating_conditions):
    control_df, sensors_df, operating_modes_df = extracting_groups_time_series(rds, set_type=set_type)
    if operating_conditions:
        indexes = get_indexes_operating_mode(operating_modes_df, operating_conditions)
        control_df = control_df.loc[indexes]
        sensors_df = sensors_df.loc[indexes]
    water_opening_features = [col for col in control_df.columns if "water" in col and "opening" in col]
    control_df = create_scaled_sum_feature(control_df, water_opening_features, "water_opening_sum")
    injector_opening_features = [col for col in control_df.columns if "injector" in col and "opening" in col]
    control_df = create_scaled_sum_feature(control_df, injector_opening_features, "injector_opening_sum")
    control_df = normalize_features(control_df)
    sensors_df = normalize_features(sensors_df)
    grouped_data = dynamic_group_features(sensors_df)
    selected_sensor_data = [sensors_df[grouped_data[key]] for key in grouped_data.keys()]
    return {'control_signals': control_df, 'sensor_signals': selected_sensor_data}

def cleaning_control_signals_turbine(control_df):
    print("      Cleaning control signals...")
    
    # Check and drop water opening columns
    water_opening_columns = control_df.columns[control_df.columns.str.startswith("water")]
    control_df = control_df.drop(columns=water_opening_columns, errors='ignore')

    # Check and drop 'pump_calculated_flow' if it exists
    if "pump_calculated_flow" in control_df.columns:
        control_df = control_df.drop(columns="pump_calculated_flow")
    else:
        print("        Column 'pump_calculated_flow' not found; skipping.")
    
    return control_df


def extract_operation_periods(control_df, time_column='turbine_rotspeed', gap_hours=72):
    time_series = control_df[time_column]
    valid_data = time_series[time_series.notna()]
    time_gaps = valid_data.index.to_series().diff()

    gap_threshold = pd.Timedelta(hours=gap_hours)
    new_operation = (time_gaps > gap_threshold).cumsum()
    operation_groups = pd.Series(index=time_series.index, data=None)
    operation_groups[valid_data.index] = new_operation

    control_df['operation_group'] = operation_groups

    operating_periods = []
    for group in control_df.groupby('operation_group'):
        if not group[1][time_column].isna().all():
            operating_periods.append(group[1])
    
    control_df.drop(columns='operation_group', inplace=True)
    return operating_periods

def save_preprocessed_data(base_dir, unit, mode, set_type, operation, control_df, sensors_df):
    save_directory = os.path.join(base_dir, unit, mode, set_type, f'period_{operation}')
    os.makedirs(save_directory, exist_ok=True)
    
    control_parquet_path = os.path.join(save_directory, 'control.parquet')
    control_csv_path = os.path.join(save_directory, 'control.csv')
    control_df.to_parquet(control_parquet_path)
    control_df.to_csv(control_csv_path, index=False)
    
    sensor_file_names = [
        'voltage_current',
        'stator_temperature',
        'cooling_system',
        'heat_exchanger',
        'complementary'
    ]
    
    for idx, sensor_group in enumerate(sensors_df):
        parquet_path = os.path.join(save_directory, f'{sensor_file_names[idx]}.parquet')
        csv_path = os.path.join(save_directory, f'{sensor_file_names[idx]}.csv')
        sensor_group.to_parquet(parquet_path)
        sensor_group.to_csv(csv_path, index=False)

def main():
    dataset_root = Path(r"./dataset")
    preprocess_root = Path(r"./preprocessed_dataset")
    units = ["VG4", "VG5", "VG6"]
    modes = ["turbine_mode", "pump_mode"]
    
    operating_conditions = {
        "turbine_mode": {
            "dyn_only_on": False, "turbine_mode": True, 
            "equilibrium_turbine_mode": True, "short_circuit_mode": False
        },
        "pump_mode": {
            "dyn_only_on": False, "pump_mode": True, 
            "equilibrium_pump_mode": True, "short_circuit_mode": False
        }
    }
    
    print("Starting preprocessing pipeline...\n")
    
    for unit in units:
        print(f"Processing unit: {unit}")
        
        # Load conditions based on the unit
        if unit == "VG4":
            print("  Loading data for VG4 (training only)...")
            rds = RawDataset(dataset_root, unit, load_training=True, load_synthetic=False)
        else:
            print(f"  Loading data for {unit} (training and synthetic)...")
            rds = RawDataset(dataset_root, unit, load_training=True, load_synthetic=True)
        
        for mode in modes:
            print(f"  Processing mode: {mode}")
            
            for set_type in rds.data_dict.keys():
                print(f"    Processing set type: {set_type}")
                
                try:
                    print("      Applying preprocessing steps...")
                    
                    current_operating_conditions = operating_conditions[mode]
                    
                    rds_processed = preprocess_hydraulic_data(
                        rds, set_type=set_type, operating_conditions=current_operating_conditions
                    )
                    
                    print("      Preprocessing completed.")
                    
                    control_df = rds_processed['control_signals']
                    sensors_df = rds_processed['sensor_signals']
                    
                    print("      Cleaning control signals...")
                    control_df = cleaning_control_signals_turbine(control_df)
                    print("      Control signals cleaned.")
                    
                    print("      Extracting operational periods...")
                    operating_periods = extract_operation_periods(control_df)
                    print(f"      Found {len(operating_periods)} operational periods.")
                    
                    print("      Saving preprocessed data...")
                    for period_idx, period_data in enumerate(operating_periods, start=1):
                        save_preprocessed_data(
                            preprocess_root, unit, mode, set_type, period_idx, period_data, sensors_df
                        )
                    
                    print("      Data saved successfully.")
                
                except Exception as e:
                    print(f"    Error during processing set type: {set_type}")
                    print(f"    Exception: {e}")
                    raise e  # Re-raise the error for debugging
                
            print(f"  Finished processing mode: {mode}\n")
        
        print(f"Finished processing unit: {unit}\n")
    
    print("Preprocessing pipeline completed.")


if __name__ == "__main__":
    main()
