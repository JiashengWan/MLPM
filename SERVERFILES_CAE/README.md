# Anomaly Detection and Thresholding Framework

This repository contains Python scripts and Jupyter notebooks designed for anomaly detection in time-series data using Convolutional Autoencoders (CAE). The framework reconstructs signals from overlapping windows, computes reconstruction errors, and identifies anomalies based on precomputed thresholds.

## Files and Their Purpose

### Jupyter Notebooks

1. **1D_CAE_EvalScitas.ipynb**
   - Evaluates the performance of a 1D CAE model in reconstructing time-series data.
   - Uses datasets with varying operational conditions for turbines and pumps.
   - Provides insights into reconstruction errors and visualization of anomalous signals.

2. **anomaliesFindingThreshold.ipynb**
   - Focuses on determining thresholds for anomaly detection.
   - Calculates reconstruction errors using quantiles and visualizes the results.
   - Suitable for fine-tuning detection sensitivity.

3. **create_training_df.ipynb**
   - Prepares training datasets for the CAE model.
   - Applies preprocessing steps such as normalization and sliding window transformations.
   - Outputs a structured dataset in parquet format for further processing.

4. **findAnomalies.ipynb**
   - Combines reconstruction errors and thresholds to identify anomalies.
   - Includes visualization tools for highlighting faulty signals and their correlations.
   - Filters out transient errors based on customizable consecutive thresholds.

5. **modelTransfer.ipynb**
   - Demonstrates transferring pre-trained CAE models to other datasets or use cases.
   - Includes steps for loading models, retraining, and evaluating results.

6. **plotReconstructionTraining.ipynb**
   - Visualizes the reconstruction performance of CAE models on training datasets.
   - Generates plots comparing original and reconstructed signals.

### Python Scripts

1. **findAnomalies.py**
   - A standalone script for identifying anomalies in datasets.
   - Implements key functions such as:
     - Signal reconstruction from overlapping windows.
     - Error computation.
     - Anomaly filtering and correlation analysis.
   - Can be integrated into automated pipelines.

2. **getThreshold.py**
   - Computes thresholds for anomaly detection using a predefined quantile.
   - Saves threshold values for each signal channel to CSV files.
   - Includes functionality for reading and applying these thresholds.

3. **loadData.py**
   - Handles loading and preprocessing of raw datasets.
   - Implements classes and functions for organizing raw data into training and testing sets.
   - Supports feature extraction, group-based normalization, and dynamic filtering based on operational modes.

4. **plotReconstructionTraining.py**
   - Automates the visualization of CAE reconstruction on training data.
   - Saves plots to specified directories for documentation and analysis.

5. **reconstructDF.py**
   - Provides utility functions for reconstructing entire datasets from sliding windows.
   - Includes options for saving reconstructed datasets and computing reconstruction errors.

6. **traincae.py**
   - Contains the CAE model architecture, training pipeline, and helper functions.
   - Implements the SlidingWindowDataset class for handling time-series data.
   - Provides utilities for training, validation, and saving models.

7. **traincae_copy.py**
   - A duplicate of `traincae.py` for potential experimentation or backup purposes.

### Additional Dependencies
- **`traincae` Module:** Contains classes and parameters for CAE models, including:
  - `CAE`: The Convolutional Autoencoder model.
  - `SlidingWindowDataset`: A dataset class for handling sliding window transformations.

### Operating Conditions
- `operating_conditions_set_turbine`: Operational flags for turbine mode.
- `operating_conditions_set_pump`: Operational flags for pump mode.

## How to Use

1. **Preparing Training Data:**
   - Use `create_training_df.ipynb` to generate datasets for training.
   - Ensure the dataset is stored in the `data` folder in parquet format.

2. **Training the Model:**
   - Train a CAE model using the `traincae.py` script.

3. **Threshold Computation:**
   - Run `getThreshold.py` or `anomaliesFindingThreshold.ipynb` to compute and save thresholds.

4. **Anomaly Detection:**
   - Use `findAnomalies.py` or `findAnomalies.ipynb` to detect and analyze anomalies in your dataset.

5. **Visualization:**
   - Use `plotReconstructionTraining.py` or `plotReconstructionTraining.ipynb` to generate plots comparing original and reconstructed signals.

## Results Visualization
- Heatmaps and time-series plots are included for visual analysis of reconstruction errors and anomaly correlations.

## Requirements
- Python 3.8+
- Required libraries: `torch`, `numpy`, `pandas`, `joblib`, `matplotlib`, `seaborn`, `tqdm`, `pyarrow`.

## Setup
1. Install the required dependencies:
   ```bash
   pip install torch numpy pandas joblib matplotlib seaborn tqdm pyarrow
   ```
2. Set paths for models and datasets in the scripts and notebooks.

## Future Enhancements
- Integration with real-time monitoring systems.
- Dynamic threshold adjustment based on historical data.

## Contributors
This repository was developed for research purposes and anomaly detection in industrial systems.
