# Artificial Scheduler

An intelligent process scheduling daemon that uses machine learning to dynamically adjust process priorities (nice values) in Linux systems. The project leverages eBPF for low-overhead process monitoring and XGBoost for prediction-based scheduling decisions.

## Overview

The Artificial Scheduler is a daemon that monitors process execution in real-time using eBPF and applies machine learning models to predict process characteristics and adjust their scheduling priorities accordingly. It aims to improve system performance by intelligently rebalancing process priorities based on predicted execution time and resource usage patterns.

## Features

- **Real-time Process Monitoring**: Uses eBPF to capture process execution and exit events with minimal overhead
- **Machine Learning Predictions**: Employs XGBoost models to predict process turnaround time and classify processes
- **Dynamic Priority Adjustment**: Automatically adjusts process nice values based on ML predictions
- **ELF Binary Analysis**: Extracts features from executable binaries to enhance prediction accuracy
- **Database Logging**: Stores process data and predictions in SQLite for analysis and model retraining
- **Intelligent Filtering**: Skips common system utilities and helpers to focus on meaningful processes

## Architecture

### Core Components

- **`__main__.py`**: Main daemon entry point and orchestration
- **`ebpf_utils.py`**: eBPF program management and event handling
- **`model.py`**: Machine learning model management and prediction logic
- **`rebalance.py`**: Process priority rebalancing algorithms
- **`handlers.py`**: Event handlers for process execution and exit
- **`elf_utils.py`**: ELF binary analysis and feature extraction
- **`db.py`**: Database operations and data persistence
- **`config.py`**: Configuration settings and constants

### Machine Learning Models

The system includes pre-trained models:

- **Turnaround Time Prediction**: `modelForTAT.joblib` - XGBoost model for predicting process execution time
- **Process Classification**: `modelForPriorty.joblib` - K-means clustering for process categorization
- **Feature Scaling**: `scaler.joblib` - StandardScaler for feature normalization

## Installation

### Prerequisites

- Linux system with eBPF support
- Python 3.8+
- Root privileges (required for eBPF and nice value adjustments)

### Dependencies

Install the required Python packages:

```bash
pip install bcc psutil pandas numpy scikit-learn xgboost joblib
```

### System Requirements

- Linux kernel 4.1+ (for eBPF support)
- BCC (BPF Compiler Collection) installed
- readelf utility (part of binutils)

## Usage

### Running the Daemon

Run the scheduler daemon with root privileges:

```bash
sudo python -m Artificial_Scheduler
```

### Configuration

Edit `config.py` to customize:

- **Label mappings**: Process categories and their nice value ranges
- **Skip helpers**: Processes to ignore during monitoring
- **File paths**: Model and database file locations
- **Logging settings**: Log level and output format

### Process Categories

The system categorizes processes into 5 types with different nice value ranges:

- **Type 0 (Merge Sort)**: Nice values -5 to -2
- **Type 1 (IO Operations)**: Nice values -1 to 1
- **Type 2 (Bubble Sort)**: Nice values -15 to -11
- **Type 3 (Matrix Operations)**: Nice values -19 to -16
- **Type 4 (Heap Sort)**: Nice values -10 to -6

## How It Works

1. **Process Detection**: eBPF programs attach to kernel tracepoints to detect process execution
2. **Feature Extraction**: When a new process starts, the system:
   - Extracts ELF binary features using `readelf`
   - Collects process memory and I/O information
   - Gathers system resource metrics
3. **ML Prediction**: Features are fed to trained models to predict:
   - Expected turnaround time
   - Process category/priority class
4. **Priority Assignment**: Based on predictions, the process is assigned a nice value
5. **Dynamic Rebalancing**: Periodically rebalances priorities based on actual vs predicted performance
6. **Data Collection**: Logs all decisions and outcomes for future model improvement

## File Structure

```
Artificial_Scheduler/
├── __init__.py
├── __main__.py              # Main daemon entry point
├── config.py                # Configuration and constants
├── db.py                    # Database operations
├── ebpf_utils.py           # eBPF program management
├── elf_utils.py            # ELF binary analysis
├── handlers.py             # Process event handlers
├── model.py                # ML model management
├── rebalance.py            # Priority rebalancing logic
├── process_data.db         # SQLite database
├── modelForTAT.joblib      # Turnaround time model
├── modelForPriorty.joblib  # Process priority model
├── scaler.joblib           # Feature scaler
└── ml_nice_adjuster.log    # Application log file
```

## Logging

The daemon logs activities to `ml_nice_adjuster.log` including:

- Process detection and classification
- Priority adjustments
- Model predictions
- Error conditions and debugging information

## Safety Features

- **Process Filtering**: Ignores critical system processes and common utilities
- **Interactive Process Focus**: Only manages processes with TTY connections
- **Graceful Shutdown**: Properly cleans up eBPF attachments and database connections
- **Error Handling**: Comprehensive exception handling to prevent daemon crashes

## Limitations

- Requires root privileges for operation
- Linux-only (uses eBPF and Linux-specific process APIs)
- Focuses on interactive processes with TTY connections
- May need model retraining for different workload patterns

## Development

To contribute or modify the scheduler:

1. Fork the repository
2. Set up the development environment with required dependencies
3. Test changes in a safe environment (VM recommended)
4. Ensure proper cleanup of eBPF programs during testing

## Models
The system uses two main models: one for predicting process turnaround time (TAT) and another for classifying process priority.

#### Task Turnaround Time (TAT) Prediction Model

The TAT prediction model estimates how long a task will take to complete, based on system-level metrics captured at dispatch time. These include:

- Memory usage (e.g., RSS, VMS)
- I/O operation statistics
- ELF binary properties
- Process/thread count
- and more

We primarily use **XGBoost Regression** due to its robustness, scalability, and ability to handle noisy, real-world data. To evaluate and validate model performance, we compared XGBoost to two alternative approaches:

- **Random Forest Regressor**, a strong ensemble model based on bagging.
- **Stacked Ensemble**, which combines predictions from both XGBoost and Random Forest using a meta-model.

##### Modeling Pipeline:

1. **Data Collection**: Collected runtime metrics from scheduled tasks.
2. **Preprocessing & Feature Engineering**: Normalized numeric fields, handled missing values, and derived new composite features.
3. **Model Training & Comparison**:
   - Trained models on various train/test splits (e.g., 70/30, 80/20).
   - Evaluated performance using MAE, RMSE, and R².
   - Conducted cross-validation for stability analysis.
4. **Model Selection**: XGBoost showed slightly better generalization and interpretability via feature importance scores.
5. **Feature Importance Analysis**: Identified key system features affecting TAT.

This model enables more informed scheduling decisions by estimating task duration before execution. The comparison between models helped select the most reliable approach for deployment in dynamic environments.
##### Model Performance Comparison
![Model Performance Comparison](https://github.com/zivshamli/MLSchdulerImage/blob/c384c3031e844ef77aa379b05f16a78e005a8767/output.png)

#### Process Priority Classification Model

In addition to predicting turnaround time, the scheduler includes a **Process Priority Classification Model** to assign appropriate scheduling classes to new processes. This model is based on **K-means clustering**, which groups processes into logical categories based on their resource usage patterns and behavior.

The same set of features used in the TAT prediction model-such as memory usage, I/O activity, and ELF binary properties and predicted TAT are also used as input for the clustering algorithm. This ensures consistency in feature representation and allows both models to leverage shared information about process characteristics.

To determine the optimal number of clusters (**K**), we performed an evaluation using the Elbow Method and Silhouette Score. A corresponding graph is included in the results section. Additionally, we generated a 3D visualization of the final clustering using PCA to demonstrate how processes are grouped in feature space.

The resulting clusters are mapped to predefined scheduling classes, each with its own range of nice values, as specified in the configuration. This enables the system to make fast, unsupervised decisions about process priority at runtime.

##### Elbow Method
The Elbow Method was used to evaluate the distortion (inertia) across different K values. The optimal K was chosen where the curve bends (the "elbow").

![Elbow Method](https://github.com/zivshamli/MLSchdulerImage/blob/116125c22949a80a8f1a2d4df568e568510505ef/elbow%20metohd.png)

##### Silhouette Score
Silhouette analysis was performed to assess the consistency of clusters. The K value with the highest average silhouette score was selected.

![Silhouette Score](https://github.com/zivshamli/MLSchdulerImage/blob/116125c22949a80a8f1a2d4df568e568510505ef/bestK.png)

##### Clustering Visualization (PCA)
After clustering, a 3D PCA projection was used to visualize the process distribution across clusters, demonstrating clear separation.
![Process Clustering (PCA)](https://github.com/zivshamli/MLSchdulerImage/blob/116125c22949a80a8f1a2d4df568e568510505ef/priority%20clustering.png)

## Results

To evaluate the effectiveness of the Artificial Scheduler, we measured the turnaround time (TAT) of processes with and without the machine learning models integrated via eBPF. The results demonstrate a consistent improvement in system performance when using our ML-based scheduling approach.

As shown in the dashboard below, the average TAT was reduced by over 5% compared to the baseline (no ML models). This improvement highlights the scheduler’s ability to make more informed and adaptive priority decisions, leading to faster process completion and better resource utilization.

<img width="811" height="1043" alt="Dashboard 1 (2)" src="https://github.com/user-attachments/assets/2dbd8715-8003-4745-a1cc-527d0ae95f77" />

These results validate the integration of eBPF-based monitoring with machine learning models for dynamic process scheduling in Linux environments.




## License

MIT License

## Author

Gal Radia & Ziv Shamli

---

**Warning**: This software modifies process priorities and requires root access. Use with caution and test thoroughly before deploying in production environments.
