# Project Structure Update

## Overview

The project has been reorganized for better maintainability and clarity. All files are now organized into logical folders.

## New Project Structure

```
Artificial_Scheduler/
├── Artificial_Scheduler/           # Main source code
│   ├── __init__.py
│   ├── __main__.py                 # Entry point
│   ├── config.py                   # Configuration and paths
│   ├── db.py                       # Database operations
│   ├── ebpf_utils.py              # eBPF utilities
│   ├── elf_utils.py               # ELF file analysis
│   ├── handlers.py                # Process event handlers
│   ├── model.py                   # ML model management
│   └── rebalance.py               # Process rebalancing logic
│
├── models/                        # Machine Learning Models
│   ├── modelForTAT.joblib         # Turnaround Time prediction model
│   ├── modelForPriorty.joblib     # Priority classification model (KMeans)
│   ├── scaler.joblib              # Feature scaler
│   ├── *.pkl                      # Backup pickle files
│   └── *.bak                      # Auto-generated backups during retraining
│
├── database/                      # Database files
│   └── process_data.db            # SQLite database for process data
│
├── logs/                          # Log files
│   ├── ml_nice_adjuster.log       # Main application log
│   ├── ml_nice_adjuster.log.1     # Rotated log backup 1
│   ├── ml_nice_adjuster.log.2     # Rotated log backup 2
│   └── ml_nice_adjuster.log.3     # Rotated log backup 3
│
├── Training_Models/               # Jupyter notebooks for model training
│   ├── TrainingKmeansPriorityModel.ipynb
│   └── TrainingModelForTAT.ipynb
│
├── LICENSE.txt
├── README.md
└── LOGGING_OPTIMIZATION.md       # Documentation about logging improvements
```

## Key Changes Made

### 1. **File Organization**

- **Models**: All `.joblib` and `.pkl` files moved to `models/` folder
- **Database**: SQLite database moved to `database/` folder
- **Logs**: All log files organized in `logs/` folder with rotation

### 2. **Configuration Updates**

The `config.py` file has been updated to use the new folder structure:

```python
# Old paths (relative to Artificial_Scheduler/)
DB_FILE = os.path.join(BASE_DIR, "process_data.db")
MODEL_TAT_PATH = os.path.join(BASE_DIR, "modelForTAT.joblib")

# New paths (organized in folders)
DB_FILE = os.path.join(PROJECT_ROOT, "database", "process_data.db")
MODEL_TAT_PATH = os.path.join(PROJECT_ROOT, "models", "modelForTAT.joblib")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "ml_nice_adjuster.log")
```

### 3. **Automatic Backup Management**

- Model retraining automatically saves backups in the `models/` folder
- Log rotation keeps 3 backup files (10MB each)
- All generated files stay organized in their respective folders

### 4. **Benefits**

- **Cleaner structure**: Easy to find models, logs, and database files
- **Better organization**: Logical separation of concerns
- **Easier maintenance**: Clear file categorization
- **Professional layout**: Standard project organization
- **Backup safety**: All backups stored in appropriate folders

## Running the Application

The application paths are automatically configured. Simply run:

```bash
cd /home/gal/Projects/Artificial_Scheduler
python -m Artificial_Scheduler
```

## Development Notes

- All paths are automatically resolved relative to the project root
- Model retraining will save new models and backups to `models/` folder
- Log files automatically rotate in the `logs/` folder
- Database operations use the centralized database in `database/` folder

This structure follows Python packaging best practices and makes the project more maintainable and professional.
