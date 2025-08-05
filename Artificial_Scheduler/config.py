import logging
import os
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "ml_nice_adjuster.log")

# Configure rotating log handler (10MB max, keep 3 backup files)
handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=3
)
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Set up logger
log = logging.getLogger()
log.setLevel(logging.INFO)  # Set to INFO for production, DEBUG for development
log.addHandler(handler)

LABELS_DICT = {
    0: (-19, -16),  # Merge Sort (Highest priority)
    1: (-5, -2),   # IO
    2: (-1, 1),    # Bubble Sort
    3: (-15, -11),  # Matrix Manipulation
    4: (-10, -6)   # Heap Sort
}

TASKS_TO_RETRAIN = 5000

SKIP_HELPERS = {
    "tr", "uniq", "awk", "sed", "cut", "head", "tail", "sort", "true", "false",
    "watch", "manpath", "gitstatusd-linux", "xargs", "grep", "stty", "mkfifo", "gitstatusd-linu", "ldconfig",
    "sleep", "uname", "dirname", "basename", "readlink", "pwd", "git", "ldconfig.real",
    "find", "node", "env", "printf", "echo", "cat", "ps",
    "renice", "size", "readelf", "bash", "zsh", "python3", "python", "gcc", "sh", "docker", "docker-init", "docker-proxy", "docker-containerd-shim", "python3.10", "python3.11", "python3.12", "python3.13",
    "g++", "ld", "ld.bfd", "ld.gold", "ld-new", "ld-2.31.so",
}

DB_FILE = os.path.join(PROJECT_ROOT, "database", "process_data.db")
MODEL_TAT_PATH = os.path.join(PROJECT_ROOT, "models", "modelForTAT.joblib")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
KMEANS_PATH = os.path.join(PROJECT_ROOT, "models", "modelForPriorty.joblib")
HIGH_PRIORITY_LABEL = 0
SECONDARY_PRIORITY_LABEL = 3


def validate_config():
    """Validate that all required files and configurations are present."""
    required_files = {
        "MODEL_TAT_PATH": MODEL_TAT_PATH,
        "SCALER_PATH": SCALER_PATH,
        "KMEANS_PATH": KMEANS_PATH
    }

    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        raise FileNotFoundError(
            f"Required model files not found:\n" + "\n".join(missing_files)
        )

    # Validate LABELS_DICT structure
    if not isinstance(LABELS_DICT, dict):
        raise ValueError("LABELS_DICT must be a dictionary")

    for label, (start, end) in LABELS_DICT.items():
        if not isinstance(label, int) or label < 0:
            raise ValueError(f"Invalid label: {label}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"Invalid nice range for label {label}: ({start}, {end})")
        if start > end:
            raise ValueError(
                f"Invalid nice range for label {label}: start ({start}) > end ({end})")

    # Validate other settings
    if not isinstance(TASKS_TO_RETRAIN, int) or TASKS_TO_RETRAIN <= 0:
        raise ValueError(
            f"TASKS_TO_RETRAIN must be a positive integer, got: {TASKS_TO_RETRAIN}")

    if HIGH_PRIORITY_LABEL not in LABELS_DICT:
        raise ValueError(
            f"HIGH_PRIORITY_LABEL ({HIGH_PRIORITY_LABEL}) not found in LABELS_DICT")

    log.debug("[CONFIG] Configuration validation passed.")  # Convert to debug


# Validate configuration on import
try:
    validate_config()
except Exception as e:
    log.error(f"[CONFIG] Configuration validation failed: {e}")
    raise
