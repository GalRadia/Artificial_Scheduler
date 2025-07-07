import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = "ml_nice_adjuster.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

LABELS_DICT = {
    0: (-5, -2), # Merge Sort
    1: (-1, 1), # IO
    2: (-15, -11), # Bubble Sort
    3: (-19, -16), # Matrix Manipulation
    4: (-10, -6) # Heap Sort
}


SKIP_HELPERS = {
    "tr", "uniq", "awk", "sed", "cut", "head", "tail", "sort", "true", "false",
    "watch", "manpath", "gitstatusd-linux", "xargs", "grep", "stty", "mkfifo","gitstatusd-linu","ldconfig",
    "sleep", "uname", "dirname", "basename", "readlink", "pwd", "git","ldconfig.real"
    "find", "node", "env", "printf", "echo", "cat", "ps",
    "renice", "size", "readelf", "bash", "zsh", "python3", "python", "gcc","sh","docker","docker-init","docker-proxy","docker-containerd-shim","python3.10", "python3.11", "python3.12", "python3.13",
    "g++", "ld", "ld.bfd", "ld.gold", "ld-new", "ld-2.31.so",
}

DB_FILE = os.path.join(BASE_DIR, "process_data.db")
MODEL_TAT_PATH = os.path.join(BASE_DIR, "modelForTAT.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
KMEANS_PATH = os.path.join(BASE_DIR, "modelForPriorty.joblib")
HIGH_PRIORITY_LABEL = 3
