import os
import ctypes
import traceback
import psutil
import subprocess
from .elf_utils import get_elf_data
from .model import ModelManager
from .rebalance import pid_start_times, signal_process_exit
from .config import log, SKIP_HELPERS, HIGH_PRIORITY_LABEL
from .db import insert_into_db, should_retrain_model

model_manager = ModelManager()


class Data(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("comm", ctypes.c_char * 16)]


def handle_exec(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    name = event.comm.decode("utf-8")
    pid = event.pid
    try:
        tty = os.readlink(f"/proc/{pid}/fd/0")  # Read the terminal link
    except Exception:
        tty = None  # Could be permission denied or process has exited

    if not tty or (not tty.startswith("/dev/pts/") and not tty.startswith("/dev/tty")):
        # Remove this debug log - too noisy
        return

    if name in SKIP_HELPERS:
        return

    try:
        # Check if process still exists and is accessible
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            # Remove debug log - too noisy for zombie processes
            return

        binary_path = os.readlink(f"/proc/{pid}/exe")
        elf_info = get_elf_data(binary_path)
        psutil.cpu_percent(interval=0.1)
        proc.cpu_percent(interval=0.1)
        memory_info = proc.memory_info()
        memory_ext = proc.memory_full_info()
        io = proc.io_counters()
        proc_info = {
            "pid": pid,
            "name": name,
            "cpu_affinity": str(proc.cpu_affinity()),
            "cpu_num": proc.cpu_num(),
            "cpu_percent": proc.cpu_percent(),
            "cpu_times": str(proc.cpu_times()),
            "memory_rss": memory_info.rss,
            "memory_vms": memory_info.vms,
            "memory_shared": memory_info.shared,
            "memory_data": memory_ext.data,
            "io_read_count": io.read_count,
            "io_write_count": io.write_count,
            "io_read_bytes": io.read_bytes,
            "io_write_bytes": io.write_bytes,
            "open_files": len(proc.open_files()),
            "connections": len(proc.connections()),
            "threads": len(proc.threads()),
            "env_vars": len(proc.environ()),
            "status": proc.status(),
            **elf_info
        }

        nice_val = model_manager.predict_nice(
            proc_info, proc.create_time(), pid_start_times)

        # Apply renice with error handling
        try:
            result = subprocess.run(['renice', str(nice_val), '-p', str(pid)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    text=True, timeout=5)
            if result.returncode != 0:
                log.warning(
                    f"[RENICE] Failed to renice PID {pid}: {result.stderr}")
        except subprocess.TimeoutExpired:
            log.warning(f"[RENICE] Timeout renicing PID {pid}")
        except Exception as renice_error:
            log.warning(f"[RENICE] Error renicing PID {pid}: {renice_error}")

        # Convert to debug - this happens frequently
        log.debug(f"[RENICE] Applied nice={nice_val} to PID {pid} ({name})")
    except Exception as e:
        log.error(
            f"[EXEC] Failed for PID {pid}: {e}\n{traceback.format_exc()}")


def handle_exit(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    pid = event.pid

    if pid in pid_start_times:
        start_time, proc_info, nice_val, label, tat_predict = pid_start_times.pop(
            pid)
        if label == HIGH_PRIORITY_LABEL:
            model_manager.high_priority -= 1
        insert_into_db(start_time, nice_val, label, tat_predict, proc_info)
        signal_process_exit()  # Signal that a process exited
        # Convert to debug - this happens for every process exit
        log.debug(f"[EXIT] Process {pid} ended, removed from tracking.")


def retrain_model():
    if should_retrain_model():
        # Keep as info - important event
        log.info("[RETRAIN] Starting model retraining...")
        try:
            model_manager.incremental_retrain()
        except Exception as e:
            log.error(
                f"[RETRAIN] Failed to retrain model: {e}\n{traceback.format_exc()}")
    else:
        # Convert to debug - only log at info when actually retraining
        log.debug("[RETRAIN] No need to retrain model.")
