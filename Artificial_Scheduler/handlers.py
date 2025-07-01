import os
import ctypes
import traceback
import psutil
import subprocess
from .elf_utils import get_elf_data
from .model import ModelManager
from .rebalance import pid_start_times
from .config import log, SKIP_HELPERS, HIGH_PRIORITY_LABEL
from .db import insert_into_db

model_manager = ModelManager()


class Data(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("comm", ctypes.c_char * 16)]


def handle_exec(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    name = event.comm.decode("utf-8")
    pid = event.pid
    try:
        tty = os.readlink(f"/proc/{pid}/fd/0")
    except Exception:
        tty = None  # Could be permission denied or process has exited

    if not tty or (not tty.startswith("/dev/pts/") and not tty.startswith("/dev/tty")):
        log.debug(
            f"[DEBUG] Skipping non-interactive process: {name} (PID {pid})")
        return

    if name in SKIP_HELPERS:
        return

    try:

        proc = psutil.Process(pid)
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
        subprocess.run(['renice',str(nice_val),'-p',str(pid)],stdout=subprocess.DEVNULL)
        # os.system(f"renice -n {nice_val} -p {pid}")

        # nice_val = 0
        # pid_start_times[proc_info['pid']] = (
        #     proc.create_time(), proc_info, nice_val, -1, -1)

        log.info(f"[RENICE] Applied nice={nice_val} to PID {pid} ({name})")
    except Exception as e:
        log.error(
            f"[EXEC] Failed for PID {pid}: {e}\n{traceback.format_exc()}")


def handle_exit(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    pid = event.pid

    if pid in pid_start_times:
        start_time, proc_info, nice_val ,label,tat_predict= pid_start_times.pop(pid) 
        if label == HIGH_PRIORITY_LABEL:
            model_manager.high_priority -= 1
        insert_into_db(start_time,nice_val,label,tat_predict,proc_info)
        log.info(f"[EXIT] Process {pid} ended, removed from tracking.")
