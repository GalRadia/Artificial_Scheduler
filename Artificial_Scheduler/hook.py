import json
from bcc import BPF
import psutil
import faulthandler
import os
import ctypes
import subprocess
import logging
import pickle
import pandas as pd
import numpy as np
import sklearn.preprocessing
import random
import traceback
import sqlite3
import time
import threading
import xgboost as xgb
from xgboost import XGBRegressor
from joblib import load, dump

# -------- Logger Setup --------
faulthandler.enable()
LOG_FILE = "ml_nice_adjuster.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger()

# -------- Initialize Model --------
modelTAT = xgb.Booster()

try:
    # with open('modelForTAT.pkl', 'rb') as f:
        # modelTAT = pickle.load(f)
        # modelTAT.get_booster().save_model("modelForTAT.json")  # save
    modelTAT.load_model("modelForTAT.json")  # load

        # booster.load_model("modelForTAT.json")  # load

    # with open('scaler.pkl', 'rb') as f:
        # scaler = pickle.load(f)
        # dump(scaler, 'scaler.joblib')  # save
    scaler = load('scaler.joblib')

    # with open('modelForPriorty.pkl', 'rb') as f:
        # best_kmeans = pickle.load(f)
        # dump(best_kmeans, 'modelForPriorty.joblib')  # save
    best_kmeans = load('modelForPriorty.joblib')

except Exception as e:
    log.error(f"Failed to load ML models: {e}")
    print(
        f"Error: Failed to load ML models.  Ensure they are in the correct directory.  Exiting.  Error Details: {e}")
    exit(1)


# -------- Initialize SQLite DB --------
DB_FILE = "process_data.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
try:

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processes (
        id INTEGER PRIMARY KEY,
        pid INTEGER,
        name TEXT,
        cpu_affinity TEXT,
        cpu_num INTEGER,
        cpu_percent FLOAT,
        cpu_times TEXT,
        nice INTEGER,
        start_time REAL,
        end_time REAL,
        tat REAL,
        memory_vms INTEGER,
        memory_data INTEGER,
        memory_rss INTEGER,
        memory_shared INTEGER,
        io_read_count INTEGER,
        io_write_count INTEGER,
        io_read_bytes INTEGER,
        io_write_bytes INTEGER,
        open_files INTEGER,
        connections INTEGER,
        threads INTEGER,
        status TEXT,
        env_vars INTEGER,
        bss INTEGER,
        comment INTEGER,
        data INTEGER,
        dynamic INTEGER,
        dynstr INTEGER,
        dynsym INTEGER,
        eh_frame INTEGER,
        eh_frame_hdr INTEGER,
        fini INTEGER,
        fini_array INTEGER,
        gnu_version INTEGER,
        gnu_version_r INTEGER,
        gnu_hash INTEGER,
        got INTEGER,
        got_plt INTEGER,
        init INTEGER,
        init_array INTEGER,
        interp INTEGER,
        jcr INTEGER,
        plt INTEGER,
        note_ABI_tag INTEGER,
        note_gnu_build_id INTEGER,
        rela_dyn INTEGER,
        rela_plt INTEGER,
        shstrtab INTEGER,
        strtab INTEGER,
        text INTEGER,
        rodata INTEGER,
        symtab INTEGER,                   
        input_size INTEGER
    )
    ''')
    conn.commit()
except Exception as e:
    log.error(f"Failed to initialize database: {e}")
    print(
        f"Error: Failed to initialize database.  Exiting. Error Details: {e}")
    exit(1)

# -------- Track PID start times --------
pid_start_times = {}
# labels_dict = {
#     0: (5, 12),
#     1: (13, 18),
#     2: (-11, -4),
#     3: (-19, -12),
#     4: (-3, 4)
# }
labels_dict = {
    0: (-5, -2),
    1: (-1, 1),
    2: (-15, -11),
    3: (-19, -16),
    4: (-10, -6)
}
high_priority = 0
# -------- ML Model --------


def get_nice_from_model(proc_info, creation_time):
    df = pd.DataFrame([proc_info])
    df = df.drop(columns=['pid',])

    df_model1 = df[['memory_vms', 'memory_data', 'io_write_count',
                    'bss', 'data', 'dynamic', 'dynstr', 'eh_frame', 'eh_frame_hdr',
                    'fini', 'fini_array', 'gnu.version', 'gnu.version_r', 'got',
                    'init', 'init_array', 'plt', 'rela.dyn', 'rela.plt',
                    'shstrtab', 'strtab', 'text', 'input_size']].copy()
    df_model1['vmsXmemorydata'] = df_model1['memory_vms'] * \
        df_model1['memory_data']
    df_model1['squaredmemorydata'] = df_model1['memory_data'] ** 2
    df_model1['memorydata'] = df_model1['memory_data'] ** 4
    df_model1['memory_vmslog'] = np.sqrt(df_model1['memory_vms'])
    df_model1['memory_datasin'] = np.sin(df_model1['memory_data']) ** 2
    df_model1["textXmemory_data"] = np.log(df_model1["memory_data"]) ** 2

    df_model1_scaled = scaler.fit_transform(df_model1)

    dmatrix_input = xgb.DMatrix(df_model1_scaled)
    tat_predict = modelTAT.predict(dmatrix_input)

    # tat_predict = modelTAT.predict(df_model1_scaled)

    df['tat_predicted'] = tat_predict

    # lb = sklearn.preprocessing.LabelEncoder()
    # df["status"] = lb.fit_transform(df["status"])
    df['io_write_count']= df['io_write_count'] * 250
    desired_columns = [
        'io_read_count', 'io_write_count', 'io_write_bytes', 'bss', 'data',
        'dynamic', 'dynstr', 'eh_frame', 'eh_frame_hdr', 'fini', 'fini_array',
        'gnu.version', 'gnu.version_r', 'got', 'init', 'init_array', 'plt',
        'rela.dyn', 'rela.plt', 'shstrtab', 'strtab', 'text', 'input_size',
        'tat_predicted'
    ]
    df = df[[col for col in desired_columns if col in df.columns]]
    df['io_read_count']
    label = best_kmeans.predict(df)
    if label == 3:
        global high_priority
        if high_priority == 3:
            label = 2
        else:
            high_priority += 1
    log.info(f"[ML] KMeans label: {label}")
    label = label[0]
    nice_val = get_nice(label)
    pid_start_times[proc_info['pid']] = (
        creation_time, proc_info, nice_val, label, tat_predict)
    log.info(f"[ML] Predicted nice value: {nice_val}")
    return nice_val

# -------- ELF Section Size Parser --------


def get_nice(label):
    # Adjust the nice value based on the label and predicted TAT
    if label > 4 or label < 0:
        log.error(f"[ML] Invalid label: {label}. Defaulting to 0.")
        return 0
    # Adjust the nice value based on the label
    return random.randint(labels_dict[label][0], labels_dict[label][1])
# -------- ELF Data Extraction --------


def get_elf_data(file_path):
    elf_data = {
        'comment': 0, 'dynamic': 0, 'dynstr': 0,
        'dynsym': 0, 'eh_frame': 0, 'eh_frame_hdr': 0, 'fini': 0, 'fini_array': 0,
        'gnu.version': 0, 'gnu.version_r': 0, 'gnu.hash': 0, 'got': 0, 'got.plt': 0,
        'init': 0, 'init_array': 0, 'interp': 0, 'jcr': 0, 'note.ABI-tag': 0,
        'note.gnu.build-id': 0, 'plt': 0, 'rela.dyn': 0, 'rela.plt': 0,
        'rodata': 0, 'shstrtab': 0, 'strtab': 0, 'symtab': 0
    }

    try:
        result = subprocess.run(
            ['readelf', '-S', file_path], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            for key in elf_data.keys():
                if key in line:
                    parts = line.split()
                    if len(parts) > 4:
                        try:
                            elf_data[key] = int(parts[4], 16)
                        except ValueError:
                            continue
    except Exception as e:
        log.error(f"[ELF] Failed to parse with readelf: {e}")

    try:
        result = subprocess.run(['size', file_path],
                                capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 4:
                try:
                    elf_data['text'] = int(parts[0])
                    elf_data['data'] = int(parts[1])
                    elf_data['bss'] = int(parts[2])
                    elf_data['input_size'] = int(parts[3])
                except ValueError:
                    pass
    except Exception as e:
        log.error(f"[ELF] Failed to parse with size: {e}")

    log.info(f"[ELF] Parsed ELF data for {file_path}: {elf_data}")
    return elf_data


# -------- eBPF Programs --------
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(events_exec);
BPF_PERF_OUTPUT(events_exit);

int on_exec(struct tracepoint__sched__sched_process_exec *ctx) {
    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events_exec.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int on_exit(struct tracepoint__sched__sched_process_exit *ctx) {
    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events_exit.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
int on_fork(struct tracepoint__sched__sched_process_fork *ctx) {
    struct data_t data = {};
    data.pid = ctx->child_pid;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events_exec.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

# -------- Load eBPF --------
try:
    b = BPF(text=bpf_text, cflags=["-Wno-duplicate-decl-specifier"])
    b.attach_tracepoint(tp=b"sched:sched_process_exec", fn_name=b"on_exec")
    b.attach_tracepoint(tp=b"sched:sched_process_exit", fn_name=b"on_exit")
    b.attach_tracepoint(tp=b"sched:sched_process_fork", fn_name=b"on_fork")

except Exception as e:
    log.error(f"Failed to load or attach eBPF program: {e}")
    print(
        f"Error: Failed to load or attach eBPF program.  Exiting.  Error Details: {e}")
    exit(1)

# -------- C struct for events --------


class Data(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint),
                ("comm", ctypes.c_char * 16)]

# -------- Handlers --------


def handle_exec(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    name = event.comm.decode("utf-8")
    pid = event.pid
    # skip_names = {"renice", "sh", "size", "readelf","cat", "bash", "python3", "python", "gcc", "g++", "ld", "ld.bfd", "ld.gold", "ld-new", "ld-2.31.so","sed","ps"}
    skip_names = {"renice", "sh", "size", "readelf","zsh"}
    if name in skip_names:
        return

    try:
        proc = psutil.Process(pid)
        if proc.terminal() and proc.name() not in skip_names:
            print(f"[eBPF] Target process: {name} (PID {pid})")
            log.info(f"[eBPF] Detected new process: {name} (PID {pid})")

            try:
                binary_path = os.readlink(f"/proc/{pid}/exe")
                elf_info = get_elf_data(binary_path)

                psutil.cpu_percent(interval=0.1)
                proc.cpu_percent(interval=0.1)

                memory_info = proc.memory_info()
                memory_ext_info = proc.memory_full_info()
                io_counters = proc.io_counters()
                open_files = proc.open_files()
                threads = proc.threads()
                env_vars = proc.environ()
                status = proc.status()
                connections = proc.connections()

                proc_info = {
                    "pid": pid,
                    "name": name,
                    "binary_path": binary_path,
                    "cpu_affinity": json.dumps(proc.cpu_affinity()),
                    "cpu_num": proc.cpu_num(),
                    "cpu_percent": proc.cpu_percent(),
                    "cpu_times": json.dumps(proc.cpu_times()),
                    "memory_rss": memory_info.rss,
                    "memory_vms": memory_info.vms,
                    "memory_shared": memory_info.shared,
                    "memory_data": memory_ext_info.data,
                    "io_read_count": io_counters.read_count,
                    "io_write_count": io_counters.write_count,
                    "io_read_bytes": io_counters.read_bytes,
                    "io_write_bytes": io_counters.write_bytes,
                    "open_files": len(open_files),
                    "connections": len(connections),
                    "threads": len(threads),
                    "env_vars": len(env_vars),
                    "status": status,
                    **elf_info
                }

                log.info(f"[INFO] Process info: {proc_info}")

                nice_val = get_nice_from_model(proc_info, proc.create_time())
                os.system(f"renice -n {nice_val} -p {pid}")
                # nice_val=0
                log.info(
                    f"[RENICE] Applied nice={nice_val} to PID {pid} name={name}")
                print(f"[+] Set nice={nice_val} for PID {pid}")

                # start_time = proc.create_time()
                # creation_time, proc_info, nice_val, label, tat_predict
                # pid_start_times[pid] = (start_time, proc_info, nice_val,-1,-1)

            except FileNotFoundError:
                log.warning(
                    f"[WARNING] Could not access /proc/{pid}/exe for {name} (PID {pid}). Skipping ELF analysis.")
            except psutil.NoSuchProcess:
                log.warning(
                    f"[WARNING] Process {name} (PID {pid}) disappeared before full info could be collected.")
            except Exception as e:
                error_msg = f"[ERROR] Failed to get detailed info for PID {pid} ({name}): {e}" + traceback.format_exc(
                )
                log.error(error_msg)
                print(error_msg)

    except psutil.NoSuchProcess:
        log.warning(
            f"[WARNING] Process {name} (PID {pid}) disappeared immediately after exec.")
    except Exception as e:
        error_msg = f"[ERROR] Failed to handle exec PID {pid} ({name}): {e}" + \
            traceback.format_exc()
        log.error(error_msg)
        print(error_msg)


def handle_exit(cpu, data, size):
    event = ctypes.cast(data, ctypes.POINTER(Data)).contents
    pid = event.pid

    if pid in pid_start_times:
        end_time = time.time()
        start_time, proc_info, nice_val, label, tat_predict = pid_start_times.pop(
            pid)
        if label == 3:
            global high_priority
            high_priority -= 1
        tat = end_time - start_time

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO processes (
                    pid, name, cpu_affinity, cpu_num, cpu_percent, cpu_times, nice, start_time, end_time, tat,
                    memory_vms, memory_data, memory_rss, memory_shared,
                    io_read_count, io_write_count, io_read_bytes, io_write_bytes,
                    open_files, connections, threads,
                    status, env_vars,
                    bss, data, dynamic, dynstr, dynsym, eh_frame, comment, interp,
                    eh_frame_hdr, fini, fini_array, gnu_version, gnu_version_r, gnu_hash,
                    got, init, init_array, plt, got_plt, rela_dyn, rela_plt,
                    shstrtab, jcr, note_ABI_tag, note_gnu_build_id, rodata,
                    strtab, symtab, text, input_size
                ) VALUES (
                    :pid, :name, :cpu_affinity, :cpu_num, :cpu_percent, :cpu_times, :nice, :start_time, :end_time, :tat,
                    :memory_vms, :memory_data, :memory_rss, :memory_shared,
                    :io_read_count, :io_write_count, :io_read_bytes, :io_write_bytes,
                    :open_files, :connections, :threads,
                    :status, :env_vars,
                    :bss, :data, :dynamic, :dynstr, :dynsym, :eh_frame, :comment, :interp,
                    :eh_frame_hdr, :fini, :fini_array, :gnu_version, :gnu_version_r, :gnu_hash,
                    :got, :init, :init_array, :plt, :got_plt, :rela_dyn, :rela_plt,
                    :shstrtab, :jcr, :note_ABI_tag, :note_gnu_build_id, :rodata,
                    :strtab, :symtab, :text, :input_size
                )
            ''', {
                'pid': pid,
                'name': proc_info['name'] + " without_ml",
                'cpu_affinity': proc_info['cpu_affinity'],
                'cpu_num': proc_info['cpu_num'],
                'cpu_percent': proc_info['cpu_percent'],
                'cpu_times': proc_info['cpu_times'],
                'nice': nice_val,
                'start_time': start_time,
                'end_time': end_time,
                'tat': tat,
                'memory_vms': proc_info['memory_vms'],
                'memory_data': proc_info['memory_data'],
                'memory_rss': proc_info['memory_rss'],
                'memory_shared': proc_info['memory_shared'],
                'io_read_count': proc_info['io_read_count'],
                'io_write_count': proc_info['io_write_count'],
                'io_read_bytes': proc_info['io_read_bytes'],
                'io_write_bytes': proc_info['io_write_bytes'],
                'open_files': proc_info['open_files'],
                'connections': proc_info['connections'],
                'threads': proc_info['threads'],
                'status': proc_info['status'],
                'env_vars': proc_info['env_vars'],
                'bss': proc_info['bss'],
                'data': proc_info['data'],
                'dynamic': proc_info['dynamic'],
                'dynstr': proc_info['dynstr'],
                'dynsym': proc_info['dynsym'],
                'eh_frame': proc_info['eh_frame'],
                'comment': proc_info['comment'],
                'interp': proc_info['interp'],
                'eh_frame_hdr': proc_info['eh_frame_hdr'],
                'fini': proc_info['fini'],
                'fini_array': proc_info['fini_array'],
                'gnu_version': proc_info['gnu.version'],
                'gnu_version_r': proc_info['gnu.version_r'],
                'gnu_hash': proc_info['gnu.hash'],
                'got': proc_info['got'],
                'init': proc_info['init'],
                'init_array': proc_info['init_array'],
                'plt': proc_info['plt'],
                'got_plt': proc_info['got.plt'],
                'rela_dyn': proc_info['rela.dyn'],
                'rela_plt': proc_info['rela.plt'],
                'shstrtab': proc_info['shstrtab'],
                'jcr': proc_info['jcr'],
                'note_ABI_tag': proc_info['note.ABI-tag'],
                'note_gnu_build_id': proc_info['note.gnu.build-id'],
                'rodata': proc_info['rodata'],
                'strtab': proc_info['strtab'],
                'symtab': proc_info['symtab'],
                'text': proc_info['text'],
                'input_size': proc_info['input_size'],
            })
            conn.commit()

            print(f"[+] Stored process {pid} with TaT {tat:.2f}s in database.")
        except Exception as e:
            log.error(f"Failed to insert process data into database: {e}")
            print(
                f"Error: Failed to insert process data into database. Error Details: {e}")


def reballance_nice():
    # sum_nice = sum(pid_start_times[pid][2] for pid in pid_start_times)
    # if sum_nice > 0:
    #     pid = max(pid_start_times, key=lambda pid: pid_start_times[pid][2])
    #     _, _, nice_val = pid_start_times[pid]
    #     new_nice = max(-20, min(19, nice_val - 1))
    #     pid_start_times[pid] = (pid_start_times[pid][0],
    #                             pid_start_times[pid][1], new_nice)
    #     os.system(f"renice -n {new_nice} -p {pid}")
    #     log.info(f"[REBALANCE] Adjusted nice={new_nice} for PID {pid}")
    #     print(f"[+] Rebalanced nice={new_nice} for PID {pid}")
    if not pid_start_times:
        log.info("[REBALANCE] No processes to rebalance.")
        return
    now = time.time()
    sorted_labels = {}
    nice_sum = sum(data[2] for data in pid_start_times.values())
    # Define nice ranges for each label

    for label in range(5):
        nice_start, nice_end = labels_dict[label]

        # Generate sorted nice values from low to high
        nice_range = list(range(nice_start, nice_end + 1)
                          ) if nice_start <= nice_end else list(range(nice_start, nice_end - 1, -1))

        # Filter for the current label
        filtered = [
            (pid, data) for pid, data in pid_start_times.items() if data[3] == label
        ]

        # Sort by expected_time_left = now - tat_predict - creation_time
        sorted_list = sorted(
            filtered,
            key=lambda item: now - item[1][4] - item[1][0]
        )

        sorted_labels[label] = sorted_list
        log.info(
            f"[REBALANCE] Sorted PIDs for label {label}: {[pid for pid, _ in sorted_list]}")

        # Rebalance nice values
        for i, (pid, data) in enumerate(sorted_list):
            if i >= len(nice_range):
                break  # More processes than nice slots — skip extras
            new_nice = nice_range[i]

            # Update pid_start_times with new nice
            creation_time, proc_info, _, label, tat_predict = data
            pid_start_times[pid] = (
                creation_time, proc_info, new_nice, label, tat_predict)

            # Apply system-level nice change
            os.system(f"renice -n {new_nice} -p {pid}")
            log.info(f"[REBALANCE] Set nice={new_nice} for PID {pid}")
            print(f"[+] Rebalanced PID {pid} to nice={new_nice}")


rebalance_timer = None


def loop_reballance():
    global rebalance_timer
    reballance_nice()
    rebalance_timer = threading.Timer(15, loop_reballance)
    rebalance_timer.start()


# -------- Main Loop --------
print("🔍 Listening for target processes...")
log.info("Started ML nice adjuster daemon.")


b["events_exec"].open_perf_buffer(handle_exec)

b["events_exit"].open_perf_buffer(handle_exit)

# Start the rebalancing loop
# loop_reballance()
# Polling loop
try:
    while True:
        b.perf_buffer_poll()


except KeyboardInterrupt:
    if rebalance_timer:
        rebalance_timer.cancel()
    print("Exiting.")
    log.info("Shutting down.")
    conn.close()
    print("connection closed.")
    log.info("Closing database connection.")
    b.detach_tracepoint(tp=b"sched:sched_process_exec")
    b.detach_tracepoint(tp=b"sched:sched_process_exit")
    b.detach_tracepoint(tp=b"sched:sched_process_fork")
    print("eBPF tracepoints detached.")
    log.info("eBPF tracepoints detached.")
    b.cleanup()
    print("eBPF program cleaned up.")
    log.info("eBPF program cleaned up.")
    log.info("Log file closed.")
    print("Exiting gracefully.")

except Exception as e:
    log.error(f"Unhandled exception: {e}")
    print(f"Error: Unhandled exception.  Exiting. Error Details: {e}")
    conn.close()
    exit(1)
    #     pid, name, cpu_affinity, cpu_num, cpu_percent, cpu_times, nice, start_time, end_time, tat,
    #     memory_vms, memory_data, memory_rss, io_read_count, io_write_count, io_read_bytes, io_write_bytes, open_files, connections, threads,
    #     status, env_vars, bss, data, dynamic, dynstr, dynsym, eh_frame, comment, interp,
    #     eh_frame_hdr, fini, fini_array, gnu_version, gnu_version_r, gnu.hash,
    #     got, init, init_array, plt, got.plt, rela_dyn, rela_plt, shstrtab, jcr, note.ABI-tag, note.gnu.build-id, rodata,
    #     strtab, symtab, text, input_size
    # ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) 52
