import threading
import time
import os
import subprocess
from .config import LABELS_DICT, log

pid_start_times = {}  # shared state, can be moved into a ProcessTracker class


def rebalance_nice():
    if not pid_start_times:
        log.info("[REBALANCE] No processes to rebalance.")
        return

    now = time.time()

    for label in range(5):
        nice_start, nice_end = LABELS_DICT[label]
        nice_range = list(range(nice_start, nice_end + 1))
        filtered = [
            (pid, data) for pid, data in pid_start_times.items() if data[3] == label
        ]
        sorted_list = sorted(
            filtered,
            key=lambda item: now - item[1][4] -
            item[1][0]  # expected_time_left
        )

        for i, (pid, data) in enumerate(sorted_list[:len(nice_range)]):
            new_nice = nice_range[i]
            creation_time, proc_info, _, label, tat_predict = data
            pid_start_times[pid] = (
                creation_time, proc_info, new_nice, label, tat_predict)
            # os.system(f"renice -n {new_nice} -p {pid}")
            subprocess.run(['renice',str(new_nice),'-p',str(pid)],stdout=subprocess.DEVNULL)

            log.info(f"[REBALANCE] Set nice={new_nice} for PID {pid}")


# Timer-based loop
rebalance_timer = None


def loop_rebalance():
    global rebalance_timer
    rebalance_nice()
    rebalance_timer = threading.Timer(15, loop_rebalance)
    rebalance_timer.start()


def cancel_rebalance():
    if rebalance_timer:
        rebalance_timer.cancel()
