import threading
import time
import subprocess
from psutil import pid_exists
from .config import LABELS_DICT, log

pid_start_times = {}  # shared state, can be moved into a ProcessTracker class

# Event-driven rebalancing system
rebalance_event = threading.Event()
rebalance_thread = None
should_stop = threading.Event()


def rebalance_nice():
    if not pid_start_times:
        return False  # Indicate no work was done

    now = time.time()
    work_done = False

    # Remove verbose debug log for every rebalance

    for label in range(5):
        nice_start, nice_end = LABELS_DICT[label]
        nice_range = list(range(nice_start, nice_end + 1))
        filtered = [
            (pid, data) for pid, data in pid_start_times.items() if data[3] == label
        ]

        if not filtered:
            continue

        # Remove debug log - too verbose for each label iteration

        # Sort by expected time left (tat_predict - elapsed_time)
        sorted_list = sorted(
            filtered,
            # tat_predict - elapsed_time
            key=lambda item: item[1][4] - (now - item[1][0])
        )

        for i, (pid, data) in enumerate(sorted_list[:len(nice_range)]):
            new_nice = nice_range[i]
            creation_time, proc_info, old_nice, label, tat_predict = data
            elapsed_time = now - creation_time
            remaining_time = tat_predict - elapsed_time

            # Remove debug log - too verbose for each process

            if old_nice != new_nice:  # Only renice if changed
                # Check if process still exists before renicing
                try:
                    if not pid_exists(pid):
                        # Remove debug log - too noisy
                        pid_start_times.pop(pid, None)
                        continue
                except Exception as e:
                    log.warning(f"[REBALANCE] Error checking PID {pid}: {e}")
                    continue

                pid_start_times[pid] = (
                    creation_time, proc_info, new_nice, label, tat_predict)

                try:
                    result = subprocess.run(['renice', str(new_nice), '-p', str(pid)],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                            text=True, timeout=5)
                    if result.returncode != 0:
                        log.warning(
                            f"[REBALANCE] Failed to renice PID {pid}: {result.stderr}")
                        continue
                except subprocess.TimeoutExpired:
                    log.warning(f"[REBALANCE] Timeout renicing PID {pid}")
                    continue
                except Exception as renice_error:
                    log.warning(
                        f"[REBALANCE] Error renicing PID {pid}: {renice_error}")
                    continue

                # Convert to debug - happens frequently
                log.debug(f"[REBALANCE] Set nice={new_nice} for PID {pid}")
                work_done = True
            else:
                # Remove debug log - too verbose
                pass

    # Remove debug log - too verbose
    return work_done


def rebalance_worker():
    """Worker thread that handles rebalancing."""
    log.info("[REBALANCE] Worker thread started.")

    while not should_stop.is_set():
        # If no processes, wait indefinitely for new processes
        if not pid_start_times:
            # Convert to debug - too verbose
            log.debug("[REBALANCE] No processes, waiting for new processes...")
            rebalance_event.wait()  # Block until signaled
            rebalance_event.clear()  # Clear the event after waking up

            if should_stop.is_set():
                break

            # After new process signal, do immediate rebalance
            if pid_start_times:
                rebalance_nice()
        else:

            rebalance_nice()
            # Wait for 15 seconds OR until signaled for new process
            signaled = rebalance_event.wait(timeout=15)

            if signaled:
                rebalance_event.clear()
                # Convert to debug - happens frequently
                log.debug(
                    "[REBALANCE] New process detected during wait, rebalancing immediately...")
                if pid_start_times:
                    rebalance_nice()

    log.info("[REBALANCE] Worker thread stopped.")


def signal_new_process():
    """Call this when a new process is added."""
    # Convert to debug - happens very frequently
    log.debug("[REBALANCE] New process detected, signaling rebalance.")
    rebalance_event.set()


def signal_process_exit():
    """Call this when a process exits."""
    # Convert to debug - happens very frequently
    log.debug("[REBALANCE] Process exit detected, checking for rebalance.")
    if pid_start_times:  # If still have processes
        rebalance_event.set()


def loop_rebalance():
    """Start the rebalancing system."""
    global rebalance_thread
    if rebalance_thread and rebalance_thread.is_alive():
        return

    should_stop.clear()
    rebalance_thread = threading.Thread(target=rebalance_worker, daemon=True)
    rebalance_thread.start()


def cancel_rebalance():
    """Stop the rebalancing system."""
    global rebalance_thread
    should_stop.set()
    rebalance_event.set()  # Wake up the worker

    if rebalance_thread and rebalance_thread.is_alive():
        rebalance_thread.join(timeout=2)
        log.info("[REBALANCE] Stopped rebalancing.")
