from bcc import BPF
from .handlers import handle_exec, handle_exit
from .config import log

bpf_text = """
#pragma clang diagnostic ignored "-Wduplicate-decl-specifier"
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct data_t {
    u32 pid;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(events_exec);
BPF_PERF_OUTPUT(events_exit);

// This function is called when a process is executed
int on_exec(struct tracepoint__sched__sched_process_exec *ctx) {
    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events_exec.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// This function is called when a process exits
int on_exit(struct tracepoint__sched__sched_process_exit *ctx) {
    struct data_t data = {};
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events_exit.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

b = None


def setup_ebpf():
    global b
    b = BPF(text=bpf_text)
    b.attach_tracepoint(tp=b"sched:sched_process_exec", fn_name=b"on_exec")
    b.attach_tracepoint(tp=b"sched:sched_process_exit", fn_name=b"on_exit")
    b["events_exec"].open_perf_buffer(handle_exec)
    b["events_exit"].open_perf_buffer(handle_exit)
    log.info("[eBPF] Tracepoints attached.")


def poll_events():
    while True:
        if b is not None:
            b.perf_buffer_poll()
        else:
            break


def detach_ebpf():
    if b:
        b.detach_tracepoint(tp=b"sched:sched_process_exec")
        b.detach_tracepoint(tp=b"sched:sched_process_exit")
        b.cleanup()
        log.info("[eBPF] Cleaned up and detached.")
