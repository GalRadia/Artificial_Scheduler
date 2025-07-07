from .ebpf_utils import setup_ebpf, detach_ebpf, poll_events
from .rebalance import loop_rebalance,cancel_rebalance
from .db import close_db,init_db
from .config import log
from .handlers import loop_retrain

if __name__ == "__main__":
    log.info("Started ML nice adjuster daemon.")
    print("Started ML nice adjuster daemon.")
    try:
        init_db()
        setup_ebpf()
        loop_rebalance()
        log.debug("Starting retrain loop.")
        # loop_retrain()  # Start the retrain loop
        log.debug("end of retrain loop.")
        poll_events()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, shutting down.")
        log.info("Detaching eBPF programs.")
        detach_ebpf()
        log.info("Cancelling rebalance loop.")
        cancel_rebalance()
        log.info("Closing database connection.")
        close_db()
        log.info("Shutdown complete.")
        print("Exiting.")
    except Exception as e:
        log.error(f"Unhandled exception: {e}")
        detach_ebpf()
        cancel_rebalance()
        close_db()        
        exit(1)
