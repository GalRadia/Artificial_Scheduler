from .ebpf_utils import setup_ebpf, detach_ebpf, poll_events
from .rebalance import loop_rebalance, cancel_rebalance
from .db import close_db, init_db
from .config import log
from .handlers import retrain_model
import signal
import sys


def cleanup():
    """Cleanup function with individual error handling."""
    log.debug("Starting cleanup...")  # Convert to debug

    try:
        detach_ebpf()
        log.debug("eBPF programs detached.")  # Convert to debug
    except Exception as e:
        log.error(f"Error detaching eBPF: {e}")

    try:
        cancel_rebalance()
        log.debug("Rebalance cancelled.")  # Convert to debug
    except Exception as e:
        log.error(f"Error cancelling rebalance: {e}")

    try:
        close_db()
        log.debug("Database connection closed.")  # Convert to debug
    except Exception as e:
        log.error(f"Error closing database: {e}")

    log.debug("Cleanup complete.")  # Convert to debug


def signal_handler(signum, frame):
    log.info(f"Received signal {signum}, performing cleanup.")
    cleanup()
    sys.exit(0)


if __name__ == "__main__":
    log.info("Started ML nice adjuster daemon.")
    print("Started ML nice adjuster daemon.")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize components with individual error handling
    try:
        # Convert to debug - not critical for users to see
        log.debug("Initializing database...")
        init_db()
        log.info("Database initialized.")  # Keep important milestone
    except Exception as e:
        log.error(f"Failed to initialize database: {e}")
        print(f"Error: Failed to initialize database: {e}")
        exit(1)

    try:
        log.debug("Setting up eBPF...")  # Convert to debug
        setup_ebpf()
        log.info("eBPF setup complete.")  # Keep important milestone
    except Exception as e:
        log.error(f"Failed to setup eBPF: {e}")
        print(f"Error: Failed to setup eBPF: {e}")
        cleanup()
        exit(1)

    try:
        log.debug("Starting rebalance loop...")  # Convert to debug
        loop_rebalance()
        log.info("Rebalance loop started.")  # Keep important milestone
    except Exception as e:
        log.error(f"Failed to start rebalance loop: {e}")
        print(f"Error: Failed to start rebalance loop: {e}")
        cleanup()
        exit(1)

    try:
        # Convert to debug
        log.debug("Performing initial model retrain check...")
        retrain_model()
        log.debug("Model retrain check complete.")  # Convert to debug
    except Exception as e:
        log.error(f"Failed during model retrain: {e}")
        print(f"Warning: Model retrain failed: {e}")
        # Don't exit here, continue with polling

    try:
        # Convert to debug - not critical
        log.debug("Starting event polling...")
        poll_events()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received, shutting down.")
        print("Shutting down...")
    except Exception as e:
        log.error(f"Unhandled exception during polling: {e}")
        print(f"Error: {e}")
    finally:
        cleanup()
        exit(0)
