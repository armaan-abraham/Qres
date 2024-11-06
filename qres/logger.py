import logging
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path

import psutil

from qres.config import config


def log_system_stats():
    """Logs current system resource usage."""
    process = psutil.Process()
    logging.info(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    logging.info(f"CPU Usage: {process.cpu_percent()}%")
    logging.info(f"Open Files: {len(process.open_files())}")
    logging.info(f"Active Threads: {threading.active_count()}")


def handle_signal(signum, frame, logger):
    """Signal handler for logging various termination signals."""
    signal_dict = {
        signal.SIGABRT: "SIGABRT",  # Abort signal
        signal.SIGALRM: "SIGALRM",  # Timer signal
        signal.SIGBUS: "SIGBUS",  # Bus error (bad memory access)
        signal.SIGCHLD: "SIGCHLD",  # Child process terminated, stopped, or resumed
        signal.SIGCONT: "SIGCONT",  # Continue executing if stopped
        signal.SIGFPE: "SIGFPE",  # Floating point exception
        signal.SIGHUP: "SIGHUP",  # Hangup
        signal.SIGILL: "SIGILL",  # Illegal instruction
        signal.SIGINT: "SIGINT",  # Keyboard interrupt
        signal.SIGIO: "SIGIO",  # I/O event
        signal.SIGKILL: "SIGKILL",  # Kill (cannot be caught)
        signal.SIGPIPE: "SIGPIPE",  # Broken pipe
        signal.SIGPROF: "SIGPROF",  # Profiling timer expired
        signal.SIGPWR: "SIGPWR",  # Power failure
        signal.SIGQUIT: "SIGQUIT",  # Quit program
        signal.SIGSEGV: "SIGSEGV",  # Segmentation fault
        signal.SIGSYS: "SIGSYS",  # Bad system call
        signal.SIGTERM: "SIGTERM",  # Termination request
        signal.SIGTRAP: "SIGTRAP",  # Trace/breakpoint trap
        signal.SIGTSTP: "SIGTSTP",  # Stop signal generated from keyboard
        signal.SIGTTIN: "SIGTTIN",  # Background process attempting read
        signal.SIGTTOU: "SIGTTOU",  # Background process attempting write
        signal.SIGURG: "SIGURG",  # Urgent condition on socket
        signal.SIGUSR1: "SIGUSR1",  # User-defined signal 1
        signal.SIGUSR2: "SIGUSR2",  # User-defined signal 2
        signal.SIGVTALRM: "SIGVTALRM",  # Virtual timer expired
        signal.SIGXCPU: "SIGXCPU",  # CPU time limit exceeded
        signal.SIGXFSZ: "SIGXFSZ",  # File size limit exceeded
    }

    signal_name = signal_dict.get(signum, f"Unknown signal {signum}")
    logger.error(f"Received {signal_name}")
    log_system_stats()

    # For signals that typically indicate serious issues, log extra info
    serious_signals = {
        signal.SIGSEGV,
        signal.SIGABRT,
        signal.SIGFPE,
        signal.SIGILL,
        signal.SIGBUS,
        signal.SIGSYS,
        signal.SIGXCPU,
        signal.SIGXFSZ,
    }

    if signum in serious_signals:
        logger.error("Stack trace at time of signal:")
        logger.error("".join(traceback.format_stack(frame)))

    # Some signals we might want to handle gracefully
    if signum in {signal.SIGTERM, signal.SIGINT, signal.SIGQUIT}:
        logger.info("Initiating graceful shutdown...")
        sys.exit(0)

    # Re-raise the signal after logging
    signal.default_int_handler(signum, frame)


def monitor_resources(logger):
    """Monitor system resources and log if they exceed thresholds."""
    while True:
        try:
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent}%")

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")

            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Resource monitoring error: {str(e)}")


def setup_logging(logging_dir: Path):
    """Sets up comprehensive process monitoring and logging."""

    handlers = [logging.StreamHandler(sys.stdout)]

    if config.save_enabled:
        handlers.append(logging.FileHandler(logging_dir / f"train.log"))

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    logger = logging.getLogger()
    logger.info("Starting process monitoring")
    logger.info(f"Local timezone: {datetime.now().astimezone().tzinfo}")

    # Set up handlers for all signals except SIGKILL and SIGSTOP
    # (which cannot be caught)
    for signame in dir(signal):
        if signame.startswith("SIG") and signame not in ["SIGKILL", "SIGSTOP"]:
            try:
                signum = getattr(signal, signame)
                if isinstance(signum, int):  # Ensure it's a valid signal number
                    signal.signal(signum, partial(handle_signal, logger=logger))
            except (ValueError, RuntimeError) as e:
                # Some signals might not be supported on all systems
                logger.debug(f"Could not set handler for {signame}: {e}")

    # Set up system exception hook
    def exception_hook(exc_type, exc_value, exc_traceback):
        logger.error(
            "Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback)
        )
        log_system_stats()
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = exception_hook

    return logger
