"""Centralized logging configuration for the agent pipeline."""

import logging
import os
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "musicrec") -> logging.Logger:
    """Return a logger that writes to both stdout and logs/agent.log.

    Idempotent: calling it multiple times does not add duplicate handlers.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if _CONFIGURED:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parent.parent
    log_dir = repo_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "agent.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(
        logging.DEBUG if os.getenv("MUSICREC_VERBOSE") else logging.WARNING
    )
    logger.addHandler(stream_handler)

    _CONFIGURED = True
    return logger
