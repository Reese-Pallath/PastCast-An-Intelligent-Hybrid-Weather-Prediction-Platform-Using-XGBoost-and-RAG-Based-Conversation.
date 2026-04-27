"""
Structured logging configuration for PastCast backend.
Replaces all print() calls with proper leveled, timestamped logging.
"""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines for production log aggregation."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable log format for development."""

    FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-24s │ %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FORMAT, datefmt="%H:%M:%S")


def setup_logging(level: str = "INFO", fmt: str = "text"):
    """
    Configure the root logger for the application.

    Parameters
    ----------
    level : str
        Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    fmt : str
        "json" for structured JSON output, "text" for human-readable.
    """
    root = logging.getLogger()

    # Remove any existing handlers to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quieten noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return root
