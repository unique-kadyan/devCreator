"""Structured logging. Every line carries job_id when there is one."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "asa"):
    return structlog.get_logger(name)
