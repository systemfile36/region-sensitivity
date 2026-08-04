"""Central logging configuration for the SSAT logger hierarchy."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TextIO

LOGGER_NAME = "ssat"
LOG_FORMAT = "%(asctime)sZ %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_MANAGED_HANDLER_ATTRIBUTE = "_ssat_managed_handler"


def _mark_managed(handler: logging.Handler) -> logging.Handler:
    """Mark a handler so repeated configuration can replace it safely."""

    setattr(handler, _MANAGED_HANDLER_ATTRIBUTE, True)
    return handler


def _base_logger() -> logging.Logger:
    """Return the package logger without changing the root logger."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    return logger


def _install_null_handler() -> None:
    """Keep library imports silent until the application configures logging."""

    logger = _base_logger()
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(_mark_managed(logging.NullHandler()))


def _resolve_level(level: int | str) -> int:
    """Normalize a logging level name or integer."""

    if isinstance(level, bool):
        raise ValueError("logging level must be an integer or standard level name")
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    raise ValueError(f"unknown logging level: {level!r}")


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger within the shared ``ssat`` hierarchy.

    Args:
        name: Module-style suffix or a fully qualified ``ssat`` logger name.

    Returns:
        The requested package logger. Output remains disabled until configured.
    """

    if not name or name == LOGGER_NAME:
        return _base_logger()
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str = "INFO",
    log_file: str | Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure UTC console logging and optional append-only file logging.

    Repeated calls replace only handlers created by this module. Handlers added
    by embedding applications are preserved, and the root logger is untouched.

    Args:
        level: Standard logging level name or integer value.
        log_file: Optional UTF-8 file that receives the same formatted records.
        stream: Optional console stream, primarily useful for embedding/tests.

    Returns:
        The configured top-level ``ssat`` logger.

    Raises:
        ValueError: If ``level`` is not a recognized logging level.
    """

    resolved_level = _resolve_level(level)
    logger = _base_logger()

    for handler in tuple(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    formatter.converter = time.gmtime

    console_handler = _mark_managed(logging.StreamHandler(stream or sys.stderr))
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _mark_managed(logging.FileHandler(path, mode="a", encoding="utf-8"))
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(resolved_level)
    return logger


_install_null_handler()

