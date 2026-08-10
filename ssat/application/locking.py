"""Advisory single-writer locking for audit dump directories."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Iterator, TextIO


class OutputLockedError(RuntimeError):
    """Indicate that another process owns the dump writer lock."""


@contextmanager
def output_lock(output: Path) -> Iterator[None]:
    """Hold a nonblocking sibling lock for one output path."""

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{output.name}.ssat.lock"
    handle: TextIO = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OutputLockedError(f"audit output is locked: {output}") from error
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
