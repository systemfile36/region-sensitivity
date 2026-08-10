"""Shared pytest configuration.

Makes ``tests/fixtures/`` importable as a plain module path (e.g.
``import synthetic_dump_builder``) from any test file, regardless of which
subdirectory collects it — ``tests/`` has no ``__init__.py`` files, so pytest
never puts ``tests/fixtures/`` on ``sys.path`` on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
