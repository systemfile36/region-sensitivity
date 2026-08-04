import hashlib
import io
import json
import logging
from pathlib import Path

import pytest

from ssat.utils.io import load_json, load_yaml, sha256_file, write_json_atomic
from ssat.utils.logger_factory import configure_logging, get_logger


def test_yaml_and_json_loaders(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    json_path = tmp_path / "manifest.json"
    yaml_path.write_text("name: audit\ncount: 2\n", encoding="utf-8")
    json_path.write_text('{"name": "audit", "count": 2}', encoding="utf-8")

    assert load_yaml(yaml_path) == {"name": "audit", "count": 2}
    assert load_json(json_path) == {"name": "audit", "count": 2}


def test_sha256_file_reads_in_chunks(tmp_path: Path) -> None:
    path = tmp_path / "mask.bin"
    content = b"region-mask" * 100
    path.write_bytes(content)

    assert sha256_file(path, chunk_size=7) == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sha256_file(path, chunk_size=0)


def test_write_json_atomic_replaces_document(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    write_json_atomic(path, {"version": 1, "name": "감사"})
    write_json_atomic(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert not list(path.parent.glob(".*.tmp"))


def test_write_json_atomic_preserves_existing_file_on_serialization_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 1}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(path, {"invalid": object()})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert not list(tmp_path.glob(".*.tmp"))


def test_get_logger_uses_ssat_namespace() -> None:
    assert get_logger().name == "ssat"
    assert get_logger("core.config").name == "ssat.core.config"
    assert get_logger("ssat.core.plan").name == "ssat.core.plan"


def test_configure_logging_uses_utc_and_optional_file(tmp_path: Path) -> None:
    stream = io.StringIO()
    log_path = tmp_path / "audit.log"
    root_handlers = tuple(logging.getLogger().handlers)
    log_path.write_text("existing\n", encoding="utf-8")
    configure_logging(level="INFO", log_file=log_path, stream=stream)

    get_logger("unit").info("config.test event=true")
    rendered = stream.getvalue()
    assert "Z INFO ssat.unit config.test event=true" in rendered
    assert log_path.read_text(encoding="utf-8") == f"existing\n{rendered}"
    assert tuple(logging.getLogger().handlers) == root_handlers


def test_reconfiguration_does_not_duplicate_managed_handlers() -> None:
    stream = io.StringIO()
    configure_logging(stream=stream)
    configure_logging(stream=stream)

    get_logger("unit").warning("logging.single")
    assert stream.getvalue().count("logging.single") == 1


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown logging level"):
        configure_logging(level="LOUD")
