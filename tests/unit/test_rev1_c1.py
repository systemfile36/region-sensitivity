"""Unit tests for experiments/revision_1/c1_captum (command map, measurement helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.revision_1.c1_captum.measure_resources import entry_bytes, workflow_order, workflow_steps


def test_workflow_steps_follow_the_command_map(tmp_path: Path) -> None:
    captum = workflow_steps("captum", tmp_path)
    assert [step for step, _, _ in captum] == ["C-1", "C-2", "C-3"]
    assert [argv[2] for _, _, argv in captum] == ["audit", "analyze", "report"]
    ssat = workflow_steps("ssat", tmp_path)
    assert [step for step, _, _ in ssat] == [f"S-{index}" for index in range(1, 8)]
    assert [stage for _, stage, _ in ssat].count("audit") == 4
    assert all(argv[argv.index("--results-dir") + 1] == str(tmp_path) for _, _, argv in ssat)
    assert [argv[argv.index("--model") + 1] for _, _, argv in ssat if "--model" in argv] == ["shortcut", "normal"] * 2
    with pytest.raises(ValueError):
        workflow_steps("other", tmp_path)


def test_workflow_order_alternates() -> None:
    assert [workflow_order(repeat) for repeat in range(3)] == [("captum", "ssat"), ("ssat", "captum"), ("captum", "ssat")]


def test_entry_bytes(tmp_path: Path) -> None:
    (tmp_path / "raw" / "sub").mkdir(parents=True)
    (tmp_path / "raw" / "sub" / "a.parquet").write_bytes(b"x" * 7)
    (tmp_path / "report.md").write_bytes(b"y" * 3)
    (tmp_path / "accuracy.json").write_bytes(b"z" * 2)
    assert entry_bytes(tmp_path) == {"files": 5, "raw": 7, "total": 12}
