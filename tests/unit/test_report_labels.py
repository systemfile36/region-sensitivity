"""Unit tests for ssat.report.labels (design plan IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md §3.5).

Reuses ``test_report_exporter``'s ``_report_model``/``_sample_card`` builders
(the same cross-test-module reuse ``test_metrics_store.py`` already
establishes for ``test_metrics_aggregate._FakeMetric``) rather than
duplicating a second full ``ReportModel`` fixture.
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import json
from pathlib import Path

import pytest

from ssat.report.assembler import AssembledReport
from ssat.report.errors import ReportDataError
from ssat.report.exporter import export as export_report
from ssat.report.labels import (
    LabelsManifest,
    export_risk_labels,
    load_assembled_report_for_labels,
)
from ssat.report.types import MetricCard, ReportModel
from ssat.utils.io import sha256_file

from test_report_exporter import _read_csv_rows, _report_model, _sample_card

_BINARY_SCORECARD = (
    MetricCard(key="accuracy", label="Clean Accuracy", value=0.9, unit="%", higher_is_better=True),
    MetricCard(key="flip_rate", label="Flip Rate", value=0.3, unit="%", higher_is_better=False),
)
_CONTINUOUS_SCORECARD = (
    MetricCard(key="accuracy", label="Clean Accuracy", value=0.9, unit="%", higher_is_better=True),
    MetricCard(
        key="flip_rate",
        label="Flip Rate",
        value=None,
        unit="%",
        higher_is_better=False,
        note="해당 없음: continuous 지표",
    ),
)


def _model(*, binary: bool) -> ReportModel:
    return dataclasses.replace(
        _report_model(), scorecard=_BINARY_SCORECARD if binary else _CONTINUOUS_SCORECARD
    )


def _assembled(
    *,
    binary: bool = True,
    cards: tuple = (
        _sample_card(sample_id="s0", clean_correct=True),
        _sample_card(sample_id="s1", clean_correct=False),
    ),
    sample_semantic_degradation: dict[tuple[str, str], float] | None = None,
) -> AssembledReport:
    return AssembledReport(
        model=_model(binary=binary),
        full_sample_rankings=cards,
        sample_semantic_degradation=sample_semantic_degradation or {},
    )


# --- clean_correct filtering (plan §3.5 item 2) -------------------------------


def test_export_risk_labels_excludes_clean_incorrect_by_default(tmp_path: Path) -> None:
    assembled = _assembled(
        sample_semantic_degradation={("s0", "upper_limb"): 0.4, ("s1", "upper_limb"): 0.9}
    )

    result = export_risk_labels(assembled, tmp_path)

    lines = result.labels_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]  # skip the leading meta line
    assert [row["sample_id"] for row in rows] == ["s0"]
    assert result.manifest.n_labels == 1


def test_export_risk_labels_includes_clean_incorrect_when_opted_out(tmp_path: Path) -> None:
    assembled = _assembled(
        sample_semantic_degradation={("s0", "upper_limb"): 0.4, ("s1", "upper_limb"): 0.9}
    )

    result = export_risk_labels(assembled, tmp_path, only_clean_correct=False)

    lines = result.labels_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    assert [row["sample_id"] for row in rows] == ["s0", "s1"]
    assert result.manifest.n_labels == 2


# --- binary vs continuous risk_label (plan §3.5 item 3) ------------------------


def test_export_risk_labels_binary_includes_risk_label_and_meta_note_is_none(
    tmp_path: Path,
) -> None:
    assembled = _assembled(
        binary=True,
        cards=(_sample_card(sample_id="s0", clean_correct=True),),
        sample_semantic_degradation={
            ("s0", "upper_limb"): 0.4,
            ("s0", "lower_limb"): 0.0,
        },
    )

    result = export_risk_labels(assembled, tmp_path)

    lines = result.labels_path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])
    assert meta == {
        "_meta": True,
        "is_binary_primary_metric": True,
        "only_clean_correct": True,
        "note": None,
    }
    rows = {row["semantic_group"]: row for row in (json.loads(line) for line in lines[1:])}
    assert rows["upper_limb"]["risk_label"] is True
    assert rows["lower_limb"]["risk_label"] is False


def test_export_risk_labels_continuous_omits_risk_label_and_meta_explains_why(
    tmp_path: Path,
) -> None:
    assembled = _assembled(
        binary=False,
        cards=(_sample_card(sample_id="s0", clean_correct=True),),
        sample_semantic_degradation={("s0", "upper_limb"): 0.4},
    )

    result = export_risk_labels(assembled, tmp_path)

    lines = result.labels_path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])
    assert meta["is_binary_primary_metric"] is False
    assert meta["note"] == "continuous primary metric: risk_label omitted, degradation only"
    row = json.loads(lines[1])
    assert "risk_label" not in row
    assert row["degradation"] == 0.4
    assert result.manifest.n_positive == 0
    assert result.manifest.n_negative == 0
    assert result.manifest.positive_rate is None


# --- labels_manifest.json positive/negative accounting -------------------------


def test_labels_manifest_positive_negative_counts_match_hand_calculation(tmp_path: Path) -> None:
    assembled = _assembled(
        binary=True,
        cards=(
            _sample_card(sample_id="s0", clean_correct=True),
            _sample_card(sample_id="s1", clean_correct=True),
        ),
        sample_semantic_degradation={
            ("s0", "upper_limb"): 0.4,   # risk_label True
            ("s0", "lower_limb"): 0.0,   # risk_label False
            ("s1", "upper_limb"): 0.9,   # risk_label True
        },
    )

    result = export_risk_labels(assembled, tmp_path)

    assert result.manifest.n_labels == 3
    assert result.manifest.n_positive == 2
    assert result.manifest.n_negative == 1
    assert result.manifest.positive_rate == pytest.approx(2 / 3)

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["n_positive"] == 2
    assert manifest_payload["n_negative"] == 1


# --- JSONL is a valid stream of standalone objects (plan §3.5 테스트) ----------


def test_labels_jsonl_every_line_is_independently_parseable_json(tmp_path: Path) -> None:
    assembled = _assembled(
        sample_semantic_degradation={("s0", "upper_limb"): 0.4, ("s0", "lower_limb"): 0.1}
    )

    result = export_risk_labels(assembled, tmp_path)

    lines = result.labels_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # 1 meta + 2 rows
    for line in lines:
        json.loads(line)  # must not raise


# --- determinism ---------------------------------------------------------------


def test_export_risk_labels_is_byte_identical_across_repeated_calls(tmp_path: Path) -> None:
    assembled = _assembled(
        sample_semantic_degradation={("s0", "upper_limb"): 0.4, ("s0", "lower_limb"): 0.1}
    )

    first = export_risk_labels(assembled, tmp_path / "first", write_csv=True)
    second = export_risk_labels(assembled, tmp_path / "second", write_csv=True)

    # labels.jsonl/labels.csv carry no timestamp, so they must be exactly
    # byte-identical; labels_manifest.json's generated_at is real wall-clock
    # time by design (the same reason ComputeMetricsResult.computed_at/
    # AnalyzeResult.computed_at are, elsewhere in this codebase) and is
    # compared field-by-field instead, excluding that one field.
    for name in ("labels_path", "csv_path"):
        first_bytes = getattr(first, name).read_bytes()
        second_bytes = getattr(second, name).read_bytes()
        assert first_bytes == second_bytes, name

    assert dataclasses.replace(first.manifest, generated_at="") == dataclasses.replace(
        second.manifest, generated_at=""
    )


# --- labels.csv (--csv, plan §3.5 item 5) --------------------------------------


def test_write_csv_flag_writes_labels_csv_with_matching_rows(tmp_path: Path) -> None:
    assembled = _assembled(
        binary=True,
        cards=(_sample_card(sample_id="s0", clean_correct=True),),
        sample_semantic_degradation={("s0", "upper_limb"): 0.4},
    )

    result = export_risk_labels(assembled, tmp_path, write_csv=True)

    assert result.csv_path is not None
    rows = _read_csv_rows(result.csv_path)
    assert rows == [
        {
            "sample_id": "s0",
            "gt_label": "0",
            "semantic_group": "upper_limb",
            "degradation": "0.4",
            "risk_label": "True",
        }
    ]


def test_write_csv_false_by_default(tmp_path: Path) -> None:
    result = export_risk_labels(_assembled(), tmp_path)

    assert result.csv_path is None


# --- load_assembled_report_for_labels (plan §5 "R0를 다시 돌리지 않음") --------


def test_load_assembled_report_for_labels_round_trips_from_exporter_output(
    tmp_path: Path,
) -> None:
    assembled = _assembled(
        binary=True,
        cards=(
            _sample_card(sample_id="s0", gt_label=0, clean_correct=True),
            _sample_card(sample_id="s1", gt_label=1, clean_correct=False),
        ),
        sample_semantic_degradation={("s0", "upper_limb"): 0.4, ("s1", "lower_limb"): 0.7},
    )
    report_dir = tmp_path / "report"
    export_report(assembled, report_dir / "data")
    (report_dir / "report_manifest.json").write_bytes(b'{"marker": "v1"}')

    loaded, source_report_manifest_hash = load_assembled_report_for_labels(report_dir)

    assert loaded.model == assembled.model
    assert [(row.sample_id, row.gt_label, row.clean_correct) for row in loaded.full_sample_rankings] == [
        ("s0", 0, True),
        ("s1", 1, False),
    ]
    assert dict(loaded.sample_semantic_degradation) == {
        ("s0", "upper_limb"): 0.4,
        ("s1", "lower_limb"): 0.7,
    }
    assert source_report_manifest_hash == sha256_file(report_dir / "report_manifest.json")


def test_load_assembled_report_for_labels_hash_is_none_when_report_manifest_missing(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    export_report(_assembled(), report_dir / "data")

    _loaded, source_report_manifest_hash = load_assembled_report_for_labels(report_dir)

    assert source_report_manifest_hash is None


def test_load_assembled_report_for_labels_end_to_end_feeds_export_risk_labels(
    tmp_path: Path,
) -> None:
    assembled = _assembled(
        binary=True,
        cards=(_sample_card(sample_id="s0", clean_correct=True),),
        sample_semantic_degradation={("s0", "upper_limb"): 0.4},
    )
    report_dir = tmp_path / "report"
    export_report(assembled, report_dir / "data")
    (report_dir / "report_manifest.json").write_bytes(b'{"marker": "v1"}')

    loaded, report_manifest_hash = load_assembled_report_for_labels(report_dir)
    result = export_risk_labels(
        loaded, tmp_path / "labels", source_report_manifest_hash=report_manifest_hash
    )

    assert result.manifest.n_labels == 1
    assert result.manifest.source_report_manifest_hash == report_manifest_hash


def test_load_assembled_report_for_labels_raises_when_report_model_missing(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    (report_dir / "data").mkdir(parents=True)

    with pytest.raises(ReportDataError, match="report_model.json not found"):
        load_assembled_report_for_labels(report_dir)


def test_load_assembled_report_for_labels_rejects_report_model_that_predates_this_plan(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    data_dir = report_dir / "data"
    data_dir.mkdir(parents=True)
    payload = json.loads(json.dumps(dataclasses.asdict(_report_model())))
    del payload["semantic_concentration"]  # simulate a pre-plan report_model.json
    (data_dir / "report_model.json").write_text(json.dumps(payload), encoding="utf-8")
    (data_dir / "sample_rankings.csv").write_text("sample_id\n", encoding="utf-8")

    with pytest.raises(ReportDataError, match="predates"):
        load_assembled_report_for_labels(report_dir)


# --- import boundary (mirrors test_report_exporter's, plan "패키지 위치") ------


def test_report_labels_module_has_no_assembler_metrics_or_analysis_imports() -> None:
    """Statically enforce §3.3-style boundary: report.labels → report.types, report.errors, ssat.utils."""

    source_path = Path(__file__).resolve().parents[2] / "ssat" / "report" / "labels.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_prefixes = (
        "ssat.report.assembler",
        "ssat.report.adapters",
        "ssat.report.exporter",
        "ssat.analysis",
        "ssat.metrics",
        "ssat.core",
        "ssat.application",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith(forbidden_prefixes), module
