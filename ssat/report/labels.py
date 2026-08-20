"""A companion to the report exporter: export a per-sample risk-label file for downstream training.

Unlike ``ssat.report.exporter``, whose output is for humans to read, this
module's output is for a *training pipeline* to consume -- the same kind
of label file a prior research workflow produced with a script like
``build_skeleton_metadata.py``. It computes no new statistics -- every
value it writes is read verbatim off ``AssembledReport``'s already-computed
data, the same non-inferential posture the exporter has. It is documented
as a companion module rather than a new top-level report stage for that
reason.

**Two entry points, two callers.** :func:`export_risk_labels` takes the
assembler's in-memory ``AssembledReport``, while
``AuditApplication.export_labels(request)`` needs to read only an
already-written ``report_dir`` *without re-running the assembler*. Those
two requirements only both hold if everything :func:`export_risk_labels`
needs is also reachable from disk -- but the assembler's
``sample_semantic_degradation`` (``(sample_id, semantic_group) ->
degradation``) is a non-serialized intermediate with no ``ReportModel``
field of its own (``ssat.report.assembler.AssembledReport`` docstring), so
a raw ``report_model.json`` alone cannot satisfy it. The gap is closed by
``ssat.report.exporter`` folding that mapping into
``sample_rankings.csv``'s ``semantic_degradation_json`` column (per-sample,
one JSON object per row) -- so this module's second entry point,
:func:`load_assembled_report_for_labels`, rebuilds an
:class:`AssembledReportLike` purely from ``report_dir``'s already-written
``data/report_model.json`` + ``data/sample_rankings.csv``, with no
dump/metrics/analysis store ever reopened. Both entry points are
duck-typed against the same :class:`AssembledReportLike`, so
:func:`export_risk_labels` itself never needs to know which one produced
its input.

**risk_label's value.** For a binary primary metric, ``ItemMetrics.value``
is already 0/1 at the item grain, and ``SpatialProfile.degradation`` (the
source of ``sample_semantic_degradation``) is a plain arithmetic mean of
that within one ``(sample, semantic_group)`` pair -- unlike
``ClassSemanticRow.flip_rate`` (always ``None``, see ``ssat.report
.assembler`` module docstring), there is no missing-axis problem here
because this row is *already* scoped to one sample, not aggregated across
many. ``risk_label = degradation > 0`` therefore means "at least one
concrete region in this semantic_group flipped this sample's correct
prediction" -- the same flip semantics a prior research workflow used,
reused rather than re-derived. For a continuous primary metric this
threshold would misrepresent a continuous score as a binary label, so the
key is omitted entirely rather than guessed.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ssat.report.errors import ReportDataError
from ssat.report.types import MetricCard, ReportModel
from ssat.utils.io import load_json, sha256_file, write_json_atomic


class _SampleRowLike(Protocol):
    """Minimal per-sample shape :func:`export_risk_labels` reads."""

    sample_id: str
    gt_label: int | None
    clean_correct: bool | None


class AssembledReportLike(Protocol):
    """Structural shape this module reads, from either an in-memory ``AssembledReport``
    or :func:`load_assembled_report_for_labels`'s disk reconstruction (module docstring).

    Attributes:
        model: A fully-typed ``ReportModel`` (only ``.scorecard`` is read,
            to reuse the binary/continuous determination
            ``ssat.report.assembler._is_binary_primary_metric`` already made).
        full_sample_rankings: Every sample the run produced.
        sample_semantic_degradation: Every ``(sample_id, semantic_group)``
            pair with a determinable mean degradation.
    """

    model: ReportModel
    full_sample_rankings: Sequence[_SampleRowLike]
    sample_semantic_degradation: Mapping[tuple[str, str], float]


@dataclasses.dataclass(frozen=True, slots=True)
class LabelsManifest:
    """``labels_manifest.json`` content.

    Attributes:
        generated_at: ISO-8601 UTC timestamp this export ran at.
        source_report_manifest_hash: sha256 of the source ``report_manifest
            .json``, when known -- ``None`` when :func:`export_risk_labels`
            was called directly with an in-memory ``AssembledReport`` that
            has no on-disk ``report_manifest.json`` yet (e.g. before
            ``render_report`` has run), matching the rest of this codebase's
            convention of an explicit ``None`` over a fabricated hash.
        only_clean_correct: The filter this run applied.
        is_binary_primary_metric: Whether every row carries ``risk_label``.
        n_labels: Total emitted rows.
        n_positive: Rows with ``risk_label is True``; ``0`` when the primary
            metric is continuous (no row has the key at all).
        n_negative: Rows with ``risk_label is False``; ``0`` under the same
            condition as ``n_positive``.
        positive_rate: ``n_positive / n_labels``; ``None`` when the primary
            metric is continuous or no rows were emitted (never a silently
            divided-by-zero ``0.0``).
    """

    generated_at: str
    source_report_manifest_hash: str | None
    only_clean_correct: bool
    is_binary_primary_metric: bool
    n_labels: int
    n_positive: int
    n_negative: int
    positive_rate: float | None

    def __post_init__(self) -> None:
        """Validate that the counts are consistent with each other.

        Raises:
            ValueError: If any count is negative, positive+negative exceeds
                n_labels, or positive_rate is populated for a non-binary run.
        """

        if self.n_labels < 0 or self.n_positive < 0 or self.n_negative < 0:
            raise ValueError("n_labels/n_positive/n_negative must be non-negative")
        if self.n_positive + self.n_negative > self.n_labels:
            raise ValueError("n_positive + n_negative must not exceed n_labels")
        if not self.is_binary_primary_metric and (
            self.n_positive or self.n_negative or self.positive_rate is not None
        ):
            raise ValueError(
                "n_positive/n_negative/positive_rate must be 0/0/None when "
                "is_binary_primary_metric is False"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class LabelsResult:
    """Where :func:`export_risk_labels` wrote each of its output files.

    Attributes:
        labels_path: The JSONL body, always written.
        manifest_path: ``labels_manifest.json``, always written.
        csv_path: ``labels.csv``, only when ``write_csv=True``.
        manifest: The same content already written to ``manifest_path``, so
            a caller (e.g. the Application layer) never has to re-read it.
    """

    labels_path: Path
    manifest_path: Path
    csv_path: Path | None
    manifest: LabelsManifest


def export_risk_labels(
    assembled: AssembledReportLike,
    output_dir: Path,
    *,
    only_clean_correct: bool = True,
    write_csv: bool = False,
    source_report_manifest_hash: str | None = None,
) -> LabelsResult:
    """Write ``labels.jsonl`` (+ optional ``labels.csv``) + ``labels_manifest.json``.

    Args:
        assembled: The assembler's output, or
            :func:`load_assembled_report_for_labels`'s disk reconstruction
            of it -- see :class:`AssembledReportLike`.
        output_dir: Directory the files are written into; created
            (including parents) if it does not already exist.
        only_clean_correct: When ``True`` (the default), only samples with
            ``clean_correct is True`` are emitted -- reproducing the source
            research's convention of keeping only samples the teacher model
            correctly classified on clean input, so only corruption-
            *induced* failures ever become positive labels.
        write_csv: Also write ``labels.csv`` alongside the JSONL body.
        source_report_manifest_hash: sha256 of the source ``report_manifest
            .json``, when the caller has filesystem access to it (the
            Application layer always does); ``None`` otherwise.

    Returns:
        The written files' paths plus the manifest that was also persisted
        to ``manifest_path``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_binary = _is_binary_primary_metric(assembled.model.scorecard)

    semantic_degradation_by_sample: dict[str, dict[str, float]] = defaultdict(dict)
    for (sample_id, semantic_group), value in assembled.sample_semantic_degradation.items():
        semantic_degradation_by_sample[sample_id][semantic_group] = value

    rows: list[dict[str, object]] = []
    for card in assembled.full_sample_rankings:
        if only_clean_correct and card.clean_correct is not True:
            continue
        per_sample = semantic_degradation_by_sample.get(card.sample_id, {})
        for semantic_group in sorted(per_sample):
            degradation = per_sample[semantic_group]
            row: dict[str, object] = {
                "sample_id": card.sample_id,
                "gt_label": card.gt_label,
                "semantic_group": semantic_group,
                "degradation": degradation,
            }
            if is_binary:
                row["risk_label"] = degradation > 0
            rows.append(row)

    n_labels = len(rows)
    n_positive = sum(1 for row in rows if row.get("risk_label") is True) if is_binary else 0
    n_negative = sum(1 for row in rows if row.get("risk_label") is False) if is_binary else 0
    manifest = LabelsManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_report_manifest_hash=source_report_manifest_hash,
        only_clean_correct=only_clean_correct,
        is_binary_primary_metric=is_binary,
        n_labels=n_labels,
        n_positive=n_positive,
        n_negative=n_negative,
        positive_rate=(n_positive / n_labels) if is_binary and n_labels else None,
    )

    labels_path = output_dir / "labels.jsonl"
    _write_labels_jsonl(labels_path, rows, is_binary=is_binary, only_clean_correct=only_clean_correct)

    manifest_path = output_dir / "labels_manifest.json"
    write_json_atomic(manifest_path, dataclasses.asdict(manifest))

    csv_path: Path | None = None
    if write_csv:
        csv_path = output_dir / "labels.csv"
        _write_labels_csv(csv_path, rows, is_binary=is_binary)

    return LabelsResult(
        labels_path=labels_path, manifest_path=manifest_path, csv_path=csv_path, manifest=manifest
    )


# --- disk reconstruction (reads from disk without re-running the assembler) ----


@dataclasses.dataclass(frozen=True, slots=True)
class _MinimalSampleRow:
    """The exact ``_SampleRowLike`` fields recoverable from ``sample_rankings.csv``."""

    sample_id: str
    gt_label: int | None
    clean_correct: bool | None


@dataclasses.dataclass(frozen=True, slots=True)
class _DiskAssembledReport:
    """``AssembledReportLike`` reconstructed from ``report_dir``'s already-written files."""

    model: ReportModel
    full_sample_rankings: tuple[_MinimalSampleRow, ...]
    sample_semantic_degradation: Mapping[tuple[str, str], float]


def load_assembled_report_for_labels(
    report_dir: Path,
) -> tuple[AssembledReportLike, str | None]:
    """Rebuild enough of the assembler's output from ``report_dir`` to call :func:`export_risk_labels`.

    Reads ``report_dir/data/report_model.json`` and ``report_dir/data/
    sample_rankings.csv`` (both written by ``ssat.report.exporter.export``)
    plus ``report_dir/report_manifest.json`` (written by ``ssat.report.
    html_renderer.render_report``, always present alongside them --
    ``AuditApplication.generate_report`` runs both in the same pipeline) --
    never the source dump, MetricsStore, or AnalysisStore.

    Args:
        report_dir: The output directory a prior ``AuditApplication.
            generate_report()`` call wrote to.

    Returns:
        A tuple of ``(assembled_like, source_report_manifest_hash)`` --
        the second element is ``None`` only when ``report_manifest.json``
        itself is missing (an incomplete/corrupted report_dir), which is
        otherwise not fatal since :func:`export_risk_labels` never requires it.

    Raises:
        ReportDataError: If ``report_model.json``/``sample_rankings.csv`` is
            missing, or ``report_model.json`` predates the semantic-
            vulnerability fields (a genuinely old-format file --
            ``ReportModel.from_dict`` requires every field since none carry
            a default, so any missing top-level key surfaces here, not just
            the semantic ones) -- never a silent empty labels file.
    """

    report_dir = Path(report_dir)
    model_path = report_dir / "data" / "report_model.json"
    rankings_path = report_dir / "data" / "sample_rankings.csv"
    manifest_path = report_dir / "report_manifest.json"

    if not model_path.is_file():
        raise ReportDataError(
            f"report_model.json not found under {report_dir} -- run `ssat report` first"
        )
    if not rankings_path.is_file():
        raise ReportDataError(
            f"sample_rankings.csv not found under {report_dir} -- run `ssat report` first"
        )

    try:
        model = ReportModel.from_dict(load_json(model_path))
    except (KeyError, TypeError, ValueError) as error:
        raise ReportDataError(
            "this report was generated without semantic-vulnerability analysis "
            "(report_model.json predates the current schema); "
            "re-run `ssat report` to regenerate it before exporting labels"
        ) from error

    try:
        full_sample_rankings, sample_semantic_degradation = _read_sample_rankings_csv(
            rankings_path
        )
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise ReportDataError(
            f"sample_rankings.csv under {report_dir} is missing or has malformed "
            "columns this plan added -- re-run `ssat report` to regenerate it"
        ) from error

    source_report_manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None

    return (
        _DiskAssembledReport(
            model=model,
            full_sample_rankings=full_sample_rankings,
            sample_semantic_degradation=sample_semantic_degradation,
        ),
        source_report_manifest_hash,
    )


def _read_sample_rankings_csv(
    path: Path,
) -> tuple[tuple[_MinimalSampleRow, ...], dict[tuple[str, str], float]]:
    rows: list[_MinimalSampleRow] = []
    sample_semantic_degradation: dict[tuple[str, str], float] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        for record in csv.DictReader(file):
            sample_id = record["sample_id"]
            rows.append(
                _MinimalSampleRow(
                    sample_id=sample_id,
                    gt_label=_parse_optional_int(record["gt_label"]),
                    clean_correct=_parse_optional_bool(record["clean_correct"]),
                )
            )
            for semantic_group, degradation in json.loads(record["semantic_degradation_json"]).items():
                sample_semantic_degradation[(sample_id, semantic_group)] = degradation
    return tuple(rows), sample_semantic_degradation


def _parse_optional_int(raw: str) -> int | None:
    return int(raw) if raw != "" else None


def _parse_optional_bool(raw: str) -> bool | None:
    if raw == "":
        return None
    if raw == "True":
        return True
    if raw == "False":
        return False
    raise ValueError(f"unexpected boolean cell value: {raw!r}")


# --- helpers ---------------------------------------------------------------


def _is_binary_primary_metric(scorecard: Sequence[MetricCard]) -> bool:
    """Reuse the same ``flip_rate``-card check ``ssat.report.assembler.
    _is_binary_primary_metric`` uses, duplicated here rather than imported
    to keep this module's dependency surface at ``report.types``/``ssat
    .utils`` only -- the same boundary ``ssat.report.exporter`` already
    keeps, statically enforced by ``tests/unit/test_report_exporter.py::
    test_report_exporter_module_has_no_assembler_metrics_or_analysis_
    imports`` (this module has the same test).
    """

    return any(card.key == "flip_rate" and card.value is not None for card in scorecard)


def _write_labels_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    is_binary: bool,
    only_clean_correct: bool,
) -> None:
    """Write one JSON object per line: a leading meta line, then every row.

    The meta line lets a streaming consumer learn ``is_binary_primary_metric``
    (hence whether to expect a ``risk_label`` key on every subsequent line)
    without peeking at row 2 -- and carries an explanation of the
    continuous-run case as its ``note``. Every line, meta included, is a
    single self-contained JSON object, so a streaming parser can consume the
    file line by line without buffering the whole thing.
    """

    meta = {
        "_meta": True,
        "is_binary_primary_metric": is_binary,
        "only_clean_correct": only_clean_correct,
        "note": (
            None
            if is_binary
            else "continuous primary metric: risk_label omitted, degradation only"
        ),
    }
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(meta, sort_keys=True) + "\n")
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _write_labels_csv(path: Path, rows: Sequence[Mapping[str, object]], *, is_binary: bool) -> None:
    fieldnames = ["sample_id", "gt_label", "semantic_group", "degradation"]
    if is_binary:
        fieldnames.append("risk_label")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fieldnames})


def _cell(value: object) -> object:
    """Render one scalar CSV cell: ``None`` becomes ``""`` (matches ``ssat.report.exporter._cell``)."""

    return "" if value is None else value
