"""Contract types for the reporting layer (``ReportModel``, design §3).

``ReportModel`` is the single source of truth this whole package assembles
into, exports as JSON, and renders as HTML (REPORT_LAYER_DESIGN_v1.md §0
"JSON 모델이 먼저이고 HTML은 그 뷰 중 하나다"). It is deliberately the *only*
thing R1 (Exporter), R2 (ChartRenderer), R3 (AssetLinker), and R4
(HTMLRenderer) consume — none of them recompute statistics, they only
select, sort, and visualize what is already here.

This module has zero dependencies, including on ``ssat.analysis.types`` and
``ssat.metrics.types`` (IMPLE_PLAN_REPORTING_v1.md §5 단계0, mirroring the
same zero-dependency rule ``ssat.analysis.types`` already follows for the
same reason — see that module's docstring). Concretely this means:

- ``AnchorKey`` is never carried through; ``FlaggedItem.anchor_key_repr``
  flattens it to ``f"{sample_id}::{region_key}::{invert_mask}"`` instead.
- ``ssat.analysis.types.ReliabilityGrade`` is not reused. ``ReportGrade``
  below re-declares the same four string values as an independent ``Enum``
  so a caller reading only this module never needs to know the analysis
  package exists.
- ``ssat.core.types.RegionKind`` is not reused either; ``RegionRow.
  region_kind`` is a plain ``str``.

The payoff (REPORT_LAYER_DESIGN_v1.md §4.6): the day a WebUI wants to serve
``ReportModel`` as an HTTP response body instead of a file, this module does
not change — every field is already a JSON-primitive, a plain ``str``/
``int``/``float``/``bool``/``None``, a tuple (JSON array), or a ``dict``
(JSON object). ``json.dumps(dataclasses.asdict(model), sort_keys=True)``
serializes any value built from this module without a custom encoder,
because every ``Enum`` here mixes in ``str``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, UnionType
from typing import Any, TypeAlias, get_args, get_origin, get_type_hints


REPORT_SCHEMA_VERSION = "1.0.0"


# --- JSON value type (mirrors ssat.core.types.JsonValue without importing it) -------

ReportJsonScalar: TypeAlias = str | int | float | bool | None
ReportJsonValue: TypeAlias = ReportJsonScalar | list["ReportJsonValue"] | dict[str, "ReportJsonValue"]


# --- Shared enums ----------------------------------------------------------


class ReportGrade(str, Enum):
    """Reliability label surfaced to the report, independently of the analysis module.

    Values are kept string-identical to ``ssat.analysis.types.
    ReliabilityGrade`` so a human reading a rendered report never sees a
    mismatch, but this is a separate ``Enum`` — see module docstring.

    Attributes:
        HIGH: Reproduced reliably across every check the analysis module ran.
        MODERATE: Reproduced with some, but not all, checks satisfied.
        LOW: Sign is consistent but the effect is marginal or under-evidenced.
        UNRELIABLE: Direction flips across conditions — the grade design
            treats as most important to surface (§4.3 "숫자 옆에 등급 배지").
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNRELIABLE = "unreliable"


_REPORT_GRADE_VALUES = frozenset(grade.value for grade in ReportGrade)

GRADE_COLORS: Mapping[ReportGrade, str] = MappingProxyType(
    {
        ReportGrade.HIGH: "#2e7d32",  # green — design §4.3 "HIGH(녹)"
        ReportGrade.MODERATE: "#1565c0",  # blue — design §4.3 "MODERATE(청)"
        ReportGrade.LOW: "#757575",  # gray — design §4.3 "LOW(회)"
        ReportGrade.UNRELIABLE: "#c62828",  # red — design §4.3 "UNRELIABLE(적)"
    }
)
"""Shared hex colors for each :class:`ReportGrade` (design §4.3 badge colors).

Lives here — rather than in ``report.charts`` or a future ``report.static``
— because ``report.types`` is the only module both already depend on
(§3.3: ``report.charts → report.types``, ``report.html_renderer →
report.types, report.static``); a color duplicated independently in each
would drift the moment one side changes, breaking §4.3's own requirement
that a grade reads the same color whether it appears on a chart bar or an
HTML badge chip.
"""


class TaskKind(str, Enum):
    """Which prediction task this report was assembled for (design §3, §5).

    Attributes:
        CLASSIFICATION: Rendered end-to-end in v1.
        DETECTION: Schema-level placeholder only in v1 (design §5) — a
            ``ReportModel`` may declare this ``task_kind`` but R0/R4 do not
            yet populate/render detection-specific content.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"


# --- Leaf row types ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricCard:
    """One task-agnostic scorecard entry (design §R5).

    Produced by a ``TaskPresentationAdapter``; ``ReportModel``/R4 never
    branch on task kind, they only ever iterate a list of these.

    Attributes:
        key: Stable machine identifier, e.g. ``"accuracy"``.
        label: Human-readable title, e.g. ``"Clean Accuracy"``.
        value: The card's headline number; ``None`` when not applicable to
            this run (e.g. ``flip_rate`` for a continuous primary metric) —
            distinct from omitting the card, so the reason can be shown via
            ``note`` instead of the card silently disappearing (design §6.2
            "결측이... 조용히 생략되지 않음").
        unit: Display unit, e.g. ``"%"``; empty string when unitless.
        higher_is_better: Whether a larger ``value`` is the desired direction.
        note: Optional human-readable caveat or "해당 없음" explanation.
    """

    key: str
    label: str
    value: float | None
    unit: str
    higher_is_better: bool
    note: str | None = None

    def __post_init__(self) -> None:
        """Validate identity fields.

        Raises:
            ValueError: If key or label is empty.
        """

        if not self.key or not self.label:
            raise ValueError("key and label must not be empty")


@dataclass(frozen=True, slots=True)
class TopRegionEntry:
    """One region's degradation surfaced inside a ``SampleCard`` (design §3).

    Attributes:
        region_key: Stable concrete-region identity.
        degradation: ``SpatialProfile.degradation`` for this sample/region,
            when available — sourced identically to what R3's heatmap
            renders, so the card's text and image never contradict each
            other (IMPLE_PLAN_REPORTING_v1.md §1 격차#4).
        reliability_grade: This anchor's grade, when an analysis run exists;
            ``None`` means "not evaluated", never a stand-in for a poor grade.
    """

    region_key: str
    degradation: float | None
    reliability_grade: ReportGrade | None

    def __post_init__(self) -> None:
        """Validate identity and the optional grade's type.

        Raises:
            TypeError: If reliability_grade is present and not a ReportGrade.
            ValueError: If region_key is empty.
        """

        if not self.region_key:
            raise ValueError("region_key must not be empty")
        if self.reliability_grade is not None and not isinstance(
            self.reliability_grade, ReportGrade
        ):
            raise TypeError("reliability_grade must be a ReportGrade or None")


@dataclass(frozen=True, slots=True)
class SampleCard:
    """One sample's vulnerability, evidence, and grade (design §3, §4.4).

    Attributes:
        sample_id: Owning sample identity.
        gt_label: Ground-truth class, when known.
        clean_correct: Whether the clean prediction matched ground truth.
        vulnerability_score: Sample-level score from the run's primary metric.
        reliability_grade: Overall grade for this sample's most notable
            evidence; ``None`` when no analysis run exists.
        heatmap_asset_ref: Relative path (from the report root) to R3's
            rendered heatmap PNG; ``None`` if it could not be produced.
        thumbnail_asset_ref: Relative path to the R3 thumbnail; ``None`` if
            unavailable (design §5 단계5, e.g. no source_provenance).
        top_regions: This sample's most-affected regions, most severe first.
        task_extra: Task-specific fields from ``TaskPresentationAdapter.
            sample_extra_fields`` (design §R5); empty for classification.
    """

    sample_id: str
    gt_label: int | None
    clean_correct: bool | None
    vulnerability_score: float | None
    reliability_grade: ReportGrade | None
    heatmap_asset_ref: str | None
    thumbnail_asset_ref: str | None
    top_regions: tuple[TopRegionEntry, ...]
    task_extra: Mapping[str, "ReportJsonValue"]

    def __post_init__(self) -> None:
        """Validate identity, grade type, and nested collections.

        Raises:
            TypeError: If reliability_grade, a top_regions element, or a
                task_extra value has the wrong/unsupported type.
            ValueError: If sample_id is empty or a task_extra key is not a
                string.
        """

        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.reliability_grade is not None and not isinstance(
            self.reliability_grade, ReportGrade
        ):
            raise TypeError("reliability_grade must be a ReportGrade or None")
        if not all(isinstance(entry, TopRegionEntry) for entry in self.top_regions):
            raise TypeError("top_regions elements must be TopRegionEntry")
        for key, value in self.task_extra.items():
            if not isinstance(key, str):
                raise TypeError("task_extra keys must be strings")
            _validate_json_value(value, path=f"task_extra.{key}")


@dataclass(frozen=True, slots=True)
class RegionRow:
    """One region_key's dataset-wide summary (design §3, extended by §1 격차#3).

    Attributes:
        region_key: Stable concrete-region identity.
        region_kind: Mask materialization strategy, e.g. ``"grid"`` — a
            plain string (see module docstring on why this is not
            ``ssat.core.types.RegionKind``).
        intended_area_px: Selected pixels before model preprocessing.
        effective_area_px: Selected pixels in model input space, when known.
        mean_degradation: Mean degradation across valid items for this region.
        flip_rate: Fraction of valid items that flipped, when the primary
            metric is binary; ``None`` for continuous metrics.
        n_valid: Number of items with an available, unexcluded value.
        reliability_grade: Worst-case grade across every anchor sharing this
            region_key (IMPLE_PLAN_REPORTING_v1.md §1 격차#3's worst-case
            policy); ``None`` when no analysis run exists.
        reliability_distribution: Count of anchors sharing this region_key
            per grade value, e.g. ``{"high": 1, "unreliable": 1}`` — added
            precisely so the worst-case ``reliability_grade`` above does not
            read as more uniformly alarming than it is (§1 격차#3). Empty
            when no analysis run exists.
    """

    region_key: str
    region_kind: str
    intended_area_px: int | None
    effective_area_px: int | None
    mean_degradation: float | None
    flip_rate: float | None
    n_valid: int
    reliability_grade: ReportGrade | None
    reliability_distribution: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate identity, areas, counts, grade, and distribution.

        Raises:
            TypeError: If reliability_grade has the wrong type.
            ValueError: If region_key/region_kind is empty, an area or
                n_valid is negative, flip_rate is outside [0, 1], or
                reliability_distribution has an unknown key or negative count.
        """

        if not self.region_key or not self.region_kind:
            raise ValueError("region_key and region_kind must not be empty")
        for name, value in (
            ("intended_area_px", self.intended_area_px),
            ("effective_area_px", self.effective_area_px),
        ):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be non-negative when provided")
        _validate_unit_interval_or_none(self.flip_rate, field_name="flip_rate")
        if isinstance(self.n_valid, bool) or self.n_valid < 0:
            raise ValueError("n_valid must be non-negative")
        if self.reliability_grade is not None and not isinstance(
            self.reliability_grade, ReportGrade
        ):
            raise TypeError("reliability_grade must be a ReportGrade or None")
        _validate_grade_distribution(self.reliability_distribution)


@dataclass(frozen=True, slots=True)
class FlaggedItem:
    """One unreliable-grade anchor surfaced in the spotlight (design §3, §4.2 step 6).

    Attributes:
        anchor_key_repr: Flattened ``AnchorKey``, formatted as
            ``f"{sample_id}::{region_key}::{invert_mask}"`` (module
            docstring — kept as a plain string so this module never imports
            ``ssat.analysis.types.AnchorKey``).
        reason_summary: One-line human-readable summary of why this item
            was flagged.
        reliability_reasons: Every underlying justification, carried
            verbatim from the analysis module's ``ReliabilityRow``.
    """

    anchor_key_repr: str
    reason_summary: str
    reliability_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate identity fields.

        Raises:
            ValueError: If anchor_key_repr or reason_summary is empty.
        """

        if not self.anchor_key_repr or not self.reason_summary:
            raise ValueError("anchor_key_repr and reason_summary must not be empty")


# --- Section types -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Header metadata for the report (design §3, §4.2 step 1).

    Attributes:
        dataset_name: Derived from source provenance (IMPLE_PLAN_REPORTING_v1
            .md §1 "부수 확인" — not a field any upstream schema carries).
        n_samples: Number of samples audited.
        n_regions_per_sample: Number of distinct regions per sample, when
            uniform across the dataset.
        n_conditions: Number of distinct perturbation conditions exercised.
        duration_seconds: Wall-clock run duration, when the source run
            manifest recorded a finish time.
        failure_rate: Fraction of perturbed items excluded from every metric.
        model_id: Identity of the audited model.
        preprocessing_desc: Free-form description of the model's
            preprocessing, for provenance display.
    """

    dataset_name: str
    n_samples: int
    n_regions_per_sample: int
    n_conditions: int
    duration_seconds: float | None
    failure_rate: float | None
    model_id: str
    preprocessing_desc: str

    def __post_init__(self) -> None:
        """Validate identity fields, counts, and rate ranges.

        Raises:
            ValueError: If dataset_name/model_id is empty, a count is
                negative, duration_seconds is negative, or failure_rate is
                outside [0, 1].
        """

        if not self.dataset_name or not self.model_id:
            raise ValueError("dataset_name and model_id must not be empty")
        for name, value in (
            ("n_samples", self.n_samples),
            ("n_regions_per_sample", self.n_regions_per_sample),
            ("n_conditions", self.n_conditions),
        ):
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.duration_seconds is not None and self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative when provided")
        _validate_unit_interval_or_none(self.failure_rate, field_name="failure_rate")


@dataclass(frozen=True, slots=True)
class VulnerabilitySummaryStats:
    """Distributional summary of the full vulnerability_score population (design §3).

    Attributes:
        mean: Arithmetic mean of vulnerability_score across every sample;
            ``None`` when no sample has a computed vulnerability_score (e.g.
            every sample had zero valid primary-metric items) — distinct
            from a real zero, per the same "unavailable ≠ false" principle
            ``MetricCard.value`` follows.
        median: 50th percentile, or ``None`` under the same condition.
        p90: 90th percentile, or ``None`` under the same condition.
        p99: 99th percentile, or ``None`` under the same condition.
    """

    mean: float | None
    median: float | None
    p90: float | None
    p99: float | None

    def __post_init__(self) -> None:
        """Validate that the four fields are all present or all absent together.

        Raises:
            ValueError: If some but not all fields are ``None`` — they are
                always computed from the same score array in one pass, so a
                partial result would indicate a caller bug.
        """

        values = (self.mean, self.median, self.p90, self.p99)
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError("mean, median, p90, and p99 must be all present or all None")


@dataclass(frozen=True, slots=True)
class VulnerabilityDistribution:
    """The "평균 뒤에 숨은 변이" section (design §4.2 step 3, §R2).

    Attributes:
        histogram_asset_ref: Relative path to R2's rendered SVG histogram;
            ``None`` if not yet rendered.
        summary_stats: Distributional summary backing the histogram.
    """

    histogram_asset_ref: str | None
    summary_stats: VulnerabilitySummaryStats

    def __post_init__(self) -> None:
        """Validate the nested summary_stats type.

        Raises:
            TypeError: If summary_stats is not a VulnerabilitySummaryStats.
        """

        if not isinstance(self.summary_stats, VulnerabilitySummaryStats):
            raise TypeError("summary_stats must be a VulnerabilitySummaryStats")


@dataclass(frozen=True, slots=True)
class SampleRankings:
    """Top-K/bottom-K sample galleries (design §3, §4.2 step 4).

    Attributes:
        most_vulnerable: Top-K by vulnerability_score, descending.
        most_robust: Bottom-K by vulnerability_score, ascending severity.
    """

    most_vulnerable: tuple[SampleCard, ...]
    most_robust: tuple[SampleCard, ...]

    def __post_init__(self) -> None:
        """Validate element types.

        Raises:
            TypeError: If any element of either tuple is not a SampleCard.
        """

        if not all(isinstance(card, SampleCard) for card in self.most_vulnerable):
            raise TypeError("most_vulnerable elements must be SampleCard")
        if not all(isinstance(card, SampleCard) for card in self.most_robust):
            raise TypeError("most_robust elements must be SampleCard")


@dataclass(frozen=True, slots=True)
class RegionSummary:
    """Dataset-wide region table (design §3, §4.2 step 5).

    Attributes:
        rows: One RegionRow per region_key present in the dataset.
        reliability_distribution: Count of anchors across the *whole*
            dataset per grade value — the dataset-level counterpart of each
            row's own ``RegionRow.reliability_distribution``. Empty when no
            analysis run exists.
    """

    rows: tuple[RegionRow, ...]
    reliability_distribution: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate row types and the distribution mapping.

        Raises:
            TypeError: If any row is not a RegionRow.
            ValueError: If reliability_distribution has an unknown key or a
                negative count.
        """

        if not all(isinstance(row, RegionRow) for row in self.rows):
            raise TypeError("rows elements must be RegionRow")
        _validate_grade_distribution(self.reliability_distribution)


@dataclass(frozen=True, slots=True)
class ReliabilitySpotlight:
    """"이 결과는 믿지 말라" section (design §3, §4.2 step 6).

    Attributes:
        flagged_examples: Every UNRELIABLE-grade anchor, with reasons.
    """

    flagged_examples: tuple[FlaggedItem, ...]

    def __post_init__(self) -> None:
        """Validate element types.

        Raises:
            TypeError: If any element is not a FlaggedItem.
        """

        if not all(isinstance(item, FlaggedItem) for item in self.flagged_examples):
            raise TypeError("flagged_examples elements must be FlaggedItem")


@dataclass(frozen=True, slots=True)
class ReportSchemaVersions:
    """On-disk schema versions this report was assembled from (design §3).

    Attributes:
        dump: The source run's core schema_version.
        metrics: The source MetricsStore's schema_version.
        analysis: The source AnalysisStore's schema_version, when an
            analysis run was provided.
        report: This module's own REPORT_SCHEMA_VERSION at assembly time.
    """

    dump: str
    metrics: str
    analysis: str | None
    report: str

    def __post_init__(self) -> None:
        """Validate that every provided version string is non-empty.

        Raises:
            ValueError: If dump/metrics/report is empty, or analysis is
                present but empty.
        """

        if not self.dump or not self.metrics or not self.report:
            raise ValueError("dump, metrics, and report versions must not be empty")
        if self.analysis is not None and not self.analysis:
            raise ValueError("analysis version must not be empty when provided")


@dataclass(frozen=True, slots=True)
class ReportMeta:
    """Top-level provenance/identity block (design §3).

    Attributes:
        run_id: Identity of the audited run (typically the dump directory name).
        generated_at: ISO-8601 timestamp this report was assembled, as a
            pre-formatted string (matching how ``AnalyzeResult.computed_at``
            and ``ComputeMetricsResult.computed_at`` already carry
            timestamps through the application layer).
        tool_version: SSAT code version that assembled this report.
        schema_versions: Versions of every source this report was built from.
        task_kind: Which prediction task this report describes.
    """

    run_id: str
    generated_at: str
    tool_version: str
    schema_versions: ReportSchemaVersions
    task_kind: TaskKind

    def __post_init__(self) -> None:
        """Validate identity fields and nested types.

        Raises:
            TypeError: If schema_versions/task_kind has the wrong type.
            ValueError: If run_id, generated_at, or tool_version is empty.
        """

        if not self.run_id or not self.generated_at or not self.tool_version:
            raise ValueError("run_id, generated_at, and tool_version must not be empty")
        if not isinstance(self.schema_versions, ReportSchemaVersions):
            raise TypeError("schema_versions must be a ReportSchemaVersions")
        if not isinstance(self.task_kind, TaskKind):
            raise TypeError("task_kind must be a TaskKind")


@dataclass(frozen=True, slots=True)
class ProvenanceInfo:
    """Collapsible reproducibility appendix (design §3, §4.2 step 7).

    Attributes:
        dump_path: Absolute path to the source dump.
        metrics_dir: Absolute path to the source MetricsStore.
        analysis_dir: Absolute path to the source AnalysisStore, when one
            was provided.
        metrics_manifest_hash: SHA-256 of the source metrics_manifest.json.
        analysis_manifest_hash: SHA-256 of the source analysis_manifest.json,
            when an analysis run was provided.
        thresholds: Every threshold used to derive grades/flags upstream
            (A2-A6), carried through verbatim for reproducibility.
    """

    dump_path: str
    metrics_dir: str
    analysis_dir: str | None
    metrics_manifest_hash: str
    analysis_manifest_hash: str | None
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate identity fields.

        Raises:
            ValueError: If dump_path, metrics_dir, or metrics_manifest_hash
                is empty.
        """

        if not self.dump_path or not self.metrics_dir or not self.metrics_manifest_hash:
            raise ValueError(
                "dump_path, metrics_dir, and metrics_manifest_hash must not be empty"
            )


@dataclass(frozen=True, slots=True)
class ReportModel:
    """The single JSON-serializable model every report view is built from (design §3).

    Attributes:
        meta: Top-level identity/provenance block.
        run_summary: Header metadata.
        scorecard: Task-agnostic metric cards (design §R5).
        vulnerability_distribution: Dataset-wide score distribution.
        sample_rankings: Top-K/bottom-K sample galleries.
        region_summary: Dataset-wide region table.
        reliability_spotlight: Unreliable-grade anchors.
        provenance: Collapsible reproducibility appendix.
    """

    meta: ReportMeta
    run_summary: RunSummary
    scorecard: tuple[MetricCard, ...]
    vulnerability_distribution: VulnerabilityDistribution
    sample_rankings: SampleRankings
    region_summary: RegionSummary
    reliability_spotlight: ReliabilitySpotlight
    provenance: ProvenanceInfo

    def __post_init__(self) -> None:
        """Validate every section's type.

        Raises:
            TypeError: If any section has the wrong type, or a scorecard
                element is not a MetricCard.
        """

        if not isinstance(self.meta, ReportMeta):
            raise TypeError("meta must be a ReportMeta")
        if not isinstance(self.run_summary, RunSummary):
            raise TypeError("run_summary must be a RunSummary")
        if not all(isinstance(card, MetricCard) for card in self.scorecard):
            raise TypeError("scorecard elements must be MetricCard")
        if not isinstance(self.vulnerability_distribution, VulnerabilityDistribution):
            raise TypeError("vulnerability_distribution must be a VulnerabilityDistribution")
        if not isinstance(self.sample_rankings, SampleRankings):
            raise TypeError("sample_rankings must be a SampleRankings")
        if not isinstance(self.region_summary, RegionSummary):
            raise TypeError("region_summary must be a RegionSummary")
        if not isinstance(self.reliability_spotlight, ReliabilitySpotlight):
            raise TypeError("reliability_spotlight must be a ReliabilitySpotlight")
        if not isinstance(self.provenance, ProvenanceInfo):
            raise TypeError("provenance must be a ProvenanceInfo")

    @classmethod
    def from_dict(cls, data: Mapping[str, "ReportJsonValue"]) -> "ReportModel":
        """Reconstruct a ``ReportModel`` from its JSON-decoded dict form.

        This is the inverse of ``dataclasses.asdict(model)`` (plus a JSON
        round trip) — the shape R1's ``report_model.json`` is written in.
        It exists so that contract is a real two-way one, not export-only:
        the same "JSON model first" principle that lets a future WebUI serve
        a ``ReportModel`` as an HTTP response body (module docstring)
        implies that body must also be receivable and reconstructable into
        typed objects, not left as loosely-typed dicts. Every nested
        dataclass, tuple, and ``Enum`` this module defines is walked back
        into its typed form using the field's own type hint (see
        :func:`_coerce_from_json`); ``Mapping``-typed fields (e.g.
        ``reliability_distribution``, ``task_extra``) are passed through as
        plain dicts, since none of them nest further dataclasses.

        Args:
            data: A JSON-decoded ``ReportModel`` document, e.g. the result
                of ``json.loads(Path("report_model.json").read_text())``.

        Returns:
            An equivalent, fully typed ``ReportModel``.

        Raises:
            KeyError: If a required field is missing from ``data``.
            TypeError: If a value's shape does not match its field's type
                hint (raised by the reconstructed dataclass's own
                ``__post_init__``).
            ValueError: If a reconstructed value fails the corresponding
                dataclass's own validation.
        """

        return _dataclass_from_dict(cls, data)


# --- Shared private validators -------------------------------------------


def _validate_unit_interval_or_none(value: float | None, *, field_name: str) -> None:
    """Validate that an optional fraction lies within [0, 1].

    Raises:
        ValueError: If value is provided and outside the unit interval.
    """

    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _validate_grade_distribution(distribution: Mapping[str, int]) -> None:
    """Validate a grade-name-keyed count mapping.

    Raises:
        ValueError: If a key is not a recognized ReportGrade value, or a
            count is negative.
    """

    for key, count in distribution.items():
        if key not in _REPORT_GRADE_VALUES:
            raise ValueError(f"unknown reliability grade key: {key}")
        if isinstance(count, bool) or count < 0:
            raise ValueError(f"reliability_distribution[{key}] must be non-negative")


def _validate_json_value(value: "ReportJsonValue", *, path: str) -> None:
    """Validate a value against the same JSON-compatible contract as ssat.core.types.

    Duplicated rather than imported (module docstring) — this keeps
    ``report.types`` dependency-free.

    Raises:
        TypeError: If value is not JSON-compatible.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


# --- ReportModel.from_dict support -------------------------------------------


def _dataclass_from_dict(cls: type, data: Mapping[str, "ReportJsonValue"]) -> Any:
    """Reconstruct one dataclass instance from its ``dataclasses.asdict`` shape.

    Every field is rebuilt via :func:`_coerce_from_json`, guided by ``cls``'s
    own resolved type hints (``get_type_hints`` rather than
    ``dataclasses.fields(cls)[i].type``, since ``from __future__ import
    annotations`` leaves the latter as unevaluated strings).
    """

    hints = get_type_hints(cls)
    kwargs = {
        field.name: _coerce_from_json(hints[field.name], data[field.name])
        for field in dataclasses.fields(cls)
    }
    return cls(**kwargs)


def _coerce_from_json(annotation: Any, value: Any) -> Any:
    """Rebuild one field's value from its JSON-decoded form, guided by its type hint.

    Walks back, in reverse, the three shapes ``dataclasses.asdict`` + a JSON
    round trip produce for values in this module: a nested dataclass became
    a dict (recurse via :func:`_dataclass_from_dict`), a tuple became a list
    (rebuild elementwise using the tuple's element type), and an ``Enum``
    member became its plain string value (reconstruct via ``EnumType(value)``
    since every ``Enum`` here mixes in ``str``, module docstring).
    ``Mapping``-typed fields are returned as-is — none of them nest further
    dataclasses in this schema, so a plain dict already has the right shape.
    """

    if value is None:
        return None

    origin = get_origin(annotation)
    if origin is UnionType:
        non_none_types = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _coerce_from_json(non_none_types[0], value)
    if origin is tuple:
        element_type, _ellipsis = get_args(annotation)
        return tuple(_coerce_from_json(element_type, item) for item in value)
    if origin is Mapping or origin is dict:
        return dict(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if dataclasses.is_dataclass(annotation):
        return _dataclass_from_dict(annotation, value)
    return value
