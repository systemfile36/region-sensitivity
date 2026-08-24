"""``inspect``/``rebuild_index`` bodies for ``AuditApplication`` (facade delegation)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ssat.application.locking import OutputLockedError, output_lock
from ssat.application.types import (
    ApplicationError,
    ApplicationErrorCode,
    DumpSectionSummary,
    DumpSummary,
    IndexRebuildResult,
    InspectRequest,
    RebuildIndexRequest,
)
from ssat.core.dump import DumpReader
from ssat.core.resume import ResumeIndex
from ssat.core.types import ItemStatus

if TYPE_CHECKING:
    from ssat.application.application import AuditApplication


def inspect(request: InspectRequest) -> DumpSummary:
    """Summarize authoritative dump rows and manifest provenance."""

    try:
        dump = request.dump.expanduser().resolve(strict=True)
        reader = DumpReader(dump)
        manifest = reader.read_manifest()
        clean = reader.read_clean()
        perturbed = reader.read_perturbed()
        clean_observed = Counter(row["status"] for row in clean.to_pylist())
        perturbed_observed = Counter(row["status"] for row in perturbed.to_pylist())
        clean_counts = {
            status.value: clean_observed.get(status.value, 0) for status in ItemStatus
        }
        perturbed_counts = {
            status.value: perturbed_observed.get(status.value, 0)
            for status in ItemStatus
        }
        total_counts = {
            status.value: clean_counts[status.value] + perturbed_counts[status.value]
            for status in ItemStatus
        }
        manifest_counts = {
            status.value: count for status, count in manifest.counts_by_status.items()
        }
        return DumpSummary(
            dump=dump,
            schema_version=manifest.schema_version,
            code_version=manifest.code_version,
            model_id=manifest.adapter_spec.model_id,
            started_at=manifest.started_at.isoformat(),
            finished_at=(
                None if manifest.finished_at is None else manifest.finished_at.isoformat()
            ),
            resume_count=len(manifest.resume_events),
            clean=DumpSectionSummary(clean.num_rows, clean_counts),
            perturbed=DumpSectionSummary(
                perturbed.num_rows,
                perturbed_counts,
            ),
            total_counts_by_status=total_counts,
            manifest_counts_match=manifest_counts == total_counts,
        )
    except Exception as error:
        raise ApplicationError(
            ApplicationErrorCode.CORRUPTION,
            f"cannot inspect dump: {error}",
        ) from error


def rebuild_index(app: AuditApplication, request: RebuildIndexRequest) -> IndexRebuildResult:
    """Rebuild the perturbed index and return the resulting summary."""

    dump = request.dump.expanduser().resolve()
    try:
        with output_lock(dump):
            ResumeIndex.rebuild(dump)
        summary = app.inspect(InspectRequest(dump))
        return IndexRebuildResult(dump, summary.perturbed.rows, summary)
    except OutputLockedError as error:
        raise ApplicationError(ApplicationErrorCode.LOCKED, str(error)) from error
    except ApplicationError:
        raise
    except Exception as error:
        raise ApplicationError(
            ApplicationErrorCode.CORRUPTION,
            f"cannot rebuild dump index: {error}",
        ) from error
