# C1 command map: Captum reference workflow vs SSAT workflow

Fixed before any C1 measurement (implementation plan section 9.2). Both
workflows answer the same synthetic-shortcut questions (Q1-Q5 plus
controls, seed repeats, operator agreement, uncertainty, and a report) on the
same inputs:
- manifests `A_audit` / `B_audit` (200 samples each), `A_test` / `C_test`
  (2,000 each);
- checkpoints `checkpoints_crop_free/m_{shortcut,normal}.pt`;
- `dataset_stats.json`, 4x4 grid, and the five operators with the same
  parameters.

The SSAT side is the set of experiment scripts whose SLOC the existing
comparison counted (`captum_baseline/analysis.py`, `glue_paths`). It is not a
new, minimal SSAT configuration. Each command runs in its own wrapper
process (`measure_step`), from `/workspace`, inside the container.

## Captum reference workflow (`experiments/reference_comparison/captum_baseline/run.py`)

`run.py all` is split into its three subcommands so the audit and the
analysis can be measured separately; `all` calls the same three functions in
one process.

| # | Command | Stage | What it does |
|---|---|---|---|
| C-1 | `run.py audit --config captum_baseline/config.yaml --output <out>` | audit | 3 `audit_runs` (291,200 raw rows) and 2 `accuracy_runs`, raw Parquet, manifest, provenance |
| C-2 | `run.py analyze --config ... --output <out>` | analysis + report | sample/region/class/dataset tables, controls, seed stability, bootstrap CI, operator consistency, Q1-Q5, area sanity, SSAT parity |
| C-3 | `run.py report --config ... --output <out>` | analysis + report | `report.md`, `comparison_metrics.json`, `capability_comparison.csv` |

## SSAT workflow (`experiments/synthetic_shortcut/`)

| # | Command | Stage | Captum counterpart |
|---|---|---|---|
| S-1 | `run_audit.py --preprocessing crop_free --checkpoint-dir checkpoints_crop_free --results-dir <out>` | audit | Q1-Q4 region rankings: 7 single-operator audits (shortcut A x 5 operators, normal A constant fill, shortcut B constant fill; 3,400 items each) with margin-drop metrics |
| S-2 | `evaluate_accuracy.py --preprocessing crop_free --checkpoint-dir checkpoints_crop_free --results-dir <out>` | audit | `accuracy_runs` (Q5) |
| S-3 | `evaluate.py --results-dir <out>` | analysis + report | Q1-Q5 verdicts, region CSVs, heatmaps, `report.md` |
| S-4 | `run_threshold_validation_full.py --model shortcut --checkpoint-dir checkpoints_crop_free --results-dir <out>` | audit | `audit_runs[0]`: shortcut A, 5 operators x 3 seeds, 2 controls per region (144,200 items) with metrics |
| S-5 | `run_threshold_validation_full.py --model normal ...` | audit | `audit_runs[1]`: same for the normal model |
| S-6 | `generate_report.py --model shortcut --results-dir <out>` | analysis + report | controls, seed stability, bootstrap CI, operator consistency, reliability grades, HTML report |
| S-7 | `generate_report.py --model normal --results-dir <out>` | analysis + report | same for the normal model |

## Differences kept as they are

- **Item counts.** The SSAT workflow evaluates 310,400 perturbed items
  (7 x 3,200 + 2 x 144,000) against Captum's 291,200 raw rows. The
  difference, 19,200 items (6.6 %), is the six single-operator shortcut A
  and normal A audits in S-1. S-4 and S-5 repeat them with seeds and
  controls. At the time, `ssat` aggregated region metrics without a
  per-operator axis, so the Q1-Q4 rankings came from separate
  single-operator runs. Times are also reported per perturbed item.
- **Execution settings.** Each workflow uses its own settings:
  - Captum: batch 16, `perturbations_per_eval` 16, 4 workers;
  - SSAT: the `ssat` runtime defaults (batch 128, 12 workers).
  Neither is tuned for this comparison.
- **Stored content.** Captum stores one row per item with clean and
  perturbed margins and the degradation. SSAT stores full logits, region
  and operator metadata, and an item index for every item. It also stores
  metrics, analysis tables, an HTML report, and heatmaps. Byte counts are
  therefore reported per subdirectory together with their content, and no
  conclusion is drawn from total bytes alone (implementation plan section
  9.2).
- **Stages.** The workflows split their work differently: SSAT computes
  metrics inside S-1 / S-4 / S-5, and Captum computes degradations during
  the audit. The comparison therefore uses two stages: audit (all model
  evaluations) and analysis + report.
