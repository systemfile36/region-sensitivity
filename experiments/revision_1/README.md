# experiments/revision_1

Additional experiments for the SSAT minor revision (reviewer requests on
reliability-threshold sensitivity, matched-control count, architecture
coverage, computational scaling, ImageNet preprocessing confound, NTU
semantic validation, and the Captum comparison). Code for all of them lives
here; the `ssat/` package keeps its v1.0.0 defaults and behavior throughout
the revision.

**All commands run inside the `region-sensitivity-workspace` Docker Compose
container** (`docker compose exec region-sensitivity-workspace bash -lc '...'`),
never on the host. Scripts are run from the repository root.

## Layout

| Path | Tracked | Content |
|---|---|---|
| `common/` | yes | Baseline run registry, column-selective loaders, agreement statistics, deterministic subsets, provenance, matrix runner |
| `phase0/` | yes | Baseline freeze and verification scripts (P0-1 to P0-7) |
| `phase0/summary/` | yes | Small JSON outputs of Phase 0, each with a `*.provenance.json` |
| `deviations.md` | yes | Every departure from the pre-registered plan, with timing |
| `<experiment>/protocol.json` | yes | Pre-registered settings, committed before the experiment runs |
| `results/` | no (`results*/`) | Dumps, recomputed stores, logs, and other large intermediates |

## Baseline runs

`common/runs.py` is the single source of truth for which stored outputs the
revision reads (`BASELINE_RUNS`):

| Run | Stores read | K (controls/target) |
|---|---|---|
| `imagenet_mnv2_{050,100}_{exact,crop_free}_k3` | `experiments/real_dataset_case_study/results_control_3/<run>/` | 3 |
| `ntu60_tsm_{exact,crop_free}` | dump in `experiments/real_dataset_case_study/results/<run>/`; metrics/analysis recomputed by P0-7 | 0 |
| `synthetic_{shortcut,normal}` | dump in `experiments/synthetic_shortcut/results_crop_free/dumps/`; metrics/analysis recomputed by P0-7 | 2 |

`K1_PARITY_RUNS` (`experiments/real_dataset_case_study/results/imagenet_*`,
K=1) are read only because the paper's accuracy / mean `margin_drop` table
came from them.

## Phase 0: baseline freeze

| ID | Command | Output (`phase0/summary/`) | Time |
|---|---|---|---|
| P0-1 | `python experiments/revision_1/phase0/build_baseline_manifest.py` | `baseline_manifest.json` | ~2 min |
| P0-2 | `python experiments/revision_1/phase0/verify_paper_numbers.py` | `paper_numbers.json` | ~1 min |
| P0-3 | `python experiments/revision_1/phase0/verify_target_parity.py` | `target_parity.json` | ~13 min, ~15 GB RAM |
| P0-4 | (doc edit) | `docs/BENCHMARK_v1.md` notes the benchmark ran with K=1 | – |
| P0-5 | `SSAT_CONTAINER_IMAGE_ID=<id> python experiments/revision_1/phase0/capture_environment.py --note ...` | `environment.json` | seconds |
| P0-6 | `python -m pytest -q tests/unit/test_rev1_common.py` | – | seconds |
| P0-7 | `python experiments/revision_1/phase0/recompute_small_baselines.py` | `recomputed_baselines.json` (+ stores in `results/phase0/recomputed/`) | ~3 min |

P0-7 must run before any experiment that reads the NTU or synthetic
baselines. Findings are summarized in `phase0/PHASE0_REPORT.md`.

To pass the image id from the host:

```bash
docker compose exec -e SSAT_CONTAINER_IMAGE_ID="$(docker inspect --format '{{.Id}}' local/region-sensitivity-workspace:latest)" \
  region-sensitivity-workspace bash -lc 'python experiments/revision_1/phase0/capture_environment.py'
```

## Phase 1A: A1 reliability threshold sensitivity (offline)

Pre-registered in `a1_threshold/protocol.json`; results in
`a1_threshold/A1_REPORT.md`.

| Step | Command | Output | Time |
|---|---|---|---|
| Features | `python experiments/revision_1/a1_threshold/build_features.py --runs all` | `results/a1/features/<run>__margin_drop.{parquet,json}` | ~1 min |
| Parity gate + sweep | `python experiments/revision_1/a1_threshold/run_sweep.py` | `a1_threshold/summary/*.csv`, `parity.json` | ~15 s |
| Tables and figures | `python experiments/revision_1/a1_threshold/summarize.py` | `sensitivity_ranking.csv`, `table_a1.*`, `fig_a1_*.pdf` | seconds |
| Tests | `python -m pytest -q tests/unit/test_rev1_grade_engine.py` | – | seconds |

`build_features.py` fails unless the recomputed 95 % intervals reproduce
`intervals.parquet`, and `run_sweep.py` stops unless the default-setting grades
reproduce `reliability.parquet` exactly.

## Phase 1B: B1 ImageNet preprocessing confound (offline)

Pre-registered in `b1_preprocessing/protocol.json`; results in
`b1_preprocessing/B1_REPORT.md`. B1-4 (effective-area-matched controls) runs
after the A2 K=20 dumps exist.

| Step | Command | Output (`b1_preprocessing/summary/`) | Time |
|---|---|---|---|
| B1-1 areas | `python experiments/revision_1/b1_preprocessing/area_tables.py` | `cell_area.csv`, `within_sample_area_ratio.csv`, `control_area_ratio.csv`, `zero_area_targets.csv` | ~1 min |
| B1-2/3 ranking and grades | `python experiments/revision_1/b1_preprocessing/compare_protocols.py` | `ranking_change.csv`, `cell_rank_change.csv`, `top_region_share.csv`, `grade_*.csv`, `by_cell_type.csv`, `yardstick.csv`, `representative_cases.csv` | ~10 s |
| Figures | `python experiments/revision_1/b1_preprocessing/plot_figures.py --image-root <dir with the 3 example JPEGs>` | `fig_b1_area.pdf`, `fig_b1_rank.pdf`; `results/b1/fig_b1_examples.pdf` (untracked, embeds ImageNet images) | ~1 min |
| Tests | `python -m pytest -q tests/unit/test_rev1_b1.py` | – | seconds |

`area_tables.py` fails unless its anchor-level area ratios reproduce the stored
`area_matched` flag exactly. The ImageNet JPEGs are currently missing from
`data/imagenet/` (`deviations.md` D-004); the example images were extracted
from the ImageNet zip and are checked against the dumps' `content_hash`.

## Running new audits

`common/run_matrix.py --matrix <matrix.json> --output-root <dir>` runs
`ssat run -> metrics -> analyze -> report` for each entry and appends per-step
wall times to `<dir>/run_matrix_log.jsonl`. Do not run two `analyze` steps
at the same time (a 3.2 M-item analysis peaks around 60 GiB RSS).

## Reviewer comment -> artifact

Filled in as experiments complete.

| Reviewer request | Artifact | Location |
|---|---|---|
| Config / seed / environment / commit for every experiment | `baseline_manifest.json`, `environment.json`, each `*.provenance.json` | `phase0/summary/` |
| Threshold sensitivity figure/table | `fig_a1_z_curve.pdf`, `fig_a1_oat.pdf`, `table_a1.*`, `sensitivity_ranking.csv` | `a1_threshold/summary/` |
| Reliability grade transition summary | `transitions.csv`, `fig_a1_transitions.pdf`, `agreement.csv` | `a1_threshold/summary/` |
| Preprocessing: nominal vs effective area | `cell_area.csv`, `within_sample_area_ratio.csv`, `control_area_ratio.csv`, `fig_b1_area.pdf` | `b1_preprocessing/summary/` |
| Preprocessing: ranking and grade change | `ranking_change.csv`, `grade_transition_exact_to_cf.csv`, `yardstick.csv`, `fig_b1_rank.pdf` | `b1_preprocessing/summary/` |
