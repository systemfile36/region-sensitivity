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
