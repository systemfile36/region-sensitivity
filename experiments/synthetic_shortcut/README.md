# experiments/synthetic_shortcut

Implements the L3 synthetic-shortcut experiment
(`docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md`): a squeezenet1_0 classifier
trained on CIFAR-10 with a synthetic patch shortcut baked into one class
(`M_shortcut`) is audited region-by-region and compared against a clean
control model (`M_normal`) to check whether region-sensitivity auditing
actually finds the shortcut. Current results are reported in
`docs/L3_Synthetic-Shortcut Experiment Report.md` (crop-free preprocessing)
and `docs/RELIABILITY_THRESHOLD_CALIBRATION_v1.md` (reliability-threshold
recalibration).

**All commands below run inside the `region-sensitivity-workspace` Docker
Compose container** (`docker compose exec region-sensitivity-workspace
bash -lc '...'`), never on the host.

## Script inventory

| Script | Status | Role |
|---|---|---|
| `common.py` | canonical | Shared grid/patch/palette constants and crop-free preprocessing helpers (`build_crop_free_transform`). No CLI. |
| `prepare_data.py` | canonical | Downloads CIFAR-10, builds the A/B/C manifests. Preprocessing-agnostic; run once. |
| `train.py` | canonical | Trains `M_shortcut` (`--dataset shortcut`) or `M_normal` (`--dataset normal`). |
| `run_audit.py` | canonical | Runs the 7 pre-registered (model, dataset, fill) audit combinations. |
| `evaluate_accuracy.py` | canonical | Q5's plain top-1 accuracy numbers. |
| `evaluate.py` | canonical | Judges Q1-Q5 and section 3.5, writes `report.md`. |
| `analyze_section35_sensitivity.py` | canonical | Follow-up diagnostic on the fill-strategy rank-correlation anomaly. |
| `analyze_sign_group_premise.py` | canonical | Compares the fill-strategy sign-group split between the cropped and crop-free runs (Sign-Group Premise Re-examination). |
| `run_threshold_validation_full.py` | canonical | Crop-free, all-5-fill-strategy audit with controls + multi-seed, for reliability-threshold recalibration. |
| `validate_reliability_thresholds_full.py` | canonical | Reports on the above run against `ssat.analysis`'s `z_vs_control`/`seed_cv` defaults. |
| `verify_crop_free_parity.py` | diagnostic | One-off pre-flight check that train-time and audit-time crop-free preprocessing are bit-identical. No results-dir dependency; run once before crop-free training. |
| `analyze_control_stability.py` | diagnostic | Regression check: does `ssat.analysis` (A0-A6) reproduce the *original cropped run's* hand-derived numbers? Expected to report FAIL against `results_crop_free` by design -- see its own module docstring. |
| `run_threshold_validation.py` | **SUPERSEDED** | Provisional, single-op (`constant_fill`), cropped-preset predecessor of `run_threshold_validation_full.py`. Kept only for historical comparison. |
| `validate_reliability_thresholds.py` | **SUPERSEDED** | Predecessor of `validate_reliability_thresholds_full.py`, reports on the run above. Kept only for historical comparison. |

### Two footguns

- **`--preprocessing` defaults to `preset`** (the original CenterCrop
  pipeline) on `train.py`, `run_audit.py`, and `evaluate_accuracy.py`.
  Omitting `--preprocessing crop_free` silently reproduces the superseded,
  cropped evidence (`docs/deprecated_L3_Synthetic-Shortcut Experiment
  Report.md`), not the current report.
- **`evaluate_accuracy.py --results-dir` defaults to `results/`**, same as
  every other script -- but to reproduce the current (crop-free) report you
  must pass `--results-dir results_crop_free` explicitly, or its accuracy
  numbers will be read from/written to the wrong (cropped) tree.

## Quick Start

### A. Reproduce `docs/L3_Synthetic-Shortcut Experiment Report.md`

```bash
cd experiments/synthetic_shortcut

python3 prepare_data.py                    # one-time; skip if data/ is populated
python3 verify_crop_free_parity.py         # pre-flight sanity check, no args

python3 train.py --dataset shortcut --preprocessing crop_free --checkpoint-dir checkpoints_crop_free
python3 train.py --dataset normal   --preprocessing crop_free --checkpoint-dir checkpoints_crop_free

python3 run_audit.py --preprocessing crop_free --checkpoint-dir checkpoints_crop_free --results-dir results_crop_free
python3 evaluate_accuracy.py --preprocessing crop_free --checkpoint-dir checkpoints_crop_free --results-dir results_crop_free
python3 evaluate.py --results-dir results_crop_free
python3 analyze_section35_sensitivity.py --results-dir results_crop_free
python3 analyze_sign_group_premise.py --cropped-results-dir results --crop-free-results-dir results_crop_free
```

Or run `bash reproduce_l3_report.sh`, which chains the same steps.

### B. Reproduce `docs/RELIABILITY_THRESHOLD_CALIBRATION_v1.md`

Requires sequence A's `checkpoints_crop_free/` to already exist.

```bash
cd experiments/synthetic_shortcut

python3 run_threshold_validation_full.py --checkpoint-dir checkpoints_crop_free --results-dir results_crop_free
python3 validate_reliability_thresholds_full.py --results-dir results_crop_free
```

Or run `bash reproduce_threshold_calibration.sh`.

### A lighter-weight alternative for a single run

For a one-off control/stability analysis of a single already-audited dump
(rather than combining multiple `shortcut_A_*` runs the way
`analyze_control_stability.py` does), use the `ssat analyze` CLI command
added to the Application layer:

```bash
ssat analyze results_crop_free/dumps/<run_id> --metrics-dir results_crop_free/metrics/<run_id> --json
```

Combining several runs into one `item_values` frame (as the sign-group and
threshold-calibration scripts above do) is still a script-level concern,
not something `ssat analyze` does.
