#!/usr/bin/env bash
# Reproduce docs/RELIABILITY_THRESHOLD_CALIBRATION_v1.md. Requires
# checkpoints_crop_free/ to already exist (run reproduce_l3_report.sh, or at
# least its train.py steps, first). Run inside the
# region-sensitivity-workspace container.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. run_threshold_validation_full.py =="
python3 run_threshold_validation_full.py --checkpoint-dir checkpoints_crop_free --results-dir results_crop_free

echo "== 2. validate_reliability_thresholds_full.py =="
python3 validate_reliability_thresholds_full.py --results-dir results_crop_free

echo "== done: see results_crop_free/threshold_validation_report_full.md =="
