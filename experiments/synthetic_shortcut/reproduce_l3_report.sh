#!/usr/bin/env bash
# Reproduce the synthetic-shortcut Q1-Q5 experiment end to end (crop-free
# preprocessing). Run inside the region-sensitivity-workspace container.
# See README.md for the equivalent step-by-step commands.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. prepare_data.py =="
python3 prepare_data.py

echo "== 2. verify_crop_free_parity.py =="
python3 verify_crop_free_parity.py

echo "== 3. train.py --dataset shortcut =="
python3 train.py --dataset shortcut --preprocessing crop_free --checkpoint-dir checkpoints_crop_free

echo "== 4. train.py --dataset normal =="
python3 train.py --dataset normal --preprocessing crop_free --checkpoint-dir checkpoints_crop_free

echo "== 5. run_audit.py =="
python3 run_audit.py --preprocessing crop_free --checkpoint-dir checkpoints_crop_free --results-dir results_crop_free

echo "== 6. evaluate_accuracy.py =="
python3 evaluate_accuracy.py --preprocessing crop_free --checkpoint-dir checkpoints_crop_free --results-dir results_crop_free

echo "== 7. evaluate.py =="
python3 evaluate.py --results-dir results_crop_free

echo "== 8. analyze_section35_sensitivity.py =="
python3 analyze_section35_sensitivity.py --results-dir results_crop_free

echo "== 9. analyze_sign_group_premise.py =="
python3 analyze_sign_group_premise.py --cropped-results-dir results --crop-free-results-dir results_crop_free

echo "== done: see results_crop_free/report.md =="
