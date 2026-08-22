#!/usr/bin/env bash
# Reproduce the synthetic-shortcut Q1-Q5 verdicts (crop-free preprocessing)
# WITHOUT training -- uses the shipped pretrained/ checkpoints. This is the
# fast demo path; see README.md's "Quick Start" section for how it differs
# from the full retrain path (reproduce_l3_report.sh). Run inside the
# region-sensitivity-workspace container.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. prepare_data.py (CIFAR-10 auto-download; no overrides, must match the shipped checkpoints' defaults) =="
python3 prepare_data.py

echo "== 2. verify_crop_free_parity.py =="
python3 verify_crop_free_parity.py

echo "== 3. run_audit.py (against shipped pretrained/ checkpoints, no training) =="
python3 run_audit.py --preprocessing crop_free --checkpoint-dir pretrained --results-dir results_crop_free_demo

echo "== 4. evaluate_accuracy.py =="
python3 evaluate_accuracy.py --preprocessing crop_free --checkpoint-dir pretrained --results-dir results_crop_free_demo

echo "== 5. evaluate.py =="
python3 evaluate.py --results-dir results_crop_free_demo

echo "== 6. check_demo_verdicts.py =="
python3 check_demo_verdicts.py --results-dir results_crop_free_demo

echo "== done: see results_crop_free_demo/report.md and results_crop_free_demo/verdicts.json =="
