# Captum reference workflow

This directory contains an independent, dataset-scale implementation of the
crop-free 4×4 synthetic-shortcut audit. It shares only manifests,
checkpoints, dataset statistics, and pre-registered numerical choices with
the main experiment. Its Python modules do not import the SSAT package or
the existing synthetic-shortcut scripts.

Captum `FeatureAblation` supplies the low-level operation: replace one feature
group with a baseline and measure the output change. This reference code
implements the surrounding workflow itself: data iteration, five
interventions, controls, repeated seeds, raw storage, multi-level
aggregation, bootstrap intervals, operator agreement, area validation,
cache/resume, provenance, and reports.

## Environment

Run every command inside the Docker Compose workspace:

```bash
docker compose restart region-sensitivity-workspace
docker compose exec region-sensitivity-workspace bash -lc \
  'cd /workspace && nvidia-smi && python -c "import torch; assert torch.cuda.is_available()"'
docker compose exec region-sensitivity-workspace bash -lc \
  'cd /workspace && pip install -e ".[reference]"'
```

The fixed full experiment requires CUDA and refuses to fall back to CPU.

## Run, interrupt, and resume

The following first writes 10,000 raw item rows and exits successfully with
an `incomplete` status. The second invocation verifies the output identity
and resumes only missing item identities.

```bash
cd /workspace
python experiments/reference_comparison/captum_baseline/run.py audit \
  --config experiments/reference_comparison/captum_baseline/config.yaml \
  --output experiments/reference_comparison/captum_baseline/results_run_a \
  --stop-after-items 10000
python experiments/reference_comparison/captum_baseline/run.py all \
  --config experiments/reference_comparison/captum_baseline/config.yaml \
  --output experiments/reference_comparison/captum_baseline/results_run_a
```

Run `all` again against the completed directory to check the cache. The
printed audit result must report `new_rows: 0` and
`forward_evaluations: 0`.

For the independent reproducibility run:

```bash
python experiments/reference_comparison/captum_baseline/run.py all \
  --config experiments/reference_comparison/captum_baseline/config.yaml \
  --output experiments/reference_comparison/captum_baseline/results_run_b
```

Compare `analysis/summary.json` in the two outputs. `canonical_hash` excludes
timestamps and paths and must match.

## Output contract

- `raw/part-*.parquet`: append-only item observations. The item identity is
  model × dataset × sample × target/control region × operator × seed.
- `run_manifest.json`: resolved configuration, input hashes, row count, and
  incomplete/complete state.
- `accuracy.json`: Q5 top-1 accuracy inputs.
- `provenance.json`: git and Python/PyTorch/torchvision/Captum/CUDA identity.
- `analysis/`: sample, region, class, dataset, control, seed, bootstrap, and
  operator-consistency tables in Parquet and CSV.
- `report.md`, `comparison_metrics.json`, `capability_comparison.csv`: final
  Q1–Q5 and workflow comparison.

Large result directories are intentionally ignored by the repository.

