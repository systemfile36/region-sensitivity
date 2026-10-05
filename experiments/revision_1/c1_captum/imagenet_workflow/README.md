# Captum reference workflow, ImageNet (C1-std)

The second setting of the Captum engineering comparison (implementation plan
section 9.3). It was made by copying the synthetic-shortcut reference
(`experiments/reference_comparison/captum_baseline/`) and changing the parts
that were specific to that setting:

- **Data:** an ImageNet `<relative_path> <label>` file list instead of JSON
  manifests. Images keep their source size, so one sample is ablated at a
  time; perturbation candidates are built in the loader workers.
- **Model and preprocessing:** a timm classifier
  (`mobilenetv2_050.lamb_in1k`) with crop-free squash to 224x224. The resize
  (bicubic, antialiased) runs on the GPU and is rounded to integer pixel
  values. The config's preprocessing is checked against timm's data config.
- **Operators:** mean fill, blur, and Gaussian noise (3 seeds), with the
  parameters of the ImageNet case study.
- **Controls:** 3 per target cell, rigid translations to deterministic
  random positions, as in the synthetic reference. The 48 controls of one
  sample are batched in one `FeatureAblation` call with a per-example
  two-group mask (control = 0, rest = 1); Captum ablates both groups, so
  half of the control forwards are not used. This is kept as it was in the
  synthetic reference.
- **Questions:** the synthetic Q1-Q5 verdicts and their SSAT parity check
  are replaced by the region profile (per operator and pooled), the
  top-region share, and a control summary (share of target rows with z > 2).
  The comparison with SSAT is done outside this workflow
  (`../summarize_std.py`).

It shares with SSAT only the inputs and these fixed numerical choices: the
sample list, model, operator parameters, the channel mean (rounded to uint8),
integer grid boundaries (`row * height // rows`), the squash geometry, and
the margin definition. It imports neither SSAT nor the synthetic reference.
Matched controls are placed by this workflow's own seeds, so their positions
differ from SSAT's.

## Run, interrupt, and resume

Inside the Docker Compose workspace, from `/workspace`:

```bash
python experiments/revision_1/c1_captum/imagenet_workflow/run.py audit \
  --output <out> --stop-after-items 10000
python experiments/revision_1/c1_captum/imagenet_workflow/run.py all --output <out>
```

The second call resumes only the missing item identities. `run.py all`
against a completed output reports `new_rows: 0` and
`forward_evaluations: 0`.

## Output contract

- `raw/part-*.parquet`: one row per item (sample x target/control region x
  operator x seed) with clean and perturbed margin, degradation, source and
  model-space area.
- `run_manifest.json`, `execution_summary.json`, `accuracy.json`,
  `provenance.json`.
- `analysis/`: sample, region, class, dataset, region profile, top-region
  share, controls, control summary, seed stability, bootstrap intervals, and
  operator consistency tables (Parquet and CSV), and `summary.json` with
  canonical hashes.
- `report.md`.
