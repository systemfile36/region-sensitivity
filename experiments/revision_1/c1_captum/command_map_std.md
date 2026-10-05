# C1-std command map: Captum ImageNet workflow vs SSAT

Fixed before any C1-std measurement (implementation plan section 9.3,
`protocol_std.json`). Both workflows audit the same 1,000 ImageNet
validation images (`data/revision_1/imagenet/val_1_per_class_c1.txt`) with
`mobilenetv2_050.lamb_in1k`, crop-free squash to 224x224, a 4x4 grid,
mean fill / blur / Gaussian noise (3 seeds), and 3 matched controls per
cell. Each command runs in its own wrapper process (`measure_step`), from
`/workspace`, inside the container.

## Captum workflow (`experiments/revision_1/c1_captum/imagenet_workflow/run.py`)

| # | Command | Stage | What it does |
|---|---|---|---|
| C-1 | `run.py audit --config imagenet_workflow/config.yaml --output <out>` | audit | clean pass and 320,000 ablations (16 cells + 48 controls x 5 variants per sample), raw Parquet, accuracy, manifest, provenance |
| C-2 | `run.py analyze --config ... --output <out>` | analysis + report | sample/region/class/dataset tables, region profile, top-region share, controls, seed stability, bootstrap CI, operator consistency, area check |
| C-3 | `run.py report --config ... --output <out>` | analysis + report | `report.md` |

## SSAT workflow

| # | Command | Stage | Captum counterpart |
|---|---|---|---|
| S-1 | `python -m ssat run configs/imagenet_mnv2_050_crop_free_c1.yaml -o <out> --yes` | audit | C-1 (all model evaluations, raw logits) |
| S-2 | `python -m ssat metrics <out>` | analysis + report | degradations (Captum computes them during C-1) and sample/region aggregates |
| S-3 | `python -m ssat analyze <out>` | analysis + report | C-2: controls, seed and strategy stability, intervals, reliability grades |
| S-4 | `python -m ssat report <out>` | analysis + report | C-3: HTML report |

## Differences kept as they are

- **Item counts.** Both evaluate 320,000 perturbed items. SSAT also stores
  1,000 clean rows; the Captum workflow runs one clean pass per sample and
  repeats it inside each `FeatureAblation` call (Captum's initial
  evaluation), and its control calls also ablate the complement group,
  which is not used (`imagenet_workflow/README.md`). Model forward passes
  therefore differ from item counts; times are reported per perturbed item.
- **Stages.** SSAT computes degradations in `ssat metrics` (S-2); Captum
  computes them during the audit. The comparison uses two stages: audit
  (all model evaluations) and analysis + report.
- **Execution settings.** Captum: 12 loader workers, 4 samples per loader
  batch, 16 perturbations per evaluation for the cells, one sample's
  controls per call (chunked by a 12 M-pixel budget). SSAT: the case-study
  runtime section (12 workers, target batch 128, 128 variants per chunk).
  Neither is tuned for this comparison.
- **Evaluation path.** Captum builds full-frame candidates in the loader
  workers, composites regions and resizes on the GPU. SSAT perturbs each
  item and resizes it with PIL on the CPU, then writes all 1,000 logits per
  item.
- **Matched controls.** Same definition (rigid translation of the target
  cell to a uniformly random in-bounds position), different seeds: Captum
  places a sample's controls once per (cell, index) for all operators;
  SSAT places them per item. Control-based results are compared only
  informationally.
- **Stored content.** Captum stores one row per item (margins and
  degradation). SSAT stores full logits for 1,000 classes, item metadata,
  metrics, analysis tables, and an HTML report. Bytes are reported by
  content; no conclusion is drawn from total bytes alone.
- **Not in the Captum workflow:** SSAT's reliability grades and HTML report.
