# Captum Reference Workflow Comparison

## Purpose

This experiment measures the engineering needed to turn Captum's
feature-ablation primitive into the same dataset-scale spatial sensitivity
audit performed by SSAT. The reference implementation is independent: it
shares experiment artifacts and fixed numerical choices but imports no SSAT
implementation.

The quantitative values below come from two independent 200-sample runs.
Their raw and analysis canonical hashes agree. Detailed commands and output contracts are in
`experiments/reference_comparison/captum_baseline/README.md`.

## Experiment and environment

- 4×4 grid, five perturbation operators, three seed salts, and two
  area-matched controls per target region.
- 200 A-audit samples for each of the shortcut and normal models, plus 200
  B-audit samples for the auxiliary control: 291,200 raw rows per run.
- 1,000 fixed-seed bootstrap resamples and 95% intervals.
- NVIDIA GeForce RTX 4090, PyTorch 2.8.0+cu129, torchvision 0.23.0+cu129,
  CUDA 12.9, and Captum 0.9.0.

## Q1–Q5 result

| Question | Captum reference | SSAT crop-free result | Match |
|---|---:|---:|---|
| Q1 patch-region rank | 1 | 1 | Yes |
| Q2 patch multiplier | 173.018 | 175.651 | Both PASS (threshold 3.0) |
| Q3 patch rank in normal model | 16 | 16 | Yes |
| Q4 operators with patch at rank 1 | 5/5 | 5/5 | Yes |
| Q5 generalization-gap margin | 95.75 points | 95.75 points | Yes |

All five pre-registered questions pass. The auxiliary B control gives patch
region rank 2 in both workflows. The effective-area check passes with 64
source pixels and 3,136 model-space pixels for every grid cell.

## Measured engineering comparison

Python SLOC is counted as physical lines containing non-comment Python
tokens; tests and generated outputs are excluded.

| Measure | Captum custom workflow | SSAT |
|---|---:|---:|
| User-authored Python SLOC | 1,461 | 704 existing experiment-glue SLOC |
| Audit-config builder SLOC | Included above | 67 |
| Semantic execution stages | 8 | 3 (`run`, `analyze`, `report`) |
| Explicit workflow-design decisions | 8 | 1 primary metric/config choice |

The eight Captum stages are mask/baseline preparation, evaluation loop, raw
serialization, controls/seeds, multi-level aggregation, uncertainty and
stability, preprocessing/area validation, and report generation. The eight
explicit decisions are raw schema, aggregation convention, degradation
sign, control definition, interval method, provenance content, cache/retry
identity, and resume conflict handling.

## Capability accounting

| Capability | Captum-based workflow | SSAT |
|---|---|---|
| Model wrapper | Custom | Built in |
| Dataset iteration | Custom | Built in |
| Region mask generation | Custom | Built in |
| Multiple fill strategies | Custom | Built in |
| Output to task metric | Captum primitive + custom wrapper | Metric interface |
| Per-sample serialization | Custom | Standard raw schema |
| Matched random control | Custom | Built in |
| Control normalization | Custom | Built in |
| Seed repeat | Custom | Built in |
| Bootstrap uncertainty | Custom | Built in |
| Operator consistency | Custom | Built in |
| Sample aggregation | Custom | Built in |
| Region aggregation | Custom | Built in |
| Class aggregation | Custom | Built in |
| Dataset aggregation | Custom | Built in |
| Preprocessing validation | Custom | Built in |
| Mask-area validation | Custom | Built in |
| Resolved configuration | Custom | Automatic |
| Provenance | Custom | Automatic |
| Cache | Custom | Built in |
| Resume | Custom | Built in |
| Report generation | Custom | Built in |

## Reproducibility and parity

- Run A was intentionally stopped at 10,000 rows and resumed to 291,200.
  A completed rerun produced zero new rows and zero model forward evaluations.
- Run A and fresh run B have the same raw canonical hash:
  `c476044bd0beb3147afaa3a4ca1d9fb09fe498819539a8ece10ee2f8d7c27477`.
- Their aggregate-table canonical hash also matches:
  `16c2e06a4d1a2925b50d0e91d180d94dba8a2f5b149fe5b749993ce5e28f501d`.
- All three deterministic constant-fill comparisons have identical complete
  region rankings and Spearman 1.0. Their maximum absolute degradation
  difference is 0.00238.
- Independently seeded stochastic operators preserve the scientific verdict
  while not hiding numerical differences: Spearman is 0.847 for Gaussian
  noise and 0.871 for patch shuffle. Q2 differs by 2.634 in absolute value
  (about 1.5% relative to SSAT), but both values are far above the fixed
  threshold.

## Runtime, memory, and storage (revision 1)

Measured for the revision (pre-registered, three alternating runs per
workflow; [`experiments/revision_1/c1_captum/C1_REPORT.md`](../experiments/revision_1/c1_captum/C1_REPORT.md)
sections 1-2). The SSAT side is the seven experiment scripts counted above.
Times are after the 1.0.1 planner fix; before it, SSAT took 1,207 s.

| Measure | Captum reference | SSAT |
|---|---:|---:|
| Perturbed model evaluations | 291,200 | 310,400 |
| End-to-end wall time | 141 s | 389 s |
| Mean GPU utilization, audit stage | 82 % | 13 % |
| Peak GPU memory above idle | 3.8 GiB | 3.2 GiB |
| Stored output | 58.9 MB | 70.7 MB |

SSAT stores full logits, the resolved configuration, provenance, reliability
grades, and an HTML report; the Captum workflow stores one scalar row per
item, so another metric needs new model evaluations.

## Second setting: ImageNet (revision 1)

A second comparison audits 1,000 ImageNet validation images (one per class)
with `mobilenetv2_050.lamb_in1k`, crop-free preprocessing, a 4x4 grid, mean
fill, blur, and Gaussian noise (three seeds), and three matched controls per
cell: 320,000 perturbed evaluations in each workflow. The Captum workflow
([`experiments/revision_1/c1_captum/imagenet_workflow/`](../experiments/revision_1/c1_captum/imagenet_workflow/README.md))
is a copy of the synthetic one adapted to ImageNet; SSAT uses the case-study
configuration with a different sample list
([C1_REPORT.md](../experiments/revision_1/c1_captum/C1_REPORT.md) section 5).

| Measure | Captum-based workflow | SSAT |
|---|---:|---:|
| Code or configuration for this setting | 1,023 lines | 31-line config |
| Changed from the synthetic setting | 419 lines added or changed | 2 config lines |
| End-to-end wall time (mean of 3) | 185 s | 1,334 s |
| Mean GPU utilization, audit stage | 67 % | 4 % |
| Peak GPU memory above idle | 11.4 GiB | 1.5 GiB |
| Peak host RSS, summed process tree | 12.0 GiB | 34.7 GiB |
| Stored output | 76 MB | 1,346 MB (1,000-class logits) |

- **Same results.** For mean fill and blur, the item-level Pearson
  correlation is 0.999 and the dataset-level 16-cell rankings are identical;
  clean top-1 is 62.3 % in both. The remaining differences come from the
  resize implementation (GPU bicubic in the Captum workflow, PIL in SSAT).
- **Speed is a limitation of SSAT's design.** SSAT is 7.2x slower here
  (2.8x in the synthetic setting). It perturbs each item in source space on
  the CPU and applies the model's preprocessing per item, through one
  pipeline shared by every region kind (including frame-dependent skeleton
  body parts), operator, and adapter. The purpose-built workflows composite
  regions and resize on the GPU. Region masks, including skeleton masks,
  could be batched on the GPU as well; a GPU pipeline, as the default or as
  an option where the regions, operators, and adapter allow it, is planned.
- Setup time was not measured, because it depends on the implementer; code
  size, changes, and capabilities are reported instead.

## Interpretation

Captum provides the low-level region-ablation operation. Dataset iteration,
task-metric conversion, matched controls, seed repetition, uncertainty,
multi-level aggregation, preprocessing and area validation, raw schemas,
cache/resume, provenance, and reporting remain workflow code that the user
must design and maintain. The comparison therefore concerns an occlusion
primitive versus a reproducible intervention-audit protocol, not competing
attribution algorithms. In both settings each workflow has one
implementation, so the code counts are indicators rather than a general
measure of engineering effort, and SSAT is not the faster option.

Machine-readable measurements are stored in
`experiments/reference_comparison/captum_baseline/measured_results.json`,
`experiments/revision_1/c1_captum/summary_plan_cache/`, and
`experiments/revision_1/c1_captum/summary_std/`.
