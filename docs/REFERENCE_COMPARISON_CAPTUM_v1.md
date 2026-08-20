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

## Interpretation

Captum provides the low-level region-ablation operation. Dataset iteration,
task-metric conversion, matched controls, seed repetition, uncertainty,
multi-level aggregation, preprocessing and area validation, raw schemas,
cache/resume, provenance, and reporting remain workflow code that the user
must design and maintain. The comparison therefore concerns an occlusion
primitive versus a reproducible intervention-audit protocol, not competing
attribution algorithms.

Machine-readable measurements are stored in
`experiments/reference_comparison/captum_baseline/measured_results.json`.
