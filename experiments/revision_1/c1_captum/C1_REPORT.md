# C1 report: Captum comparison, resource measurement (C1-min)

Pre-registered: `protocol.json` and `command_map.md` (commit `77d437c`,
before any measurement). Scope: C1-min, the existing synthetic-shortcut
comparison (`docs/REFERENCE_COMPARISON_CAPTUM_v1.md`) with runtime, memory,
GPU, and storage added. C1-std (a second setting on ImageNet) was not run;
that is the protocol default and is left to the author.

The two workflows:
- **Captum reference workflow:** `run.py audit`, `analyze`, `report`.
- **SSAT workflow:** the seven experiment scripts whose code the existing
  comparison counted (`command_map.md`).

Each workflow was measured 3 times, in alternating order, on 2026-09-30
from 06:06Z to 07:14Z. Nothing else ran in the container, and the host
desktop session stayed open. The host has 32 CPU cores, 125 GiB RAM, and
an RTX 4090. Software: captum 0.9.0, torch 2.8.0+cu129, the versions of
the original comparison. Deviation: `../deviations.md` D-010 (the SSAT
execution settings in the command map were corrected; commands unchanged).

## 1. Runtime, memory, GPU, and storage (`resources.csv`, `runs.csv`, `measurements.csv`)

Mean of 3 repeats (range in parentheses).

| Measure | Captum reference workflow | SSAT workflow |
|---|---|---|
| Commands | 3 (or 1 with `run.py all`) | 7 scripts |
| Perturbed model evaluations | 291,200 | 310,400 |
| End-to-end wall time | 143 s (142.7-143.5) | 1,207 s (1,196-1,227) |
| - audit stage (all model evaluations) | 133 s | 1,186 s |
| - analysis + report stage | 10.4 s | 20.8 s |
| Wall time per perturbed item | 0.49 ms | 3.89 ms |
| Mean GPU utilization, audit stage | 81 % | 5 % |
| Peak GPU memory above idle (device total) | 3.8 GiB (5.9 GiB) | 3.2 GiB (5.4 GiB) |
| Peak host RSS, largest process | 1.6 GiB | 1.8 GiB |
| Peak host RSS, summed process tree | 6.4 GiB (main + 4 loader workers) | 4.8 GiB (accuracy step, 4 loader workers); 1.8-1.9 GiB in the audits |
| Storage, total | 58.9 MB (202 B per item) | 70.7 MB (228 B per item) |

- **Repeatability.** The coefficient of variation of the end-to-end time
  is 0.3 % for Captum and 1.5 % for SSAT. Every repeat produced the same
  results (section 2).
- **SSAT is about 8x slower in this setting**: 7.9x per perturbed item,
  and 20 min vs 2.4 min end to end. The difference comes from how each
  workflow evaluates items (`captum_baseline/workflow.py`):
  - **Captum reference.** It is written for this one experiment:
    squeezenet1_0, 32x32 images, grids that divide the image.
    - It builds one perturbed full frame per sample, operator, and seed on
      the CPU.
    - Captum's `FeatureAblation` composites each region into the image on
      the GPU. The model wrapper then resizes and normalizes on the GPU,
      16 samples x 16 perturbations per forward call.
  - **SSAT.** It perturbs each item in the source image on the CPU and
    applies the model's preprocessing pipeline per item. It then writes the
    full logits with region, operator, seed, and area metadata.
    - The synthetic scripts set no `runtime`, so `ssat` runs its defaults:
      one process, batch 32 (D-010).
    - A4 found the same pattern on ImageNet: `ssat run` is CPU-bound, with
      GPU utilization below 5 %.
  - The time difference is therefore mostly an implementation difference
    (a special-purpose batched GPU path vs a general per-item pipeline). It
    is not the cost of the added workflow functions: the analysis and
    report stages take 10 s and 21 s.
- **Inside the SSAT audit stage** (loop times from the dump manifests):
  - **S-1:** single-operator runs without controls cost 0.94-1.07 ms per
    item.
  - **S-4 / S-5:** runs with 2 matched controls per region and 3 seeds
    cost 3.76-3.97 ms per item, plus about 25 s of metric computation
    each.
  - The control / seed runs were not profiled further.
  - The 19,200 items that SSAT evaluates beyond Captum (the S-1
    single-operator runs; `command_map.md`) take 32 s, 2.6 % of the SSAT
    time.
- **Memory is small for both workflows.** GPU memory above the idle
  baseline is 3.2-3.9 GiB in both. The idle baseline, from the host
  desktop, was 2.0-2.2 GiB. The process-tree sums count shared pages once
  per process, so they are upper bounds.

### Storage by content (`storage.csv`)

| Workflow | Entry | MB | Share | Content |
|---|---|---|---|---|
| Captum | `raw/` | 23.5 | 40 % | one row per item: ids, clean and perturbed margin, degradation, areas (81 B per row) |
| Captum | `analysis/` | 35.4 | 60 % | aggregate, control, seed, bootstrap, operator tables, each in Parquet and CSV |
| Captum | files | 0.02 | 0 % | manifest, provenance, accuracy, report, comparison metrics |
| SSAT | `dumps/` | 43.7 | 62 % | clean and per-item logits (10 classes), region / operator / seed / area metadata, item index, manifests with resolved config (141 B per item) |
| SSAT | `metrics/` | 13.9 | 20 % | per-item margin drop and region aggregates |
| SSAT | `report/` | 9.0 | 13 % | two HTML reports with data CSVs and figures |
| SSAT | `analysis/` | 3.8 | 5 % | reliability grades, controls, seed and strategy stability, intervals, rank correlation |
| SSAT | heatmaps, files | 0.25 | 0 % | heatmaps, verdicts, accuracy, report, region CSVs |

- **The totals are similar, but the workflows store different evidence.**
  - Captum stores one scalar per item, so any new metric needs new model
    evaluations.
  - SSAT stores full logits, so metrics can be recomputed without
    inference, and it adds reliability grades and an HTML report.
- **Totals depend on the number of classes.** With 1,000 ImageNet classes,
  SSAT's raw dump grows to about 3.6 kB per item (A4), while Captum's
  scalar rows would not grow. Total bytes therefore do not rank the
  workflows (implementation plan section 9.2).

## 2. Identical results (`verification.json`)

- **Captum, all 3 repeats:**
  - raw rows 291,200;
  - raw canonical hash `c476044b...` and analysis canonical hash
    `16c2e06a...`, identical to the stored runs A and B;
  - Q1-Q5 and the area check pass.
- **SSAT, all 3 repeats:**
  - all 9 dumps have the stored item counts;
  - `verdicts.json` and `accuracy.json` are identical to the stored crop-free
    results;
  - the Q2 multiplier's relative difference is 0.
- **Reliability grades (informational).**
  - The shortcut model's grades are identical to the stored ones.
  - For the normal model, all three fresh repeats give 764 / 417 / 2,019
    (HIGH / LOW / UNRELIABLE), against 763 / 418 / 2,019 stored: 1 of 3,200
    region rows differs.
  - The stored analysis was produced on 2026-08-14 with metrics schema
    1.0.0. The current `ssat` writes 1.1.0 and refuses to reanalyze the
    stored metrics. The difference is therefore between code versions, not
    between runs.

## 3. Engineering comparison (existing measurement, unchanged)

| Measure (one synthetic setting) | Captum reference workflow | SSAT |
|---|---|---|
| User-authored Python SLOC | 1,461 | 704 experiment glue, of which 67 build the audit configs |
| Semantic execution stages | 8 | 3 (`run`, `analyze`, `report`) |
| Explicit workflow-design decisions | 8 | 1 |
| Capabilities (22 listed) | all custom (Captum supplies the ablation primitive) | built in or automatic |
| Commands run in this measurement | 3 | 7 scripts |

- These counts come from `measured_results.json` (2026-08-20). C1-min adds
  no new effort data.
- **Single-implementation case study.** SLOC and design decisions depend on
  who wrote the code and are not a productivity measurement.
- The SSAT workflow is invoked as seven scripts because Q1-Q4 were answered
  from separate single-operator runs, while controls, seeds, and reports
  came from two all-operator runs. The "3 semantic stages" count the
  `ssat` operations, not the script invocations.

## Interpretation

- **Supported (this setting).**
  - Both workflows reach identical scientific results, and each is exactly
    reproducible across repeats.
  - The SSAT workflow needed about half the user-authored code and one
    explicit design decision instead of eight.
  - SSAT stores more reusable evidence (logits, resolved config and
    provenance, reliability grades, report) in a similar number of bytes.
  - Both workflows need a few GiB of host and GPU memory.
- **Not supported.**
  - SSAT does not reduce runtime: here it is about 8x slower than a purpose-built,
    GPU-batched Captum workflow, although both finish in minutes.
  - One synthetic setting and one implementer do not show a general
    reduction in engineering burden.
  - C1-std, a second setting, was not run.

## Action under the original plan (section 15): narrow the claim

C1-std is not run (protocol default). The engineering claim is narrowed to
the measured case (implementation plan section 9.4).

**Impact paragraph (replaces "reduces the repetitive engineering work"),
draft:**

> In a synthetic case study, re-implementing SSAT's audit around Captum's
> feature-ablation primitive required 1,461 lines of custom workflow code
> for matched controls, seed repeats, uncertainty, multi-level aggregation,
> caching, provenance, and reporting, compared with 704 lines of experiment
> code around SSAT; both reached identical verdicts. SSAT's contribution is
> a standardized audit workflow with automatic provenance and
> reliability-aware analysis, not lower computational cost: the
> purpose-built Captum workflow, which batches region ablations on the GPU,
> ran in 2.4 min versus 20 min for SSAT on one RTX 4090. We do not claim a
> general reduction in engineering effort beyond this case study.

**Captum comparison section: add a resource table** (section 1: time,
per-item time, GPU utilization, memory, storage by content), with a note
that SSAT stores full logits and provenance.

**Limitations, draft:**

> The engineering comparison is a single-implementation case study in one
> synthetic setting; code size and design effort depend on the implementer.
> SSAT's general per-item pipeline is slower than a special-purpose,
> GPU-batched implementation (Section X).

**Paper locations:**
- the Impact paragraph;
- the Captum comparison section or table (`docs/REFERENCE_COMPARISON_CAPTUM_v1.md`
  is the repository counterpart);
- Limitations;
- Response to Reviewer (runtime / memory / storage request).

## Deviations

D-010: the command map first stated the case-study SSAT settings (batch
128, 12 workers). The synthetic scripts run with the `ssat` defaults (one
process, batch 32). The text was corrected after the first repeat pair; the
commands, measurements, and verification did not change.

All summaries were generated from a clean tree at `f2d3c7b`.
