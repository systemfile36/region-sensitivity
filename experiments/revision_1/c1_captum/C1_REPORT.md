# C1 report: Captum comparison, resource measurement (C1-min) and second setting (C1-std)

Sections 1-4 are C1-min. Pre-registered: `protocol.json` and
`command_map.md` (commit "docs(revision-1): Pre-register C1 resource
comparison and add measurement script", before any measurement). Scope: the
existing synthetic-shortcut comparison
(`docs/REFERENCE_COMPARISON_CAPTUM_v1.md`) with runtime, memory, GPU, and
storage added. Section 5 is C1-std, the second setting on ImageNet, which the
author decided to run after C1-min (`protocol_std.json`).

The two workflows:
- **Captum reference workflow:** `run.py audit`, `analyze`, `report`.
- **SSAT workflow:** the seven experiment scripts whose code the existing
  comparison counted (`command_map.md`).

Each workflow was measured 3 times, in alternating order, on 2026-09-30
from 06:06Z to 07:14Z. Nothing else ran in the container, and the host
desktop session stayed open. The host has 32 CPU cores, 125 GiB RAM, and
an RTX 4090. Software: captum 0.9.0, torch 2.8.0+cu129, the versions of
the original comparison. Deviations: `../deviations.md` D-010 (the SSAT
execution settings in the command map were corrected; commands unchanged)
and D-012 (a planner inefficiency in `ssat` was fixed after these results
were seen, and both workflows were measured again; section 2).

## 1. Runtime, memory, GPU, and storage, pre-registered measurement (`summary/`)

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
  results (section 3).
- **SSAT is about 8x slower in this measurement**: 7.9x per perturbed item,
  and 20 min vs 2.4 min end to end. Most of this was a planner
  inefficiency, since fixed (section 2). The two workflows also evaluate
  items differently (`captum_baseline/workflow.py`):
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
  - **Correction (D-012).** This report first attributed the time
    difference mostly to these two evaluation paths. Section 2 shows that
    about two thirds of the SSAT time was the planner inefficiency; the
    evaluation paths account for the remaining 2.8x. The difference is not
    the cost of the added workflow functions: the analysis and report
    stages take 10 s and 21 s.
- **Inside the SSAT audit stage** (loop times from the dump manifests):
  - **S-1:** single-operator runs without controls cost 0.94-1.07 ms per
    item.
  - **S-4 / S-5:** runs with 2 matched controls per region and 3 seeds
    cost 3.76-3.97 ms per item, plus about 25 s of metric computation
    each.
  - The difference between them is the planner inefficiency (section 2).
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

## 2. Post hoc: planner fix and re-measurement (D-012, `summary_plan_cache/`)

This section was added after the results of section 1 were seen.

**Cause.**
- `PlanBuilder.materialize` recomputed all of a sample's items for each
  chunk, and the runtime materializes each chunk twice (worker and main
  process).
  - S-1 has 16 items per sample in 1 chunk.
  - S-4 / S-5 have 720 items per sample in 45 chunks (`variants_per_chunk`
    16), so every item was built about 90 times.
- A diagnostic outside the protocol confirmed this:
  - In 2-sample runs, the S-4 run stage built 129,600 work items for 1,440
    items; S-1 built 64 for 32.
  - Planner time alone, on two P-cores, was 2.78 ms per item for S-4 and
    0.06 ms for S-1. This matches the S-4 / S-5 vs S-1 gap in section 1.
- **Fix** ("fix(plan): Recompute a sample's work items once per sample in
  materialize"): a one-entry per-sample cache. Afterwards the same S-4 run
  builds 1,440 work items. Item ids, chunk ids, and logits are identical
  before and after the fix.

**Re-measurement.** Both workflows were measured with the same scripts,
commands, settings, and checks: 3 alternating repeats on 2026-10-01 from
11:00Z to 11:27Z, at "docs(revision-1): Record the post hoc C1 planner fix and
re-measurement plan (D-012)" (clean tree). Mean of 3 repeats (range in
parentheses).

| Measure | Captum, section 1 | Captum, post hoc | SSAT, section 1 | SSAT, post hoc |
|---|---|---|---|---|
| End-to-end wall time | 143 s | 141 s (139.9-141.3) | 1,207 s | 389 s (384-391) |
| - audit stage | 133 s | 130.5 s | 1,186 s | 368 s |
| - S-4 / S-5 | – | – | 579 / 570 s | 166 / 166 s |
| - analysis + report stage | 10.4 s | 10.2 s | 20.8 s | 20.3 s |
| Wall time per perturbed item | 0.49 ms | 0.48 ms | 3.89 ms | 1.25 ms |
| S-4 / S-5 loop time per item | – | – | 3.76-3.97 ms | 0.99-1.01 ms |
| Mean GPU utilization, audit stage | 81 % | 82 % | 5 % | 13 % |
| Peak GPU memory above idle | 3.8 GiB | 3.8 GiB | 3.2 GiB | 3.2 GiB |
| Peak host RSS, largest process | 1.6 GiB | 1.6 GiB | 1.8 GiB | 1.8 GiB |
| Peak host RSS, summed process tree | 6.4 GiB | 6.5 GiB | 4.8 GiB | 5.2 GiB |
| Storage, total | 58.9 MB | 58.9 MB | 70.7 MB | 70.7 MB |

- **The fix removes about two thirds of the SSAT time.** SSAT is now 2.8x
  slower than Captum end to end (6.5 min vs 2.3 min), and 2.6x per
  perturbed item.
  - S-4 / S-5 items now cost the same as S-1 items (0.92-1.05 ms).
  - Captum did not change (within 2 %).
- **The remaining 2.6x per item fits the evaluation paths in section 1.**
  SSAT perturbs and preprocesses each item on the CPU, and the GPU is used
  13 % of the time. This was not profiled further.
- **Memory and storage did not change.** The SSAT process-tree peak is in
  S-2, the accuracy step, which does not use the planner.
- **Results are the same.** All 6 runs pass every check in
  `summary_plan_cache/verification.json`. The reliability grades equal the
  section 1 repeats.

## 3. Identical results (`verification.json`)

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

## 4. Engineering comparison (existing measurement, unchanged)

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

## 5. Second setting on ImageNet (C1-std, `summary_std/`)

Pre-registered: `protocol_std.json` and `command_map_std.md` (commit
"docs(revision-1): Pre-register C1-std ImageNet comparison and add the Captum
ImageNet workflow", before the measurement).

- **Setting.** 1,000 ImageNet validation images (1 per class, salt
  `rev1-c1`), `mobilenetv2_050.lamb_in1k`, crop-free squash to 224x224, 4x4
  grid, mean fill / blur / Gaussian noise (3 seeds), 3 matched controls per
  cell: 320,000 perturbed items in each workflow.
- **Captum workflow:** `imagenet_workflow/`, a copy of the synthetic
  reference adapted to ImageNet (its `README.md` lists the changes); run as
  `run.py audit`, `analyze`, `report`.
- **SSAT workflow:** the case-study crop-free config with the sample list
  changed (`configs/imagenet_mnv2_050_crop_free_c1.yaml`); run as `ssat run`,
  `metrics`, `analyze`, `report`.
- **Measurement.** As in C1-min: 3 repeats per workflow in alternating
  order, on 2026-10-05 from 08:31Z to 09:47Z, at the pre-registration commit
  (clean tree). Nothing else ran in the container; the host desktop session
  stayed open. Summaries were generated from a clean tree at the same commit.
- **Setup time is not measured.** Implementation time and effort depend on
  the implementer's skill and approach and cannot be compared between
  people. The adaptation is compared through indicators only (section 5.4);
  setup time is a limitation (`protocol_std.json` decisions.setup_time).

### 5.1 Checks and result parity (`verification.json`, `parity*.csv`, `parity.json`)

- **All protocol checks pass.**
  - Captum: every repeat is complete with 320,000 raw rows, and the area
    check passes. Raw and analysis canonical hashes are identical across the
    3 repeats.
  - SSAT: every repeat has 320,000 perturbed items and no failures.
    `counts_by_status` and the margin-drop grade counts (1,989 HIGH /
    4,192 LOW / 9,819 UNRELIABLE of 16,000 anchors) are identical across
    repeats.
  - Informational: the SSAT items equal the stored 10k crop-free run's items
    for the same samples (identical item-id sets). 74.9 % of the margin-drop
    values are bitwise equal, and the largest difference is 0.026 (D-001).
- **Parity, deterministic operators (primary criteria pass).** Unit: target
  cells, seed-averaged per (sample, cell, operator).

| Operator | Item Pearson | Item Spearman | Median abs. diff | 16-cell profile Spearman | Rank order identical | Top cell | Per-sample top cell agreement |
|---|---|---|---|---|---|---|---|
| mean fill | 0.9991 | 0.998 | 0.013 | 1.000 | yes | same | 96.4 % |
| blur | 0.9988 | 0.998 | 0.012 | 1.000 | yes | same | 96.3 % |
| pooled (all variants) | – | – | – | 1.000 | yes | same | 94.6 % |
| Gaussian noise (draws differ by design) | 0.915 | 0.878 | 0.029 | 0.979 | no | same | 60.3 % |

- **Clean results.** Top-1 is 62.3 % in both workflows; per-sample
  correctness agrees for 99.6 % of samples; the median clean-margin
  difference is 0.021 (max 0.21).
- The remaining item-level differences come from the resize: the Captum
  workflow resizes on the GPU (torch bicubic, antialiased) and SSAT with PIL
  (`protocol_std.json` parity.criteria_note).
- **Controls (informational).** The share of target rows with z > 2 is
  13.5 % (Captum) and 14.9 % (SSAT). The units differ (Captum: per-seed rows,
  control positions shared across operators; SSAT: seed-averaged anchors,
  positions drawn per item), so this is not a parity criterion.

### 5.2 Runtime, memory, GPU, and storage (`resources.csv`, `runs.csv`, `storage.csv`)

Mean of 3 repeats (range in parentheses).

| Measure | Captum workflow | SSAT workflow |
|---|---|---|
| Commands | 3 | 4 |
| Perturbed model evaluations | 320,000 | 320,000 |
| End-to-end wall time | 185 s (181-192) | 1,334 s (1,318-1,354) |
| - audit stage | 173 s | 1,033 s (`ssat run`) |
| - analysis + report stage | 12.4 s | 301 s (metrics 148, analyze 112, report 41) |
| Wall time per perturbed item | 0.58 ms | 4.17 ms |
| Audit wall time per perturbed item | 0.54 ms | 3.23 ms |
| Mean GPU utilization, audit stage | 67 % | 4 % |
| Peak GPU memory above idle (device total) | 11.4 GiB (13.0) | 1.5 GiB (3.1) |
| Peak host RSS, largest process | 4.1 GiB | 16.2 GiB (14.8-18.9; `ssat run`, largest process of the worker pool, D-009) |
| Peak host RSS, summed process tree | 12.0 GiB (audit, 12 workers) | 34.7 GiB (`ssat run`, 12 workers) |
| Storage, total | 75.9 MB (237 B per item) | 1,346 MB (4.2 kB per item) |

- **SSAT is 7.2x slower end to end** (22.2 min vs 3.1 min; 6.0x in the
  audit stage). In the synthetic setting the gap was 2.8x (section 2).
- **Repeatability.** The coefficient of variation of the end-to-end time is
  3.2 % for Captum (the first repeat was 10 s slower) and 1.4 % for SSAT.
- **Memory goes to different places.** SSAT holds more host memory, mostly
  in the `ssat run` worker pool; the Captum workflow holds more GPU memory
  (one sample's control images at source size per call, up to a 12 M-pixel
  budget, plus Captum's expanded copies). Both fit the 125 GiB / 24 GiB host.
- **Storage.** SSAT stores 1,000-class logits for every item: `perturbed/`
  is 1,192 MB (89 % of the total), `report/` 89 MB, `metrics/` 27 MB,
  `analysis/` 22 MB, `index/` 12 MB, `clean/` 4 MB. The Captum workflow
  stores one row per item: `raw/` 27 MB (85 B per row) and `analysis/` 49 MB.
  SSAT's metrics, grades, and new metrics can be recomputed without
  inference; the Captum rows cannot give another metric without new
  inference.

### 5.3 Where the time goes, and the limitation it shows

- **The SSAT audit is CPU-bound.** Its per-item time (3.23 ms) matches A4's
  post-fix 3.6 ms on the exact protocol, and A4's component profile
  (`../a4_scaling/A4_REPORT.md` section 1) splits it: the model forward pass
  is 0.14 ms per item; model preprocessing in the main process is 2.19 ms
  and worker-side preparation (decode, mask, perturbation) 1.94 ms.
- **The Captum workflow is specialized for this audit.** It builds one
  full-frame candidate per sample and variant in the loader workers,
  composites every region on the GPU with `FeatureAblation`, and resizes on
  the GPU (67 % utilization).
- **Why the gap is larger on ImageNet.** SSAT perturbs and resizes each item
  at source resolution on the CPU; ImageNet images have a median of 187,500
  pixels against 1,024 for the 32x32 synthetic images. It also writes 1,000
  logits per item. The analysis + report stage (301 s vs 12 s) adds 9
  metrics, reliability grades, and an HTML report, which the Captum workflow
  does not compute.
- **Structural cause.** SSAT runs every region kind, operator, and adapter
  through one per-item pipeline in source space: grid cells, explicit masks,
  matched controls, and per-frame skeleton body-part masks on video, as in
  the NTU RGB+D case study, use the same path, and each adapter applies its
  model's own preprocessing (PIL/timm transforms, MMAction2 pipelines) to
  the perturbed source array. This keeps semantic regions and grid cells on
  one consistent path, but it leaves the GPU almost idle. The speed gap is
  the cost of that design choice, not of the grid setting, and it is a
  clear limitation.
- **Improvement needed (not done in this revision).** Where the region
  kind, operator, and adapter allow it, a GPU pipeline, as the default or
  as an option, would composite masks into GPU tensors and apply GPU
  preprocessing that matches the adapter's geometry, while keeping the same
  item identities and dumps.

### 5.4 Engineering indicators (`engineering.json`)

| Indicator | Captum workflow | SSAT |
|---|---|---|
| Code or config for the ImageNet setting | 1,023 SLOC | 31 config lines |
| Changed from the synthetic setting | 419 SLOC added or changed, 857 removed (from 1,461) | 2 config lines changed from the case-study config (sample list, relative depth) |
| Shared input script (subset rule) | 44 SLOC, used by both | same |
| Commands | 3 | 4 |
| Capabilities (17 listed) | 16 custom (the report as Markdown), reliability grades not implemented | all built in or automatic |

- SLOC follow the `count_python_sloc` rule; changes are `difflib` opcodes
  over those lines. Per file: `workflow.py` 284 added or changed / 375
  removed (the batched multi-run loop, JSON manifests, and the Q5 accuracy
  runs gave way to the per-sample loop); `analysis.py` 134 / 481 (the Q1-Q5
  verdicts, SSAT parity, SLOC accounting, and the synthetic report gave way
  to the region profile and control summary); `run.py` 1 / 1.
- **Qualitative.** Adapting the Captum workflow meant changing data
  loading (variable image sizes, so one sample per ablation call),
  re-implementing the model preprocessing on the GPU with a check against
  timm's data config, restructuring the control path (`FeatureAblation`
  cannot ablate overlapping groups in one mask, so controls are batched per
  sample with a per-example two-group mask), and replacing the analysis
  questions. On the SSAT side the sample list changed; model loading,
  preprocessing geometry and effective areas, controls, provenance, resume,
  grades, and the report came from the existing pipeline.
- Single implementation per workflow; SLOC depend on the implementer and
  are indicators, not a productivity measure.

## Interpretation

- **Supported (two settings: synthetic shortcut and ImageNet).**
  - Both workflows reach the same scientific results: identical Q1-Q5
    verdicts in the synthetic setting; in the ImageNet setting, item-level
    Pearson 0.999 and identical 16-cell rankings for the deterministic
    operators. Each workflow is reproducible across repeats.
  - The SSAT workflow needed about half the user-authored code and one
    explicit design decision instead of eight (synthetic). Moving to
    ImageNet changed 2 config lines in SSAT and 419 lines of code in the
    Captum workflow, whose data loading, preprocessing, and control path had
    to be restructured.
  - SSAT stores more reusable evidence (full logits, resolved config and
    provenance, reliability grades, report): a similar number of bytes in
    the 10-class synthetic setting, 18x more in the 1,000-class setting.
- **Not supported.**
  - SSAT does not reduce runtime. With the planner fix it is 2.8x slower
    than the purpose-built, GPU-batched Captum workflow in the synthetic
    setting (6.5 min vs 2.3 min) and 7.2x slower on ImageNet (22.2 min vs
    3.1 min). The cause is structural (section 5.3): one per-item CPU
    pipeline in source space shared by grid, control, and semantic regions.
    This is a limitation; a GPU pipeline where possible is future work.
  - SSAT needs more host memory and, with 1,000 classes, much more storage.
  - A general reduction in engineering effort or setup time is not shown:
    setup time was not measured, because it depends on the implementer and
    cannot be compared between people, and each setting has one
    implementation per workflow.

## Action under the original plan (section 15): narrow the claim

C1-std was run; the engineering claim stays narrowed to the measured cases
(implementation plan section 9.4), now two settings, and the runtime result
is stated as a limitation together with its structural cause.

The drafts use the post hoc synthetic times (section 2), because the
revised release contains the fix; C1-std ran on the fixed code. The Response
to Reviewers should also give the pre-registered synthetic times and the fix
(D-012). The author decides which times the paper reports.

**Impact paragraph (replaces "reduces the repetitive engineering work"),
draft:**

> In two case studies, the synthetic audit and a 1,000-image ImageNet audit,
> a Captum-based re-implementation had to add matched controls, seed
> repeats, uncertainty, multi-level aggregation, caching, provenance, and
> reporting around Captum's feature-ablation primitive: 1,461 lines of
> workflow code, of which 419 changed for ImageNet. SSAT needed 704 lines of
> experiment code and, for ImageNet, a two-line config change; both reached
> the same results. SSAT's contribution is a standardized audit workflow with
> automatic provenance and reliability-aware analysis, not lower
> computational cost: the purpose-built Captum workflows batch region
> ablations on the GPU and ran 2.8x and 7.2x faster on one RTX 4090.

**Captum comparison section: add resource rows for both settings**
(sections 2 and 5.2: time, per-item time, GPU utilization, memory, storage),
with a note that SSAT stores full logits and provenance.

**Limitations, draft:**

> The engineering comparison covers two settings with one implementation of
> each workflow; implementation time was not measured because it depends on
> the implementer. SSAT passes every region kind, operator, and model
> adapter through one per-item CPU pipeline in source space, so that grid
> cells and semantic regions such as skeleton body parts share one path;
> this leaves the GPU mostly idle, and purpose-built GPU-batched
> implementations were 2.8-7.2x faster. A GPU pipeline, by default or as an
> option where the regions and the adapter allow it, is planned.

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

D-012: after section 1 was seen, a planner inefficiency in `ssat` was found
and fixed ("fix(plan): Recompute a sample's work items once per sample in
materialize"), and both workflows were measured again with the same protocol
(section 2). The section 1 results stay as recorded.

The section 1 summaries (`summary/`) were generated from a clean tree at
"feat(revision-1): Add C1 resource summaries and verification against stored
runs"; the section 2 measurement and summaries (`summary_plan_cache/`) at
"docs(revision-1): Record the post hoc C1 planner fix and re-measurement plan
(D-012)".

C1-std (section 5) has no deviations from `protocol_std.json`. Its parity
thresholds were set after a 6-sample development smoke test, before the
pre-registration commit, as the protocol states. The section 5 measurement
and summaries (`summary_std/`) were made from a clean tree at
"docs(revision-1): Pre-register C1-std ImageNet comparison and add the
Captum ImageNet workflow".
