# Deviations from the pre-registered revision plan

Each entry records what changed relative to the implementation plan
(`SSAT_minor_revision_implementation_plan.md`), when, why, and whether any
experiment result had been seen at that point. Phase 0 entries were all made
before any A1-C1 result existed.

## D-001 (2026-09-23, Phase 0): P0-3 logit parity is not bitwise; identity is gated instead

**Plan.** P0-3 requires the K=1 and K=3 ImageNet runs to share target
item ids and to have `np.array_equal` logits; plan sections 0.2 and 5.1 state
that the K=3 run's `control_index` 0 controls are "bit-for-bit" the K=1
controls, the premise of A2's nested K=20 design.

**Observed** (`phase0/summary/target_parity.json`, all 10,000 samples):

- Items, masks, and seeds are identical in all four pairs: same target
  item-id sets, same `region_params_json`/`effective_area_px`/`seed_used`,
  every K=1 control reappears as the K=3 `control_index == 0` control,
  and clean logits are bitwise equal.
- Logits are not always bitwise equal. mnv2_100: targets 100 % equal,
  `control_index` 0 controls differ by at most 3.8e-6. mnv2_050: 99.7 % of
  targets equal (the rest within 5.7e-6, no top-1 change), but 33 % of
  `control_index` 0 controls differ by 1e-3 to 2.4e-2 (152-160 top-1
  changes out of 800,000).
- Cause: PyTorch's default `torch.backends.cudnn.allow_tf32=True` with
  `cudnn.deterministic=False`. The same input can take a different
  convolution algorithm depending on the batch it is evaluated in; in the
  K=1 run the last 32 of each sample's 160 items form a separate chunk
  (`variants_per_chunk=128`), and exactly that block carries the large
  differences in the 20-sample diagnostic.

**Change.** `verify_target_parity.py` gates on identity and records the
bitwise result and a max-|diff| histogram instead of failing on it. The
plan's bitwise criterion is reported as not met. `ssat` defaults are not
changed during the revision (plan section 2.1).

**Consequences for later experiments.**

- The paper's accuracy/`margin_drop` table (K=1) and reliability numbers
  (K=3) describe the same target evaluations up to 5.7e-6 in logits
  (P0-2 reproduces both at the paper's rounding).
- A2: the K=20 run's `control_index` 0-2 items will match
  `results_control_3` as items, masks, and seeds, but not necessarily in
  logits. `verify_nested.py` reports identity plus the logit difference
  distribution, and A2's ablation compares K subsets within the K=20 run
  only (the plan's section 5.3 fallback).

## D-002 (2026-09-23, Phase 0): NTU and synthetic baselines read recomputed stores (new step P0-7)

**Plan.** Section 1.3 lists the stored NTU and synthetic metrics/analysis as
direct inputs.

**Observed.**

- The synthetic metrics stores use metrics schema 1.0.0 and hold only
  `margin_drop`; the current `ssat.metrics.store` (1.1.0) rejects them,
  so `AnalysisReader`-based tooling cannot open them.
- `ntu60_tsm_crop_free/run_manifest.json` was rewritten on 2026-08-27,
  after its metrics were computed (2026-08-21); the metrics store's
  `source_run_manifest_hash` no longer matches. The dump fragments are
  unchanged since 2026-08-21.
- All four analyses predate the last change to `ssat/analysis`
  (`145e5e1`, analysis vectorization).

**Change.** New step P0-7 (`phase0/recompute_small_baselines.py`) reruns
`ssat metrics` and `ssat analyze` with default settings on the unchanged
dumps into `results/phase0/recomputed/` and compares every table.
`BASELINE_RUNS` now points the NTU and synthetic runs at these stores
(dumps unchanged); the originals stay in place as `ORIGINAL_SMALL_RUNS`.

**Result** (`phase0/summary/recomputed_baselines.json`).

- NTU exact and crop-free: all 12 tables identical.
- Synthetic, over the stored metric (`margin_drop`): metrics,
  `control_comparison`, `strategy_stability`, `rank_correlation`, and
  `strategy_profile` identical; `seed_stability` within 1e-11; bootstrap
  CI bounds differ by up to 0.018 because the vectorized code feeds the
  per-sample means to the bootstrap in a different order. In
  `synthetic_shortcut` this moves region `r2/c1`'s CI across zero and
  changes 8 of 3,200 grades (HIGH 8.38 % -> 8.13 %); `synthetic_normal`
  grades are unchanged. The paper does not report synthetic grades; its
  Q1-Q5 values come from metrics and are unaffected.

## D-003 (2026-09-23, Phase 0): P0-2 check scope

- The "roughly 14-17 %" top-region share is gated on the paper figure's run
  (mnv2_050 exact, K=3 and K=1). The other runs are reported only: the
  four central cells lead in every run, at 14.6-17.3 % (exact) and
  12.3-15.1 % (crop-free).
- Synthetic Q1-Q5 are checked against `results_crop_free/verdicts.json`,
  which `evaluate.py` derived from separate single-operator runs, not from
  the `*_all_ops_thresholds_crop_free` baseline runs.

## D-004 (2026-09-24, Phase 1B): B1 additions and data source

A1 ran as registered (`a1_threshold/protocol.json`); no deviation.

**Added after the area tables were first computed, before any ranking or
grade comparison was run.**

- Under P_exact, 896 target cells (115 samples) have `effective_area_px == 0`:
  the cell lies entirely outside the center crop. The registered
  within-sample max/min area ratio is unbounded for those samples, so
  `within_sample_area_ratio.csv` reports them as a count and computes the
  quantiles over the other 9,885 samples. Ratios against a zero-area target
  are undefined in `control_area_ratio.csv` (counted, and treated as "not
  within tolerance", as `ssat.analysis.indexer` does).
- New table `zero_area_targets.csv`: grades and `margin_drop` of those cells.
- Extra outputs not listed in the protocol: `grade_transition_yardstick.csv`
  and `sample_spearman_hist.csv` (the histogram behind `fig_b1_rank.pdf`).

**Image source.** The ImageNet validation JPEGs the audits read
(`data/imagenet/ILSVRC/Data/CLS-LOC/val/`, per the case-study configs) are no
longer in the working tree. For `fig_b1_examples.pdf` the three pre-registered
samples were extracted from
`/media/limdongha/LARGE_DATA/imagenet-object-localization-challenge.zip` into
`results/b1/images/`, and `plot_figures.py` checks each file's SHA-256
against the `content_hash` the dumps recorded (all three match). A2 and A3
need the full 10,000-image set and cannot start until the directory is
restored.

**Location.** The example renderer is `b1_preprocessing/plot_figures.py`, not
`scripts/paper_figures/rev1_b1_examples.py` (implementation plan section 4.2),
because it reads raw images and dumps rather than `summary/` files only. Its
output `fig_b1_examples.pdf` goes to `results/b1/` (untracked) instead of
`summary/`, because it embeds ImageNet photographs that the ImageNet terms of
access do not allow us to redistribute in the repository.

## D-005 (2026-09-24, Phase 2 start): ImageNet images restored; A2 run mechanics

Recorded before any A2 run started.

**Images restored.** `data/imagenet/ILSVRC/Data/CLS-LOC/val/` (all 50,000
validation JPEGs) was extracted from
`/media/limdongha/LARGE_DATA/imagenet-object-localization-challenge.zip` with
the author's approval. Train and test images were not extracted; no config
reads them. New check `phase0/verify_imagenet_images.py`
(`phase0/summary/imagenet_images.json`): every one of the 10,000 audited
files matches the `content_hash` recorded by all four ImageNet baseline
dumps.

**Synthetic K=20 run.** Implementation plan section 5.2 names a
`run_synthetic_k20.py` that calls `_build_config` in-process. Instead,
`a2_control_count/make_inputs.py` writes that config (only
`controls[0].n_samples` changed, 2 -> 20) to
`configs/synthetic_shortcut_k20.yaml`, and the run goes through the same
`run_matrix.py` as the ImageNet runs, so it gets the same `ssat metrics` /
`ssat analyze` defaults as the P0-7 synthetic baseline stores and a timing
record in `run_matrix_log.jsonl`. `run_threshold_validation_full.py` is not
modified.

## D-006 (2026-09-28, Phase 3 start): DeiT-S fails selection condition 3 and is kept

Recorded before any A3 run started.

The author chose `convnext_tiny.fb_in1k` and
`deit_small_patch16_224.fb_in1k` (the first-priority candidates of
implementation plan section 6.1) on 2026-09-24, before
`a3_multi_arch/inspect_models.py` checked them. The check
(`a3_multi_arch/summary/model_selection.json`) shows that `deit_small` uses
`crop_pct` 0.9 instead of mobilenetv2_050's 0.875, so it fails condition 3
(same official eval geometry). Under exact preprocessing its model-space cell
areas differ from mobilenetv2_050's for 99.5 % of sample cells (center
0.94x, corner 1.16x on average). The other two transformer candidates
(`vit_small_patch16_224.augreg_in21k_ft_in1k`,
`swin_tiny_patch4_window7_224.ms_in1k`) also use `crop_pct` 0.9, so section
6.1's fallback ("no candidate passes") applies whichever transformer is
taken. `deit_small` is kept, and the section 4.5 rule decides the
comparison: all four models are compared on the crop-free runs (identical
geometry, verified cell by cell), and `deit_small` exact is reported per
model with its cell-area table. `convnext_tiny` passes all three conditions,
and its cell areas equal mobilenetv2_050's in both protocols.

Weights are downloaded into `HF_HOME=/workspace/data/hf_cache`
(implementation plan section 6.2); their SHA-256 are in
`model_selection.json`.

## D-007 (2026-09-29, Phase 3 analysis): post-hoc per-operator profile table

Added after the pre-registered `strategy_rank_corr.csv` was computed. For
`deit_small` crop-free the operator-pair Spearman over the 16 cells was
negative for `gaussian_noise` (-0.41 with `blur`, -0.44 with `mean_fill`),
while it is 0.64-0.99 in every other run. To show why, the new script
`a3_multi_arch/operator_profiles.py` writes `summary/operator_profile.csv`
(per run and operator: 16-cell mean target `margin_drop`, its range, and the
pairwise Spearman). The table is descriptive and changes no pre-registered
output.

## D-008 (2026-09-29, Phase 4 start): A4 budget and measurement mechanics

Recorded before any A4 measurement.

- **Budget.** The standard sweep of implementation plan section 7.2 has
  15,030,750 items (repeats included). At the ~260-290 items/s the
  reference workload runs at, `ssat run` alone takes about 16 h, and the
  whole sweep about 20 h, not the ~10 h the plan estimates. The design is
  unchanged.
- **GPU polling.** `run_scaling.py` uses its own nvidia-smi poller (same
  0.2 s interval) instead of `run_benchmark._poll_gpu_memory`, so that GPU
  utilization is recorded together with memory. `measure_step` is reused as
  planned.
- **Run split.** Preflight and audit loop are separated with `ssat
  --log-file` and the `runtime.started` / `runtime.finished` events (1 s
  resolution).
- **Warm-up.** Each axis's N=50 warm-up runs all four phases, not only
  `ssat run`, so metrics/analyze/report are also warm.
- **Component breakdown (section 7.4).** Stages (a)-(c) are timed. Dump
  writing (d) is not timed separately; it is part of the gap between the
  main-process stages and the measured audit loop.
- **Region id.** The sweep configs name their grid family
  `grid_<g>x<g>` (for example `grid_8x8`) instead of reusing `grid_4x4`.
  This only changes identifiers.

## D-009 (2026-09-30, Phase 4 analysis): host OOM during the sweep; meaning of the run-phase RSS

Recorded after the sweep finished, before `fit_scaling.py` ran.

- **Host OOM.** At 2026-09-30 02:27:10 KST (17:27:10Z), during
  `n1000_g8_v5_k3` repeat 2 (`ssat run` phase), the host ran out of memory.
  The kernel's task dump shows the run's 12 DataLoader workers
  (`pt_data_worker`) at 109.5 GiB RSS in total (8-19 GiB each) and the
  main process at 4.7 GiB. The kernel killed the author's VS Code process,
  not the audit. The measurement completed with return code 0 and a run
  time in line with the other two repeats (4,701 s vs 4,807 / 4,813 s). No
  other OOM event occurred during the sweep. All 41 measurements are kept.
- **Run-phase RSS is the largest single process, not the process tree.**
  `measure_step` reads `getrusage(RUSAGE_CHILDREN).ru_maxrss`, which on
  Linux is the peak RSS of the largest descendant. For `ssat run`, whose
  worker pool holds most of the memory, the recorded value (9-43 GiB)
  therefore understates the total. The OOM task dump above is the only
  measurement of the total. The metrics / analyze / report phases are single
  processes, so their values are totals. BENCHMARK_v1's run-phase value has
  the same meaning. No supplementary memory measurement was run (author's
  decision, 2026-09-30).
- **Timing outliers are kept.** `n500_g4_v5_k3` repeat 0 (815 s vs 624 /
  653 s) and `n1000_g6_v5_k3` repeat 0 (4,039 s vs 2,555 / 2,436 s) are
  slower than their repeats, most likely because of other activity on the
  host desktop. They are reported through the per-setting CV, and fits are
  shown with all repeats.
- **Workload table.** After the first `fit_scaling.py` output, the
  run-phase RSS was removed from the workload predictions and from the
  "items at which RSS reaches 125 GiB" rows. It had extrapolated the
  single-process value linearly, which the saturating data do not support.
  `fits.csv` still reports its fit (R^2 0.60 pooled).

## D-010 (2026-09-30, Phase 6 C1-min): SSAT execution settings in the command map

`c1_captum/command_map.md` (committed before the measurement) described the
SSAT workflow as running with batch 128 and 12 workers. Those are the
settings of the ImageNet case-study configs used in A4. The synthetic
scripts set no `runtime` section, so the `ssat` defaults apply:
- `num_workers` 0 (one process);
- `target_batch_size` 32;
- `variants_per_chunk` 16.

The resolved values are recorded in each dump's `run_manifest.json`. This
was found from the first repeat pair's process-tree data (two processes
during S-4 / S-5), and the text was corrected. The commands, measurements,
and verification criteria did not change.

## D-011 (2026-09-30, Phase 5 B2): single annotator, empty annotator metadata, post hoc group breakdown

- **One annotator.** Only `annotator_A.csv` was completed and committed
  (`3076550`); `annotator_B.csv` is blank. The analysis ran with
  `--annotators A`, the pre-registered single-annotator fallback. No
  annotator agreement can be reported, and the `annotator_*_only`
  robustness variants do not apply.
- **Annotator metadata.** `annotation/annotators.csv` (role, prior exposure
  to SSAT's NTU results, times) was still empty when the analysis ran.
  The report therefore does not state whether the rating was blind to
  earlier NTU results.
- **Post hoc analysis.** `group_breakdown.py` (per-group agreement across
  classes, top-group counts) was added after the pre-registered results
  were seen. It is labelled post hoc and used only to explain the
  pre-registered results.

## D-012 (2026-10-01, Phase 6 C1-min): post hoc SSAT planner fix and re-measurement

Recorded after the C1-min results were seen, before the re-measurement ran.

- **Finding.** `C1_REPORT.md` attributed the ~8x runtime gap to a general
  per-item pipeline. A diagnostic outside the protocol found a planner
  inefficiency instead:
  - `PlanBuilder.materialize` re-enumerated every item of a sample for
    each chunk, and the runtime materializes each chunk twice (worker and
    main process).
  - In a 2-sample S-4 run, the run stage built 129,600 WorkItems for
    1,440 items. S-1 built 64 for 32.
  - Planner time alone, on two P-cores: 2.78 ms per item for S-4 and
    0.06 ms for S-1. C1-min measured 3.76-3.97 ms and 0.94-1.07 ms per
    item in total.
- **Fix.** `fb7168b` adds a one-entry per-sample cache. Item ids, chunk ids,
  and dumps do not change: in 2-sample S-1 and S-4 runs before and after
  the fix, the item ids are identical and the logits differ by 0.
- **Re-measurement (post hoc).** Both workflows are measured again on the
  fixed code. Everything else is the same as C1-min: scripts, commands,
  execution settings, 3 alternating repeats, and verification criteria.
  - Outputs: `results/c1_plan_cache/` and `c1_captum/summary_plan_cache/`.
  - The pre-registered C1-min results stay as recorded. The
    re-measurement is reported next to them and labelled post hoc.
