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
