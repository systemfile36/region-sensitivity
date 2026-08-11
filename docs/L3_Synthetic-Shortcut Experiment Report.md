# L3 Synthetic-Shortcut Experiment Report

> 2026-08-11
> lr=0.01, warmup-epoch=10, weight-decay=5e-4, epochs=40, batch-size=128, momentum=0.9

## Q1-Q5 (pre-registered, docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md section 5)

| Question | Result | Detail |
|---|---|---|
| Q1. identifies patch region | **PASS** | patch_region_rank=1 |
| Q2. separated from baseline | **PASS** | multiplier=35.10820063632307, threshold=3.0 |
| Q3. distinguishes M_normal | **PASS** | patch_region_rank_in_m_normal=16 |
| Q4. robust to fill strategy | **PASS** | reproduced_in=['constant_fill', 'mean_fill', 'blur', 'gaussian_noise', 'patch_shuffle'], min_required=2 |
| Q5. predicts generalization gap | **PASS** | shortcut_accuracy_drop_points=89.95, normal_accuracy_drop_points=-4.900000000000004, margin_points=94.85, threshold_points=10.0 |

**B auxiliary control** (not a pass/fail criterion): patch region rank under random placement = 5. not part of the pre-registered Q1-Q5 verdict; expected to be far from rank 1 if the tool tracks the patch's *position* rather than some unrelated bias toward the top-left cell.

## Section 3.5 -- fill-strategy sensitivity (Spearman correlation of region ranking against constant_fill)

| fill strategy | Spearman correlation |
|---|---|
| mean_fill | -0.079 |
| blur | -0.518 |
| gaussian_noise | 0.776 |
| patch_shuffle | -0.003 |

_Thresholds used (from thresholds.json): {"primary_metric": "margin_drop", "q2_multiplier": 3.0, "q4_min_fill_strategies": 2, "q5_min_margin_points": 10.0}_

# Section 3.5 Sensitivity Follow-Up: the blur=-0.518 Anomaly

Investigates why evaluate.py's section 3.5 output reported blur's Spearman correlation against constant_fill as -0.518, using only the already-completed run's stored metrics (no retraining or re-auditing). See the module docstring of analyze_section35_sensitivity.py for the three checks below.

## Check 1 -- non-patch region degradation distribution

If the 15 non-patch regions' values are all tiny and statistically indistinguishable from 0 (|mean| within ~2 standard errors of 0), an unstable rank order among them is expected sampling noise, not a sign that the tool is measuring something wrong.

| fill | patch mean | non-patch mean | non-patch std | non-patch min | non-patch max | # non-patch regions distinguishable from 0 (|mean|>2*SE) |
|---|---|---|---|---|---|---|
| constant_fill | 18.9687 | 0.5403 | 0.2233 | 0.2024 | 0.9676 | 15/15 |
| mean_fill | 18.6210 | -0.0947 | 0.2024 | -0.6701 | 0.0138 | 7/15 |
| blur | 12.5238 | -0.1292 | 0.2048 | -0.7336 | -0.0093 | 15/15 |
| gaussian_noise | 4.1500 | 0.1269 | 0.0665 | 0.0331 | 0.2615 | 15/15 |
| patch_shuffle | 17.7160 | -0.4154 | 0.4055 | -1.3558 | 0.2148 | 11/15 |

## Check 2 -- Spearman correlation with vs. without the patch region

The patch region is rank 1 in every fill strategy (Q1/Q4), so including it mechanically pulls every pairwise correlation toward +1. This isolates how stable the ranking of the *other 15* (weak-signal) regions really is.

| fill (vs. constant_fill) | all 16 regions | 15 non-patch regions only |
|---|---|---|
| mean_fill | -0.079 | -0.311 |
| blur | -0.518 | -0.843 |
| gaussian_noise | 0.776 | 0.729 |
| patch_shuffle | -0.003 | -0.218 |

## Check 3 -- is this blur-specific, or shared by any neutral-color fill?

BlurOperator computes a full-frame Gaussian blur before compositing only the target region back in (ssat/core/perturb/operators.py), so at sigma=4 on a 32x32 image, a region near the patch can plausibly pick up some of the patch's color when it is blurred. If that mechanism is what's driving the -0.518 correlation, |degradation| among the 15 non-patch regions should correlate with distance from the patch *specifically for blur*.

| fill | Spearman(|degradation|, distance from patch) |
|---|---|
| constant_fill | -0.284 |
| mean_fill | -0.304 |
| blur | -0.436 |
| gaussian_noise | 0.129 |
| patch_shuffle | -0.518 |

The two regions orthogonally adjacent to the patch cell, across all five fills (a spatial, non-blur-specific mechanism would predict these look similar across fills that produce a similarly neutral replacement color, not just for blur):

| region | constant_fill | mean_fill | blur | gaussian_noise | patch_shuffle |
|---|---|---|---|---|---|
| grid::grid/r0/c1 | 0.9676 | -0.4941 | -0.4957 | 0.1520 | -0.2007 |
| grid::grid/r1/c0 | 0.7208 | -0.6701 | -0.7336 | 0.0741 | -0.7672 |

