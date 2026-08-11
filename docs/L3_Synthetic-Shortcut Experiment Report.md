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
