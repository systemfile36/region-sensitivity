# L3 Synthetic-Shortcut Experiment Report

> 2026-08-13 (crop-free re-run)
> lr=0.01, warmup-epoch=10, weight-decay=5e-4, epochs=40, batch-size=128, momentum=0.9
> preprocessing: `Resize([224,224])` only (no CenterCrop) — see note below.

**This document supersedes `deprecated_L3_Synthetic-Shortcut Experiment Report.md`.**
The original run audited squeezenet1_0's ImageNet preset preprocessing
(Resize(256)→CenterCrop(224)), which was hardcoded and not configurable at
the time. CenterCrop made the model-space *effective* area of the 16
nominally-equal grid cells depend on position (corner cells 2304px, edge
cells 3072px, center cells 4096px — a 1.78x spread), which turned out to
correlate strongly with several fill strategies' per-region degradation.
This run instead uses `Resize([224,224])` directly (no crop), which the
adapter now supports via a configurable `preprocessing` field
(`ssat/core/adapter/torchvision_adapter.py`'s `preprocessing_ops`), and
`M_shortcut`/`M_normal` were retrained from scratch under the exact same
pipeline (`experiments/synthetic_shortcut/common.py`'s
`build_crop_free_transform`, which drives the identical
`DeclarativePreprocessor` code the adapter uses — not a hand-written
approximation of it). Under this preprocessing all 16 grid cells land at
exactly 3136px in model space (confirmed below), eliminating the area
confound by construction.

## Q1-Q5 (pre-registered, docs/STAGE9_SYNTHETIC_SHORTCUT_DESIGN_v1.md section 5)

| Question | Result | Detail |
|---|---|---|
| Q1. identifies patch region | **PASS** | patch_region_rank=1 |
| Q2. separated from baseline | **PASS** | multiplier=175.65, threshold=3.0 |
| Q3. distinguishes M_normal | **PASS** | patch_region_rank_in_m_normal=16 |
| Q4. robust to fill strategy | **PASS** | reproduced_in=['constant_fill', 'mean_fill', 'blur', 'gaussian_noise', 'patch_shuffle'], min_required=2 |
| Q5. predicts generalization gap | **PASS** | shortcut_accuracy_drop_points=89.55, normal_accuracy_drop_points=-6.20, margin_points=95.75, threshold_points=10.0 |

**B auxiliary control** (not a pass/fail criterion): patch region rank under random placement = 2. Not part of the pre-registered Q1-Q5 verdict.

**All five Q1-Q5 criteria still pass, unchanged from the original run's verdict.** Q2's multiplier is even larger (175.65 vs. the original 35.11) — removing the crop confound did not weaken the patch-detection signal; if anything the corner-cell crop was previously *trimming away* some of the patch region's own pixels along with the confound, and Resize-only preserves the full patch.

## Section 3.5 -- fill-strategy sensitivity (Spearman correlation of region ranking against constant_fill, all 16 regions)

| fill strategy | Spearman correlation |
|---|---|
| mean_fill | 0.085 |
| blur | 0.179 |
| gaussian_noise | 0.747 |
| patch_shuffle | 0.126 |

_Thresholds used (from thresholds.json): {"primary_metric": "margin_drop", "q2_multiplier": 3.0, "q4_min_fill_strategies": 2, "q5_min_margin_points": 10.0}_

**Every fill strategy's correlation with constant_fill is now positive** — the original run reported blur at -0.518 (a strong negative outlier). See the sign-group re-examination below for the 15-non-patch-region breakdown and why this changed.

## Sign-Group Premise Re-examination (docs/CONTROL_STABILITY_DESIGN_v1.md section 0)

The control/stability analysis module's design was directly motivated by
this experiment's original observation that non-patch regions split into
two sign groups by fill strategy: {constant_fill, gaussian_noise} positive
vs. {mean_fill, blur, patch_shuffle} negative, with the split's magnitude
(-0.843 to +0.729) treated as evidence of a real, substantive
cross-strategy disagreement. This re-run tests whether that split survives
removing the CenterCrop area confound.

### effective_area_px distribution, 15 non-patch regions

| run | distinct effective_area_px values |
|---|---|
| cropped (original) | {2304, 3072, 4096} |
| crop-free (this run) | {3136} (uniform) |

Confirms the confound is structurally eliminated, not just reduced.

### Spearman(degradation, effective_area_px) per fill strategy

| fill | cropped | crop-free |
|---|---|---|
| constant_fill | 0.790 | n/a (zero-variance area) |
| mean_fill | 0.026 | n/a |
| blur | -0.676 | n/a |
| gaussian_noise | 0.822 | n/a |
| patch_shuffle | -0.413 | n/a |

Under the cropped run, degradation correlated strongly with effective area
for constant_fill, blur, and gaussian_noise in particular — closely
tracking the sign and rough magnitude of each strategy's group membership
below. Under crop-free, this correlation is structurally undefined (no
variance left in the independent variable), removing the mechanism
entirely.

### spearman_excl_top1 vs constant_fill (15 non-patch regions)

| fill | cropped | crop-free |
|---|---|---|
| mean_fill | -0.311 | -0.111 |
| blur | **-0.843** | **+0.004** |
| gaussian_noise | +0.729 | +0.693 |
| patch_shuffle | -0.218 | -0.061 |

**Finding: the sign-group split does not reproduce under crop-free preprocessing.**

- **gaussian_noise's strong positive correlation is robust** (0.729 → 0.693,
  a ~5% change) — this appears to be a genuine, area-independent effect,
  not an artifact.
- **blur's strong negative correlation was almost entirely a crop artifact**
  (-0.843 → +0.004): it crosses sign and lands essentially at zero once the
  area confound is removed.
- **mean_fill and patch_shuffle retain the same sign but shrink to roughly
  a third of their original magnitude** (-0.311→-0.111, -0.218→-0.061) — a
  much weaker residual effect remains; whether it is a genuine (if small)
  disagreement or residual noise is not established by this analysis alone.

Empirical clustering (the same procedure `ssat.analysis`'s A4 strategy
profiler uses) confirms this structurally: under crop-free the five
strategies form **three** clusters (`{blur, mean_fill}`, `{constant_fill,
gaussian_noise}`, `{patch_shuffle}` alone), not the original clean two-way
split. Re-running `analyze_control_stability.py --results-dir
results_crop_free` reports Q3/Q4/Q5 as `FAIL` against the *original run's*
pre-registered numbers by design (that script checks reproduction of the
original hand-derived values, not crop-free correctness) — those mismatches
are this section's finding, not a defect.

**Implication for docs/CONTROL_STABILITY_DESIGN_v1.md section 0**: the
module's founding observation was overstated by the area confound — the
dramatic, cleanly-binary two-group split does not hold once the confound is
removed. The module's core design principle ("reveal disagreement between
conditions rather than averaging it away") remains sound in a weaker form:
strategies still disagree in sign and magnitude on non-patch regions, just
less dramatically than originally measured, and gaussian_noise's
disagreement with constant_fill in particular looks genuine rather than
artifactual. See that document's section 0 addendum for the full
implication. This does not affect Q1/Q2/Q4's patch-region conclusions,
which were already robust to the confound (see Q2 above).

## Section 3.5 Sensitivity Follow-Up (crop-free)

Re-running `analyze_section35_sensitivity.py --results-dir results_crop_free`
(the same three checks the original blur=-0.518 investigation used):

### Check 1 -- non-patch region degradation distribution

| fill | patch mean | non-patch mean | non-patch std | non-patch min | non-patch max | # non-patch regions distinguishable from 0 (\|mean\|>2*SE) |
|---|---|---|---|---|---|---|
| constant_fill | 14.0303 | 0.0799 | 0.2317 | -0.0259 | 0.6683 | 6/15 |
| mean_fill | 13.9914 | -0.0804 | 0.2081 | -0.6323 | 0.0095 | 4/15 |
| blur | 8.8046 | -0.0666 | 0.1690 | -0.5314 | 0.0054 | 3/15 |
| gaussian_noise | 7.3625 | 0.0186 | 0.0320 | -0.0048 | 0.1028 | 6/15 |
| patch_shuffle | 13.3310 | -0.1406 | 0.2383 | -0.5356 | 0.3131 | 9/15 |

### Check 2 -- Spearman correlation with vs. without the patch region

| fill (vs. constant_fill) | all 16 regions | 15 non-patch regions only |
|---|---|---|
| mean_fill | 0.085 | -0.111 |
| blur | 0.179 | 0.004 |
| gaussian_noise | 0.747 | 0.693 |
| patch_shuffle | 0.126 | -0.061 |

### Check 3 -- is this blur-specific, or shared by any neutral-color fill?

| fill | Spearman(\|degradation\|, distance from patch) |
|---|---|
| constant_fill | -0.247 |
| mean_fill | -0.384 |
| blur | -0.073 |
| gaussian_noise | -0.333 |
| patch_shuffle | -0.722 |

The two regions orthogonally adjacent to the patch cell, across all five fills:

| region | constant_fill | mean_fill | blur | gaussian_noise | patch_shuffle |
|---|---|---|---|---|---|
| grid::grid/r0/c1 | 0.6683 | -0.5499 | -0.4284 | 0.0765 | -0.2274 |
| grid::grid/r1/c0 | 0.6315 | -0.6323 | -0.5314 | 0.1028 | -0.5356 |

blur's distance correlation (-0.073) is now the *weakest* of the five
strategies under crop-free, consistent with the sign-group finding above
that blur's original anomaly was substantially area-driven rather than a
blur-specific spatial bleed effect.
