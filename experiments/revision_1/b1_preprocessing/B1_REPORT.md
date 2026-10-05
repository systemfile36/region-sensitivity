# B1 report: ImageNet preprocessing confound

Pre-registered settings: `protocol.json` (commit "docs(revision-1):
Pre-register B1 preprocessing confound protocol", before any B1 table was
computed). Additions made after the first area table are listed in
`../deviations.md` D-004. B1-1 to B1-3 use only the stored K=3 runs; B1-4
(P_exact with effective-area-matched controls) uses the A2 K=20 runs on their
2,000-sample subset. Primary metric: `margin_drop`. All B1-1 to B1-3 numbers
are for all 10,000 samples unless stated; the "both runs correct" population
gives the same picture (`ranking_change.csv`, `grade_change.csv`).

## B1-1: effective area (`cell_area.csv`, `within_sample_area_ratio.csv`, `control_area_ratio.csv`, `fig_b1_area.pdf`)

Target areas are identical across the three operators and across the two
models, so the area tables hold for both MobileNetV2 widths.

| Cell type | P_exact mean fraction (x nominal 1/16) | P_cf |
|---|---|---|
| center (4) | 0.108 (1.73x) | 0.0625 (1.00x) |
| edge (8) | 0.058 (0.74-1.10x) | 0.0625 |
| corner (4) | 0.026 (0.41-0.43x) | 0.0625 |

- **The paper's "up to 1.78x" is a lower bound, not an upper bound.** 1.78x
  (= 4096/2304) is the center/corner ratio for a square image. Within a sample,
  the largest/smallest cell area under P_exact is at least 1.78x for every
  sample, with median 4.36x and p95 8.0x (over the 9,885 samples whose cells
  all remain visible). Non-square images lose more of the long side to the
  crop.
- **Some cells are not in the model input at all.** In 115 samples (1.15 %),
  896 border cells have effective area 0 under P_exact. Occluding them leaves
  the model input unchanged apart from resize interpolation at the crop edge
  (|mean margin_drop| <= 0.001), yet 116 of them (mnv2_050; 107 for mnv2_100)
  are graded HIGH and 749 LOW (`zero_area_targets.csv`).
- Under P_cf every cell covers 0.0603-0.0648 of the input (1/16 +- 4 %).
- **Control matching breaks under P_exact.** A control is drawn uniformly in
  the source image, so its visible area rarely matches the target's: only
  14.8 % of control items are within +-5 % of their target's area (center
  51 %, edge 4 %, corner 0.04 %), and 0.69 % of target anchors have all
  three controls within +-5 % (the stored `area_matched` flag, reproduced
  exactly). Corner controls are 3.5-3.6x the target's area at the median. Under
  P_cf 100 % are within +-5 %. At +-10 % / +-20 % the P_exact anchor shares
  are 3.6 % / 16.9 %.

## B1-2: ranking change (`ranking_change.csv`, `cell_rank_change.csv`, `top_region_share.csv`, `fig_b1_rank.pdf`)

| Comparison | 16-cell mean profile Spearman | Per-sample Spearman median (IQR) | Top-1 cell agreement | Top-3 overlap |
|---|---|---|---|---|
| mnv2_050 exact vs crop-free | 0.959 | 0.38 (0.17-0.58) | 33.0 % | 44.4 % |
| mnv2_100 exact vs crop-free | 0.976 | 0.39 (0.18-0.57) | 35.0 % | 44.9 % |
| Yardstick: 050 vs 100, exact | 0.991 | 0.27 (0.04-0.46) | 30.2 % | 40.7 % |
| Yardstick: 050 vs 100, crop-free | 0.968 | 0.27 (0.05-0.47) | 27.3 % | 38.5 % |

- **Dataset level: stable.** The mean 16-cell profile keeps its order
  (Spearman 0.96-0.98), and the four center cells lead under both protocols.
- **Location shift in the expected direction.** The share of samples whose top
  cell is a center cell falls from 63.9 % to 54.0 % (mnv2_050; 64.6 % -> 55.1 %
  for mnv2_100), and corners rise from 4.9 % to 9.4 % (3.9 % -> 9.1 %). Mean
  center-cell `margin_drop` drops from 0.115 to 0.096. Across the 16 cells,
  the cf/exact area ratio correlates with the mean rank change at Spearman
  -0.54 (mnv2_050) and -0.46 (mnv2_100): cells that gain area move up.
- **Sample level: weakly consistent under any change.** The within-image
  ranking agrees only moderately between protocols (median 0.38), but this is
  higher than between the two model widths under the same protocol (median
  0.27). Per-image 16-cell rankings are noisy, and the preprocessing change
  perturbs them less than a model change does.

## B1-3: grade change (`grade_transition_exact_to_cf.csv`, `grade_change.csv`, `by_cell_type.csv`)

| Comparison | Agreement | kappa_w | >= 2 steps | Changes with a sign-consistency change | Changes from `exceeds_control` only |
|---|---|---|---|---|---|
| mnv2_050 exact vs crop-free | 50.3 % | 0.12 | 20.7 % | 44.1 % | 5.6 % |
| mnv2_100 exact vs crop-free | 50.8 % | 0.13 | 20.0 % | 43.9 % | 5.4 % |
| Yardstick 050 vs 100, exact | 47.8 % | 0.08 | 21.3 % | 46.3 % | 5.9 % |
| Yardstick 050 vs 100, crop-free | 48.5 % | 0.07 | 22.1 % | 45.9 % | 5.6 % |

- The grade distribution barely moves (HIGH 14.28 % -> 14.54 %), but half of
  the individual anchors change grade, and almost all of those changes come
  from `sign_consistent` flipping. A model-width change produces the same
  amount of anchor-level change. Anchor-level grades therefore should not be
  read as stable per-image labels across pipeline changes; how much of this is
  run-to-run noise is measured by A2's replicate analysis.
- By cell type, HIGH moves from center to corners (center 21.9 % -> 19.5 %,
  corner 9.5 % -> 11.5 % for mnv2_050), and the cell's area ratio correlates
  with its grade change rate (Spearman 0.78 / 0.64 across the 16 cells).

## B1-4: effective-area-matched controls (`eff_area_controls.csv`)

Run after A2, on its 2,000-sample subset and the two mnv2_050 K=20 runs:

- (a) P_exact with control_index 0-2
- (b) P_exact with all 20 controls
- (c) P_exact_EA: only control items within +-5 % of the target's effective
  area; control anchors average their eligible items; z needs at least 3
  such anchors
- (d) P_cf with all 20 controls

| | all | corner | edge | center |
|---|---|---|---|---|
| Eligible control items | 14.7 % | 0.04 % | 4.0 % | 50.7 % |
| (anchor, condition) with >= 3 eligible controls, out of 20 | 34.1 % | 0.0 % | 18.4 % | 99.5 % |
| (anchor, condition) with no eligible control | 45.2 % | 98.7 % | 41.1 % | 0.2 % |
| HIGH (a) exact K=3 | 14.4 % | 9.2 % | 13.4 % | 21.6 % |
| HIGH (b) exact K=20 | 5.8 % | 1.7 % | 4.1 % | 13.5 % |
| HIGH (c) exact EA | 3.5 % | 0 % | 2.6 % | 8.7 % |
| `exceeds_control` UNAVAILABLE (c) | 52.8 % | 100 % | 55.4 % | 0.2 % |
| HIGH (d) crop-free K=20 | 6.8 % | 3.9 % | 5.8 % | 11.6 % |

- **Effective-area matching is not a workable correction under P_exact.**
  Even with 20 source-plane controls per target, corner targets essentially
  never get an area-matched control, and 53 % of all anchors lose
  `exceeds_control` (they become MODERATE: 21.6 %). Only center cells can be
  area-matched.
- **Where it is possible, it lowers HIGH.** For center cells, matched controls
  change the grade of 6 % of anchors relative to (b) (agreement 93.8 %), and
  HIGH falls from 13.5 % to 8.7 %. The unmatched controls of an enlarged
  center cell are smaller than the target, so their effect is weaker, which
  inflates the target's excess.
- **The protocol gap is not closed by EA.** Agreement with crop-free (d) is
  52.7 % for (b) and 44.9 % for (c), versus 90.8 % between (a) and (b).
  Anchor-level grades depend much more on the preprocessing protocol than on
  K or on area matching.
- **Consequence for the recommendation.** Crop-free region definitions,
  which keep every cell and control at 1/16 of the input, remain the
  recommended setting. Post-hoc effective-area filtering is not a substitute.

## Representative cases (`representative_cases.csv`; figure `results/b1/fig_b1_examples.pdf`, untracked because it embeds ImageNet images)

Selected by the registered rule (mnv2_050, per-sample Spearman closest to p10 /
p50 / p90): ILSVRC2012_val_00000978 (-0.04), 00002204 (0.38), 00001722 (0.71).

## Interpretation

The confound is larger than the paper states: 1.78x is the best case, a
typical 4:3 image has a 4.4x area spread across cells, and 1 % of images have
cells the model never sees, some of which are graded HIGH. It also breaks
control matching (0.7 % of anchors area-matched). Despite this, the
dataset-level region profile and the center-leading pattern are stable across
protocols. Per-image rankings and anchor-level grades change a lot, but no
more than they do between the two models.

## Actions (experiment plan section 15)

- **Claim correction.** Replace "up to 1.78x" with "at least 1.78x (square
  images); median 4.4x and up to 8x at p95 for the 10,000 ImageNet images,
  with 1.15 % of images having border cells entirely outside the crop".
- **Limitation / recommended use.** State that under center-crop
  preprocessing the random-area-match controls are not area matched (0.7 %),
  and recommend crop-free (or effective-area-aware) region definitions when
  regions are compared across locations. A3 follows implementation plan
  section 4.5 (both protocols for every model; architecture comparison on
  crop-free if geometries differ).
- **Uncertainty.** Report dataset-level profiles as robust, and per-image
  rankings and anchor grades as weakly reproducible, citing the model-width
  yardstick.
- **Possible software follow-up (after the revision).** A preflight or
  analysis warning for target regions with zero effective area.

## Paper locations

- ImageNet case study, confound paragraph and Fig. `fig:area-confound`
  caption (manuscript lines ~443-460): correct the 1.78x statement, add the
  zero-area cells and the control-matching share.
- ImageNet results (lines ~487-503) and Discussion: the protocol comparison
  table and the yardstick.
- Limitations: anchor-level grade reproducibility across pipeline changes.

## Deviations from the pre-registration

See `../deviations.md` D-004: zero-area handling and the
`zero_area_targets.csv` table (added after the first area table), two extra
output tables, the image source (three images extracted from the ImageNet zip
and hash-verified), and the example renderer location.
