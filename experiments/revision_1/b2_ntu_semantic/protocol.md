# B2 protocol: NTU RGB+D semantic validation (Route A)

Registered on 2026-09-30, before any annotation, following implementation
plan section 8 and experiment plan section 9. Machine-readable version:
`protocol.json`. Annotator instructions: `annotation/README.md`.

## Question and limits

Do the body parts that SSAT scores as most sensitive for an action class
agree with the parts that people, given the action name and original
NTU RGB+D clips of the class, consider relevant to that action? The human
ratings are an external semantic reference, not causal ground truth.
Agreement does not show that SSAT is "correct", and disagreement does not
show that it is wrong (experiment plan section 9.7).

## Data (no new inference)

- Stored case-study runs:
  - `results/ntu60_tsm_exact` (primary);
  - `results/ntu60_tsm_crop_free` (robustness).
- Input: 1,200 NTU-60 x-sub test videos, 20 per class. Model: TSM-R50 with
  8 frames.
- Scores use `metrics/spatial_profile.parquet` (margin drop, averaged over
  mean fill, blur, and Gaussian noise with 3 seeds), `sample_metrics`
  (label, clean correctness), and `region_metrics` (effective area). Input
  SHA-256 values are in `protocol.json`.
- `check_inputs.py` checks the structure (1,200 x 10 region pairs, labels,
  group mapping) without reading any score. It found zero effective area
  for 123 (exact) and 46 (crop-free) of the 12,000 pairs, in 16 and 9
  samples respectively. These pairs are kept in the primary score, as in
  the report, and dropped from the area-adjusted score.

## Ontology

- **Five atomic groups**, left and right merged: `head`, `torso`, `arms`,
  `hands`, `legs`. They map the config's `semantic_group` values.
- **Auxiliary groups.** `upper_body` and `lower_body` are supersets with
  large areas. Their raw scores are reported only to show the bias.
- **Region definition.** Each SSAT region is the per-frame bounding box of
  its joints, scaled by 1.15 (`scripts/dataset_prep/ntu_rgb_d.py`
  `BODY_PARTS`). The arm box contains the hand joints.
- **Two-person actions.** For A050-A060, SSAT's regions come from one body
  per video, the one with the most valid joints. This may not be the actor.

## Annotation

- **Classes:** all 60 classes, so there is no class selection. The official
  names are in `annotation/classes.csv`; label id = action number - 1.
- **Ratings:** 0 (not involved), 1 (supporting), 2 (core: hard to
  recognize the action without seeing the part), one per class x group
  cell. Definitions and rules are in `annotation/README.md`.
- **Blind conditions.**
  - Annotators see the action name and may watch any original
    (unperturbed) NTU RGB+D clips of the class. They see no SSAT outputs,
    no paper figures, and no other annotator's sheet.
  - Each sheet is committed before any SSAT score is computed, and the
    commit time is the record.
  - `ssat_part_scores.py` and `evaluate_alignment.py` refuse to run unless
    every sheet is complete and committed unchanged at HEAD.
- **Annotators.** Two annotators are planned. If a second annotator is not
  available, the analysis runs with `--annotators A`, and single-annotator
  agreement is reported as a limitation (implementation plan section 12,
  item 3).
- **Prior exposure.** Each annotator records in `annotation/annotators.csv`
  whether they had seen SSAT's NTU results before. An author who has seen
  the paper's NTU figures is not blind to them; this is reported.
- **Consensus.** The mean rating over annotators. Relevant: consensus >= 1.
  Primary: consensus = 2.

## SSAT scores (`ssat_part_scores.py`)

- **Per-sample group value.** Margin-drop degradation per (sample,
  region), averaged over the group's regions (left and right) within the
  sample.
- **Class matrix.** `S[c, g]` is the mean of the per-sample values over the
  class's samples. This is the report's class x semantic-group matrix
  (checked by a unit test against `ssat.report.assembler`).
- **Primary score:** `L[c, g] = (S[c, g] - mean_c S[., g]) / sd_c S[., g]`,
  with the population sd (ddof 0) across the 60 classes. It removes each
  group's constant bias, for example size.
- **Robustness scores:**
  - raw `S`;
  - area-adjusted lift: each (sample, atomic part) value is replaced by the
    residual of `log1p(max(d, 0))` on `log(effective_area_px)`, from one
    pooled OLS fit over all such pairs with positive area; then the same
    aggregation and z-score are applied.

## Metrics (`evaluate_alignment.py`)

| Kind | Definition |
|---|---|
| Annotator agreement | Linear weighted Cohen's kappa (0/1/2) and raw agreement, per group (60 cells) and overall (300 cells); agreement on relevant (>= 1). |
| **Primary** | Per-class AUROC of `L` for relevant vs non-relevant groups (Mann-Whitney, ties 1/2), over classes with both kinds. Reported: the mean over classes, a 95 % percentile CI from 10,000 class bootstrap resamples, and a one-sided permutation p-value from 10,000 random reassignments of annotation rows to classes, `(1 + #null >= observed) / (1 + 10,000)`. Classes with all or no relevant groups are excluded and counted. |
| Secondary | Top-1 hit rate: the share of classes with a primary group whose highest-`L` group (ties by the order head, torso, arms, hands, legs) is primary. Chance: the mean share of primary groups. The same permutations give a p-value. Also the mean per-class Spearman between the consensus ratings and `L` (classes with constant ratings excluded). |
| Robustness | The primary and secondary metrics for: crop-free run; clean-correct samples only; raw `S`; area-adjusted lift; single-person classes only (A001-A049); each annotator alone. |

- **Seed:** 20260930 for the bootstrap and the permutations.
- **Representative cases:** the top 3 and bottom 3 classes by AUROC (ties
  by label id), shown in `fig_b2_examples.pdf` as the lift heatmap with the
  consensus ratings.

## Outputs (`summary/`)

- `class_group_scores.csv`;
- `consensus.csv`, `annotator_agreement.csv`;
- `alignment_by_class.csv`, `alignment_summary.json`, `robustness.csv`,
  `permutation_null.csv`;
- `fig_b2_alignment.pdf`, `fig_b2_examples.pdf`;
- `B2_REPORT.md`, which includes the limits above.

## Known limitations (reported regardless of the result)

- **No matched controls.** NTU has no matched controls, so SSAT scores
  include area effects. The class z-score removes constant per-group bias
  but not per-sample size differences. The area-adjusted score is a partial
  check only.
- **Two-person classes.** SSAT regions may belong to the non-acting person
  in these classes; this is checked in the single-person robustness
  analysis.
- **The ratings are coarse.** They are made from the action name and
  example clips and describe typical relevance for the action, not a rating
  of each of the 20 audited videos per class.
