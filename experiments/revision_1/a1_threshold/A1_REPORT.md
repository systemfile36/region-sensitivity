# A1 report: reliability grade threshold sensitivity

Pre-registered settings: `protocol.json` (commit `cb7ef30`, before any sweep
ran). All grades were recomputed offline from the stored analysis and metrics
stores of the eight Phase 0 baseline runs; no model was re-run. Primary metric:
`margin_drop`. Numbers below are shares of anchors (sample x region); tables are
in `summary/`.

## Parity gate

The vectorized engine (`common/grade_engine.py`) reproduces the stored
`reliability.parquet` with zero mismatches in all four flags and the grade, for
every anchor of all eight runs (`summary/parity.json`, 670,400 anchors). The
recomputed 95 % bootstrap intervals are bitwise equal to `intervals.parquet`:
`build_features.py` replays the analysis's shared random stream across all nine
metrics (`results/a1/features/*.json`). The unit tests
(`tests/unit/test_rev1_grade_engine.py`) check the same parity against
`ssat analyze` on small dumps, including zero control std, one control, no
control, and single-operator anchors, and check the 90/99/99.9 % intervals
against `compute_intervals` called with those percentiles.

## How each threshold enters the grade

| Parameter | Effect in these runs | Why |
|---|---|---|
| tau_z (z threshold) | Moves anchors between HIGH and LOW only | Every other core flag is TRUE for sign-consistent anchors, so `exceeds_control` alone separates HIGH from LOW. MODERATE never occurs. |
| alpha (sign deadband) | Largest effect; moves anchors into UNRELIABLE | Under the v1.0.0 rule a zero sign is a separate sign, so (+,+,0) becomes inconsistent. |
| m (multi-strategy count) | **No effect, by construction** | `sign_consistent` TRUE already means every operator has the same sign, so the dominant count equals the operator count (3 or 5) and m <= that count always passes. m only matters for anchors with fewer operators than m, which these runs do not have. |
| CI level | None on ImageNet and NTU; 0.75 % of anchors on synthetic M_shortcut | ImageNet region CIs sit far from zero (`ci_excludes_zero` TRUE for 100 % at every level). NTU region keys are per clip (one sample per region), so the bootstrap CI collapses to the point value and the level cannot matter. |
| ddof | 2.1 % (ImageNet), 2.2 % (M_normal), 0.3 % (M_shortcut), HIGH -> LOW only | sqrt(3/2) = 1.22 shrinks every K=3 z. |
| beta (std floor) | <= 1.3 % | Only anchors with near-zero control std are affected. |
| `seed_cv_threshold`, `area_match_tolerance` | None | Neither flag enters the grade. `seed_cv_flag_rates.csv` reports the `seed_stable` TRUE share (ImageNet 6-7 % at 0.1, 18-20 % at 0.2, 47-50 % at 0.5); area-tolerance rates are reported by B1. |

## Results

**One-at-a-time** (`agreement.csv`, `sensitivity_ranking.csv`, `fig_a1_oat.pdf`).
Agreement with the default grade over the registered range:

| Parameter (range) | ImageNet (4 runs) | Synthetic M_shortcut | Synthetic M_normal | NTU60 (2 runs) |
|---|---|---|---|---|
| tau_z (1.0-4.0) | 93.0-93.9 % at the ends; 97.0-97.7 % at 1.5 / 2.5 | 98.2 % | 95.6 % | n/a (no controls) |
| alpha (0.01-0.25) | 99.2-99.3 % at 0.01; 84.8-85.1 % at 0.25 | 92.8 % at 0.01; 85.1 % at 0.25 | 99.3 % / 88.5 % | 98.7-98.8 % / 83.0-83.3 % |
| m | 100 % | 100 % | 100 % | 100 % |
| CI level | 100 % | 99.3 % | 100 % | 100 % |
| ddof = 1 | 97.9 % | 99.7 % | 97.8 % | n/a |
| beta (0.10-0.25) | 98.7-99.9 % | 99.0 % | 99.8 % | n/a |

alpha is the most sensitive parameter in every run; tau_z is second wherever
controls exist. Every tau_z change is a two-step change (HIGH <-> LOW), since
MODERATE is unreachable when all other flags are available.

**Presets** (`table_a1.csv`/`.tex`, `fig_a1_transitions.pdf`).

| Run | HIGH default | Lenient: agr. / kappa_w / HIGH | Conservative: agr. / kappa_w / HIGH | Conservative HIGH -> UNRELIABLE |
|---|---|---|---|---|
| ImageNet (4 runs) | 13.9-14.5 % | 97.0-97.3 % / 0.94 / 16.9-17.3 % | 90.5-90.7 % / 0.80-0.81 / 7.5-7.8 % | 0.83-1.00 % |
| Synthetic M_shortcut | 8.1 % | 98.7 % / 0.96 / 9.4 % | 89.0 % / 0.79 / 6.7 % | 0.66 % |
| Synthetic M_normal | 23.8 % | 98.7 % / 0.98 / 25.2 % | 92.9 % / 0.87 / 17.9 % | 1.44 % |
| NTU60 (2 runs) | 0 % | 100 % / 1.00 / 0 % | 94.8-94.9 % / 0.89 / 0 % | 0 % |

The lenient preset only adds HIGH anchors (LOW -> HIGH, 2.7-3.0 % on ImageNet).
The conservative preset roughly halves the HIGH share on ImageNet (e.g. mnv2_050
exact: 53.5 % of default-HIGH anchors stay HIGH, 40.5 % become LOW, 5.9 % become
UNRELIABLE), and moves 10.7 % of default-LOW anchors to UNRELIABLE through the
sign deadband. At most 0.03 % of anchors move from UNRELIABLE to HIGH under
either preset (M_shortcut; 2 of 160,000 on mnv2_050 exact). The paper's ranking of grades is therefore
preserved; the HIGH share itself is threshold-dependent (ImageNet 7.5-17.3 %
across the presets, 13.9-14.5 % at the default).

**Is the default at a steep point?** (`z_curve.csv`, `boundary_mass.csv`,
`fig_a1_z_curve.pdf`). On ImageNet, 7.2-7.6 % of anchors have max z within
+-0.25 of 2.0 and 14.7-15.4 % within +-0.5. The HIGH share falls smoothly with
tau_z (28-30 % at 0, 13.9-14.5 % at 2, 4.4-4.6 % at 6); there is no cliff at
2.0. Around the default the HIGH share changes by about 0.5 percentage points
per 0.1 in tau_z (e.g. mnv2_050 exact: 14.81 % at 1.9, 13.76 % at 2.1); the
default lies just above the median of the max-z distribution (1.09-1.25).

**Synthetic ground truth** (`synthetic_gt.csv`). In M_shortcut, the patch cell
(r0/c0) is HIGH for 100 % of samples at every tau_z in 0-6, every alpha, and
both presets. The HIGH share of the other 15 cells is 2.0 % at the default,
5.9 % at tau_z = 0, 1.1 % at tau_z = 6, 3.4 % lenient, and 0.5 % conservative.
In M_normal, which has no shortcut, the patch cell is HIGH for 13 % of samples
and the other cells for 24.6 %; these are not false positives in a strict
sense, because the normal model relies on real image content.

**Zero-abstain alternative** (`alt:zero_abstain:*` rows, design alternative
only). Treating zero signs as abstentions reverses alpha's effect: anchors move
out of UNRELIABLE (UNRELIABLE 59.0 % -> 41.8 % on mnv2_050 exact at
alpha = 0.25) and the HIGH share rises slightly (14.3 % -> 16.3 %).

## Interpretation

1. The grade is stable under tau_z, ddof, beta, and CI level changes inside the
   registered ranges (agreement >= 92.9 %, weighted kappa >= 0.84); the changes
   are all HIGH <-> LOW.
2. The sign rule is the most sensitive part: a small deadband (alpha = 0.05)
   already moves 3.5-3.7 % of ImageNet anchors to UNRELIABLE, and about 15 %
   of ImageNet grades change at alpha = 0.25.
   This follows from treating an exact zero as its own sign (plan section 3.6).
3. Two of the four flags carry no information in these runs: `multi_strategy`
   is implied by `sign_consistent` whenever all operators are present, and
   `ci_excludes_zero` is TRUE for every ImageNet region at every level. The
   ImageNet grade is effectively decided by `sign_consistent` and
   `exceeds_control` (plan section 1.4). This is a property of the rule, not
   evidence of robustness.
4. NTU60 has no matched controls, so no anchor can reach HIGH; its grade is
   decided by the sign rule, and its CI flag is degenerate (one clip per
   region key).

## Actions (experiment plan section 15)

- **Claim scope / uncertainty.** Report the HIGH share with its threshold
  range (ImageNet 7.5-17.3 % across the lenient and conservative presets) next
  to the default value, and state that every tau_z change moves anchors between
  HIGH and LOW only.
- **Limitation.** State that `multi_strategy` is redundant with
  `sign_consistent` when every operator is evaluated, that the region CI is
  degenerate for per-sample region definitions (NTU), and that the sign rule
  treats exact zeros as a separate sign.
- **Recommended use.** Recommend reporting the continuous `z_vs_control` next
  to the grade. The default configuration is not changed (implementation plan
  section 12, item 5, is decided after A2).

## Paper locations

- Section "Functionalities" (grade definition, manuscript lines ~255-263):
  add the threshold-sensitivity summary and the flag redundancy note.
- ImageNet example (lines ~487-503): add the preset range next to
  14.28/26.69/59.03 and cite `fig_a1_z_curve.pdf`.
- Discussion (lines ~563-572) and Limitations (lines ~648-653): the sign-rule
  sensitivity, the NTU CI degeneracy, and the dependence on K (A2).

## Deviations from the pre-registration

None. The NTU CI degeneracy and the structural inertness of m were found while
running the registered sweep; they are reported, not acted on. The
zero-abstain alternative (optional, plan section 3.7) was run with the
registered alpha values.
