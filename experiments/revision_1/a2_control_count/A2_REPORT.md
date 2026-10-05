# A2 report: matched-control count ablation

Pre-registered settings: `protocol.json` (commit "docs(revision-1):
Pre-register A2 control-count ablation protocol and K=20 runs", before any
K=20 run started). Three K=20 runs, each compared only with subsets of its own
20 controls (`deviations.md` D-001):

| Run | Samples | Anchors | Perturbed items | `ssat run` time |
|---|---|---|---|---|
| `a2_imagenet_mnv2_050_cf_k20` (primary) | 2,000 (2 per class) | 32,000 | 3,360,000 | 3.31 h (282 items/s) |
| `a2_imagenet_mnv2_050_exact_k20` | same 2,000 | 32,000 | 3,360,000 | 3.56 h (262 items/s) |
| `a2_synthetic_shortcut_k20` | 200 | 3,200 | 1,008,000 | 6.31 h (44 items/s) |

Primary metric `margin_drop`, ddof 0 (the v1.0.0 rule) unless stated.
"Random" rows are means over R=200 random K-subsets per K; the 5-95 % band
across replicates is in the CSVs and is narrow (about +-0.2 pp on agreement).
Prefix subsets (`{0..K-1}`, what `n_samples=K` produces) agree with the
random mean to within 0.24 pp in agreement and HIGH share over all anchors
(0.73 pp in the smallest cell-type group).

## Checks (`nested_parity.json`, `ablation_parity.json`)

- **Nested identity passes for all three runs.** On the A2 samples, every
  baseline target and control item reappears in the K=20 run with the same
  mask parameters, effective area, and seed; controls with `control_index`
  >= the baseline K are new items. Target logits are bitwise equal to the
  baseline everywhere. Nested control logits are bitwise equal for all
  synthetic items and for 77.4 % of ImageNet items (the rest differ by at
  most 7.6e-6, the TF32 effect of D-001; no top-1 change).
- Clean logits are bitwise equal for synthetic but not for ImageNet (max
  difference 0.055 / 0.038, one of the 2,000 crop-free samples changes top-1).
  The clean pass now batches 2,000 samples instead of 10,000, which triggers
  the same TF32 effect. `margin_drop` is measured from the clean margin, so
  A2 values can differ slightly from the baseline's for the same items. This
  is why A2 compares K subsets only within each K=20 run.
- **The offline recomputation reproduces ssat exactly.** The K=20 control
  statistics reproduce the stored `control_comparison` bit for bit
  (96,000 / 96,000 / 16,000 rows), prefix K=20 reproduces
  `reliability.parquet` for every anchor, and prefix K=3 (and K=2 for
  synthetic) reproduces the ssat analysis path run on the filtered items.

## Results

### 1. Control baseline convergence (`control_mean_convergence.csv`)

Median |mu_K - mu_20| / sigma_20 over (anchor, condition):

| K | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| ImageNet crop-free | 0.65 | 0.47 | 0.37 | 0.27 | 0.16 |
| ImageNet exact | 0.64 | 0.46 | 0.37 | 0.27 | 0.16 |
| Synthetic | 0.44 | 0.41 | 0.38 | 0.30 | 0.18 |

The baseline estimate converges at the expected rate. With K=3 its typical
error is about a third of the control spread (p90: 0.87 sigma).

### 2. Location-specific effect (`effect_convergence.csv`, `ddof_effect.csv`)

| K | 2 | 3 | 5 | 10 |
|---|---|---|---|---|
| median \|z_K - z_20\| (crop-free) | 1.15 | 0.63 | 0.38 | 0.19 |
| p90 \|z_K - z_20\| (crop-free) | 10.7 | 3.1 | 1.3 | 0.56 |
| Spearman(max z_K, max z_20) (crop-free) | 0.68 | 0.79 | 0.88 | 0.96 |
| median anchor max z (crop-free) | 2.01 | 1.24 | 0.91 | 0.72 (K=20: 0.64) |

z is much more variable than the mean at small K, because the control std
itself is estimated from K values. It is also biased upward: the median
anchor max z at K=3 is 1.9x its K=20 value.

### 3. Ranking (`ranking_vs_k.csv`)

- Target `margin_drop`, and therefore the region ranking the paper reports,
  does not depend on K.
- The dataset-level 16-cell profile of mean excess (target minus control
  mean) is stable: Spearman with K=20 is 0.977 at K=3 (5th percentile 0.956)
  for crop-free and 0.984 for exact. For synthetic it is 0.93, because 15 of
  the 16 cells have near-zero excess.
- Per-image top-1 cell by excess agrees with K=20 for 72 % (crop-free) and
  73 % (exact) of samples at K=3, and 87-88 % at K=10. In synthetic the
  patch cell is top-1 in every sample at every K.

### 4. Reliability grade (`grade_vs_k.csv`, `transitions_vs_k.csv`, `fig_a2_convergence.pdf`, `fig_a2_high_share.pdf`)

| | K=2 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| Agreement with K=20, crop-free | 87.1 % | 91.4 % | 94.9 % | 97.6 % | – |
| Agreement with K=20, exact | 85.6 % | 90.9 % | 94.9 % | 97.8 % | – |
| Agreement with K=20, synthetic | 97.9 % | 99.1 % | 99.7 % | 99.9 % | – |
| HIGH share, crop-free | 19.1 % | 14.5 % | 10.7 % | 8.0 % | 6.8 % |
| HIGH share, exact | 19.9 % | 14.3 % | 10.0 % | 7.1 % | 5.8 % |
| HIGH share, synthetic | 8.4 % | 7.2 % | 6.6 % | 6.4 % | 6.3 % |

- **K=3 roughly doubles the HIGH share on ImageNet.** HIGH is 14.5 %
  (crop-free) and 14.3 % (exact) at K=3, in line with the paper's 10,000-sample
  K=3 figures, but 6.8 % and 5.8 % at K=20. The HIGH share is still falling
  between K=10 and K=20, so K=20 is itself not converged.
- **The disagreements are almost all K=3 HIGH -> K=20 LOW.** For crop-free,
  8.2 % of anchors move HIGH -> LOW and 0.4 % move LOW -> HIGH. Only 44 %
  (crop-free) and 38 % (exact) of K=3 HIGH anchors remain HIGH at K=20.
  K=3 misses few K=20 HIGH anchors (6 % in crop-free).
- **UNRELIABLE is unaffected** (61.6 % crop-free, 58.6 % exact at every K),
  because `sign_consistent` does not use the controls. MODERATE stays 0.
- **By cell type** (K=3 vs K=20, crop-free): center 19.8 % -> 11.6 %,
  edge 13.5 % -> 5.8 %, corner 11.2 % -> 3.9 %. Agreement is 90.5-92.2 % in
  every cell type.
- **ddof 1** lowers the K=3 HIGH share only to 12.4 % (crop-free), so the
  small-sample std bias accounts for about a quarter of the inflation.
- **Synthetic ground truth** (`synthetic_gt_vs_k.csv`): the patch cell is
  HIGH in 100 % of samples at every K. HIGH on the 15 non-patch cells falls
  from 1.0 % at K=3 to 0.03 % at K=20 (random subsets). At K=3 these are the
  only disagreements.

**Why z > 2 means different things at different K.** This is a derivation
under an idealized null, not a measurement. If the target and the K controls
are i.i.d. normal, the ddof-0 z maps onto a prediction t statistic
t = z * sqrt((K-1)/(K+1)) with K-1 degrees of freedom. The chance that one
condition passes z > 2 is then 14.7 % at K=3, 8.9 % at K=5, 5.2 % at K=10,
and 3.6 % at K=20 (Monte Carlo, 2M draws). With the max over three
conditions, treating them as independent, it is 38 %, 24 %, 15 %, and 10 %.
The fixed threshold of 2 is therefore far more permissive at K=3. This is
consistent with the measured HIGH inflation, which is concentrated in
downgrades from HIGH.

### 5. Seed variance (`replicate_variability.csv`, `fig_a2_flip.pdf`)

"Flip probability" is the probability that two random K-subsets give an
anchor different grades.

| K | 2 | 3 | 5 | 10 |
|---|---|---|---|---|
| Mean flip probability, crop-free | 9.4 % | 8.0 % | 5.8 % | 3.1 % |
| Share of anchors that can flip, crop-free | 35 % | 32 % | 25 % | 14 % |
| Mean flip probability, exact | 11.1 % | 8.9 % | 5.9 % | 3.0 % |
| Mean flip probability, synthetic | 2.2 % | 1.1 % | 0.4 % | 0.07 % |

With K=3, rerunning the audit with another control seed would change about
8-9 % of ImageNet anchor grades, and a third of anchors are exposed to this.
The dataset-level HIGH share, by contrast, is stable across seeds (replicate
std 0.1 pp): the bias is systematic, not noise.

### 6. Marginal gain and cost (`marginal_gain.csv`)

Run time is linear in 1 + K. The times below use the measured crop-free
throughput (282 items/s) for 2,000 samples.

| K | Items | `ssat run` time | Agreement | Gain per added control |
|---|---|---|---|---|
| 3 | 0.64 M | 0.63 h | 91.4 % | – |
| 5 | 0.96 M | 0.95 h | 94.9 % | +1.7 pp |
| 10 | 1.76 M | 1.73 h | 97.6 % | +0.54 pp |
| 20 | 3.36 M | 3.31 h | 100 % (ref) | +0.24 pp |

Most of the gain has been collected by K=10. Its cost is 2.75x that of K=3.

## Interpretation (experiment plan section 6.6: "K=3 is unstable")

- **What holds at K=3.** Dataset-level region profiles and the region
  ranking (which does not use controls), the UNRELIABLE share, and the
  synthetic shortcut detection (100 % HIGH on the patch at every K).
- **What does not hold at K=3.** The HIGH share, and with it
  `exceeds_control`. At K=3 it is about twice the K=20 value on ImageNet,
  about 8-9 % of anchor grades depend on the control seed, and the fixed
  z > 2 threshold has a nominal false-pass rate about 4x that at K=20.
- **Recommended wording and actions** (the `ssat` default is not changed
  during the revision; implementation plan section 2.1):
  1. Report the ImageNet HIGH shares as K-dependent: "14.3 % with K=3
     controls, 5.8 % with K=20 (2,000-sample subset)". Frame the K=3 HIGH
     share as an upper bound, and label HIGH at K=3 as "candidate"
     rather than confirmed.
  2. Recommend K >= 10 when anchor-level grades are used: at least 97.6 %
     agreement with K=20, mean flip probability 3 %, and 2.75x the cost of
     K=3. K=3 remains acceptable for dataset-level profiles and screening.
  3. Limitation: z > 2 is not calibrated across K. A K-aware threshold
     (for example the t-based prediction interval above) or ddof 1 plus a
     larger K is future work. ddof 1 alone removes only about a quarter of
     the inflation.
  4. Candidate software follow-up after the revision: raise the default
     `n_samples`, or warn when `exceeds_control` rests on fewer than 10
     controls.

## Paper locations

- Reliability numbers (ImageNet 14.28 / 26.69 / 59.03): add the K
  dependence, citing `grade_vs_k.csv` (K=3 vs K=20 on the A2 subset).
- Method section on matched controls: state K, the z threshold's K
  dependence, and the recommended K.
- Limitations: control-count sensitivity of `exceeds_control`, and the
  per-anchor seed variance.

## Deviations

None beyond `deviations.md` D-005 (how the synthetic run was launched). The
cf and exact ablations were first run into `results/a2/preview*/` as soon as
each run finished, with the same committed code. All committed summaries
were regenerated from a clean tree after the third run.
