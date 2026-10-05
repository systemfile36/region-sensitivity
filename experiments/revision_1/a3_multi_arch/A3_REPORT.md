# A3 report: multi-architecture comparison

Pre-registered settings: `protocol.json` (commit "docs(revision-1):
Pre-register A3 multi-architecture protocol, configs, and model check"). The
model check `summary/model_selection.json` was committed in "docs(revision-1):
Record A3 model selection check" before any A3 run. The four new runs use the
MobileNetV2 case-study configs unchanged apart from the model: the same 10,000
ImageNet validation images, the 4x4 grid, three operators (five operator-seed
conditions), and K=3 matched controls. The MobileNetV2 runs are the stored
baselines and were not rerun. Primary metric: `margin_drop`.

| Model | timm checkpoint | Params | Clean top-1, exact / crop-free | `ssat run` time, exact / crop-free |
|---|---|---|---|---|
| MobileNetV2-0.5 | `mobilenetv2_050.lamb_in1k` | 2.0 M | 66.1 % / 62.2 % | (baseline) |
| MobileNetV2-1.0 | `mobilenetv2_100.ra_in1k` | 3.5 M | 73.0 % / 70.9 % | (baseline) |
| ConvNeXt-T | `convnext_tiny.fb_in1k` | 28.6 M | 82.0 % / 81.7 % | 3.56 h / 3.58 h (250 / 248 items/s) |
| DeiT-S | `deit_small_patch16_224.fb_in1k` | 22.1 M | 79.7 % / 79.0 % | 3.42 h / 3.34 h (260 / 266 items/s) |

`ssat metrics` took 25-28 min and `ssat analyze` 19-20 min per run. Each dump
is 13 GB.

## Model selection and comparison rule (`model_selection.json`, `deviations.md` D-006)

- **ConvNeXt-T passes all three selection conditions.** Its model-space cell
  areas equal mobilenetv2_050's for every sample and cell under both protocols.
- **DeiT-S fails condition 3**, because its `crop_pct` is 0.9 rather than
  0.875. The other two transformer candidates (ViT-S augreg, Swin-T) also use
  0.9, so no transformer candidate passes.
- Under exact preprocessing, DeiT-S's cells differ in area from
  mobilenetv2_050's in 99.5 % of sample cells: center 0.94x, edge 1.03x,
  corner 1.14x (`deit_exact_area.csv`). The larger crop also narrows the
  within-image area spread (median max/min 3.5x vs 4.4x), and 76 samples
  (vs 115) have a cell outside the crop.
- **Rule applied (implementation plan sections 4.5, 6.1):**
  - All four models are compared on the crop-free runs, where the geometry is
    identical.
  - The exact comparison covers MobileNetV2-0.5, MobileNetV2-1.0, and
    ConvNeXt-T.
  - DeiT-S exact is reported per model only.

## Checks (`run_checks.json`)

- **Workflow.** All four runs completed every item (3,210,000 per run,
  including 10,000 clean) with no load, prepare, predict, or OOM failures.
  They ran at batch 128 and fp32 with the unchanged config.
- **Same images, items, and masks as the baselines.** Each run's clean images
  match the baseline's `content_hash` for all 10,000 samples. Its 3.2 M
  item_ids, `seed_used`, and `region_params_json` equal the mobilenetv2_050
  run of the same protocol.
- **Effective areas.** They equal mobilenetv2_050's for every item wherever
  the geometry is shared. For DeiT-S exact, every target area equals the area
  recomputed with DeiT-S's own transform.
- **Weights and preprocessing** equal `model_selection.json` (weight
  SHA-256 and preprocessing fingerprint).

## Results (population "all" unless stated; the correct-only subsets give the same picture)

### 1. Spatial sensitivity profile (`region_profile.csv`, `common_patterns.csv`, `top_region_share.csv`, `fig_a3_profiles.pdf`)

**Common pattern.** In all eight runs, the four center cells hold the top
four ranks of mean `margin_drop` within the model, the eight edge cells come
next, and the four corners rank last. The mean rank by cell type is exactly
2.5 / 8.5 / 14.5 in every run. The paper's center-dominance conclusion is
therefore not specific to MobileNetV2.

| Share of samples whose top cell is center / edge / corner | MNv2-0.5 | MNv2-1.0 | ConvNeXt-T | DeiT-S |
|---|---|---|---|---|
| crop-free | 54.0 / 36.6 / 9.4 % | 55.1 / 35.8 / 9.1 % | 49.8 / 39.2 / 10.9 % | 48.8 / 39.7 / 11.5 % |
| exact | 63.9 / 31.2 / 4.9 % | 64.6 / 31.5 / 3.9 % | 66.3 / 29.2 / 4.4 % | 58.4 / 34.8 / 6.8 % (own geometry) |

**Architecture-specific.** Under crop-free, the two larger models
concentrate less on the center: the top cell is central for 49-50 % of
samples, compared with 54-55 % for MobileNetV2. Their center-to-corner
contrast is smaller: mean `margin_drop` is 0.08 at the center vs 0.02 at the
corners, compared with 0.09-0.10 vs 0.01 for MobileNetV2. Raw values are
logit-margin units and are compared only within each model; the ratios are
descriptive.

### 2. Cross-model agreement (`cross_model.csv`, `fig_a3_similarity.pdf`)

| Pair | Protocol | Profile Spearman | Per-sample Spearman median (IQR) | Top-1 cell agreement (kappa) | Top-3 overlap | Grade agreement (kappa_w) |
|---|---|---|---|---|---|---|
| MNv2-0.5 vs MNv2-1.0 (yardstick) | crop-free | 0.968 | 0.26 (0.05-0.47) | 27.3 % (0.20) | 38.5 % | 48.5 % (0.07) |
| MNv2-0.5 vs ConvNeXt-T | crop-free | 0.891 | 0.12 (-0.10-0.32) | 18.9 % (0.11) | 30.4 % | 46.7 % (0.04) |
| MNv2-0.5 vs DeiT-S | crop-free | 0.947 | 0.12 (-0.09-0.32) | 20.1 % (0.13) | 30.7 % | 47.6 % (0.03) |
| MNv2-1.0 vs ConvNeXt-T | crop-free | 0.924 | 0.16 (-0.06-0.36) | 22.2 % (0.15) | 32.4 % | 47.0 % (0.05) |
| MNv2-1.0 vs DeiT-S | crop-free | 0.935 | 0.15 (-0.07-0.35) | 22.8 % (0.15) | 32.2 % | 47.9 % (0.04) |
| ConvNeXt-T vs DeiT-S | crop-free | 0.944 | 0.18 (-0.04-0.38) | 22.9 % (0.16) | 32.9 % | 48.0 % (0.05) |
| MNv2-0.5 vs MNv2-1.0 (yardstick) | exact | 0.991 | 0.26 (0.04-0.46) | 30.2 % (0.21) | 40.7 % | 47.8 % (0.08) |
| MNv2-0.5 vs ConvNeXt-T | exact | 0.971 | 0.13 (-0.09-0.35) | 21.9 % (0.11) | 34.2 % | 46.0 % (0.05) |
| MNv2-1.0 vs ConvNeXt-T | exact | 0.974 | 0.16 (-0.06-0.37) | 24.5 % (0.14) | 35.9 % | 46.6 % (0.06) |

The chance level for top-1 agreement over 16 cells, given these marginals,
is about 9 %.

- **Dataset level: robust across architectures.** The 16-cell profile
  Spearman is 0.89-0.95 across architectures on crop-free (yardstick 0.97)
  and 0.97 on exact (yardstick 0.99).
- **Sample level: weak.** Per-image rankings agree less across
  architectures (median 0.12-0.18) than within the MobileNetV2 family (0.26).
  Top-1 agreement is 19-23 %, compared with 27 % within the family and about
  9 % by chance.
- **Anchor grades: near chance for every model pair.** Agreement is 46-48 %
  with kappa_w 0.03-0.05 across architectures, compared with 48 % and 0.07-0.08
  within the family. This extends B1's and A2's finding that anchor-level
  grades are not stable per-image labels. The grade distribution is stable,
  but individual anchors are model-specific.
- The same holds on the common-correct subset (5,639 crop-free / 6,002 exact
  samples) and on pair-correct subsets. Cross-architecture profile Spearman is
  0.87-0.94 on crop-free and 0.94-0.98 on exact. Per-sample medians and grade
  agreement are within 0.025 of the values above.

### 3. Reliability grades (`grade_distribution.csv`, `flag_rates.csv`, `fig_a3_grades.pdf`)

| HIGH / LOW / UNRELIABLE | MNv2-0.5 | MNv2-1.0 | ConvNeXt-T | DeiT-S |
|---|---|---|---|---|
| crop-free | 14.5 / 23.9 / 61.6 % | 14.3 / 24.6 / 61.1 % | 16.0 / 23.4 / 60.6 % | 15.5 / 20.8 / 63.7 % |
| exact | 14.3 / 26.7 / 59.0 % | 13.9 / 26.9 / 59.2 % | 15.1 / 26.1 / 58.8 % | 14.2 / 23.7 / 62.0 % |

- **No architecture's grade distribution breaks down.** HIGH is 13.9-16.0 %
  in every run, and MODERATE is 0 everywhere, as in the baselines.
- **UNRELIABLE is driven by `sign_consistent`** (36-41 % true, lowest for
  DeiT-S), as in the baselines.
- **HIGH is center-weighted in every model.** Under exact, ConvNeXt-T is the
  most center-weighted: 26.9 % HIGH at the center vs 7.8 % at the corners.
- **`exceeds_control` is true for 35-39 % of anchors on crop-free**, in every
  model.
- **Area matching follows the protocol, not the model.** The share of
  area-matched anchors is 100 % under crop-free and 0.7 % under exact (1.6 %
  for DeiT-S, whose larger crop narrows the area spread).
- A2 applies: these HIGH shares use K=3 and are about twice what K=20 would
  give.

### 4. Perturbation consistency (`strategy_rank_corr.csv`, `operator_profile.csv`)

Operator-pair Spearman of the 16-cell profiles:

| | blur vs noise | blur vs mean_fill | noise vs mean_fill |
|---|---|---|---|
| MNv2-0.5 cf / exact | 0.92 / 0.90 | 0.92 / 0.99 | 0.94 / 0.91 |
| MNv2-1.0 cf / exact | 0.94 / 0.88 | 0.98 / 0.99 | 0.91 / 0.87 |
| ConvNeXt-T cf / exact | 0.67 / 0.80 | 0.72 / 0.92 | 0.94 / 0.88 |
| DeiT-S cf / exact | **-0.41** / 0.64 | 0.95 / 0.99 | **-0.44** / 0.68 |

- **The two strong operators agree in every model.** `mean_fill` and `blur`
  give the same center-leading order in every run (0.72-0.99).
- **DeiT-S crop-free: Gaussian noise gives a different, near-flat profile.**
  Its 16 cell means span only 0.012-0.024, against 0.025-0.24 for
  `mean_fill`. The largest values are in the top row and the smallest in the
  center. The ranking of this near-flat profile runs opposite to the other two
  operators.
  - All models show a small noise effect (range 0.011-0.027). DeiT-S
    crop-free is the only run where it does not follow the center pattern.
    Under exact preprocessing, DeiT-S's noise profile is again center-leading
    (0.64 / 0.68).
  - This is an architecture- and preprocessing-specific difference that
    SSAT's multi-operator check surfaces. It is not a failure of the
    workflow: a location's importance is consistent across occlusion and
    blur, while DeiT-S's response to local noise at sigma 12.5 carries almost
    no spatial signal after the crop-free resize.
  - The anchor-level `multi_strategy` flag is true for every anchor in every
    run, so this does not reach the grades.

## Interpretation (experiment plan section 7.3)

1. **Applicability.** The SSAT workflow ran unchanged on a modern CNN and a
   vision transformer. Only the model name changed, and the runs had the
   same throughput (about 250-265 items/s, CPU-bound like the MobileNetV2
   runs) and no failures.
2. **Common vs specific patterns.**
   - Common to all four models:
     - Center dominance: the 16-cell rank order separates center, edge, and
       corner exactly in all eight runs.
     - Dataset-level profiles (Spearman at least 0.89).
     - The grade distribution.
   - Architecture-specific:
     - The two larger models concentrate less on the center under crop-free.
     - DeiT-S responds differently to local Gaussian noise.
   - Per-image rankings and anchor grades differ between any two models, more
     so across architectures than within MobileNetV2.
3. **Reliability does not break down** for any architecture (HIGH
   13.9-16.0 %, UNRELIABLE 59-64 %).
4. **Matched controls and operator consistency work in the transformer.**
   Controls are exactly area-matched under crop-free. `exceeds_control` rates
   match the CNNs, and occlusion and blur agree. The exception is Gaussian
   noise under crop-free (section 4).
5. **The paper's qualitative ImageNet conclusion is not MobileNet-specific.**
   Its per-image and anchor-level statements should be framed as
   model-specific.

## Recommended wording and paper locations

- **ImageNet case study / Discussion.** Add one paragraph and one figure
  (`fig_a3_profiles.pdf` or `fig_a3_similarity.pdf`) or a table (the
  cross-model table above):
  - The center-leading profile and the grade distribution replicate on
    ConvNeXt-T and DeiT-S (profile Spearman 0.89-0.95 with MobileNetV2 under
    crop-free).
  - Per-image rankings agree weakly across architectures (median 0.12-0.18,
    compared with 0.26 within MobileNetV2).
- **Methods.** State the model list, weights (SHA-256 in
  `model_selection.json`), and that cross-architecture comparisons use
  crop-free inputs because DeiT-S's official crop (0.9) differs from the
  CNNs' (0.875).
- **Limitations.**
  - Anchor grades and per-image rankings are model-specific.
  - ViT patches (16 px) do not align with the 56 px grid cells.
  - Gaussian noise at sigma 12.5 carries little spatial signal for DeiT-S
    under crop-free.

## Deviations

- D-006: DeiT-S was kept although it fails condition 3. The comparison rule
  follows implementation plan section 4.5.
- D-007: the post-hoc `operator_profile.csv` table.
- Pre-registered outputs were produced from a clean tree after all four runs
  finished.
