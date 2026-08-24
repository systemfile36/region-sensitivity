# Tutorial: your first audit

This walks through the committed `configs/examples/quickstart.yaml` example end to end, one command at a time, explaining what each command's output means and how the pipeline stages connect. It assumes the [installation](INSTALLATION.md) steps are done and `ssat --help` runs.

The quickstart uses a random, untrained Torchvision model over a tiny synthetic image fixture. Its purpose is to exercise every stage of the pipeline safely and quickly — its metrics carry no scientific meaning. Point the same commands at a trained checkpoint and a real dataset for an actual audit; see the [configuration reference](CONFIG_REFERENCE.md) for the available sources, adapters, regions, and perturbations, and the [real dataset case study](REAL_DATASET_CASE_STUDY_v1.md) for a worked example with trained models.

## 0. What the pipeline does

```text
YAML configuration
      |
      v
estimate -> run -> raw Parquet dump -> metrics -> analysis -> HTML/CSV/JSON report
                    ^
                    +------------ inspect / resume
```

`estimate` and `run` share one preflight step: `estimate` reports what would happen without writing anything, `run` performs the same preflight and then executes. `metrics`, `analyze`, and `report` are each separate, re-runnable steps over an existing dump — none of them re-run the model.

## 1. `ssat estimate` — look before you run

```bash
ssat estimate configs/examples/quickstart.yaml
```

```text
SSAT estimate
  dump mode: none
  clean samples: 20 pending / 20 total
  perturbed items: 80 pending / 80 total
  chunks: 40 pending / 40 total
  estimated remaining time: 0.3s
  estimated dump: 258.4 KiB remaining / 274.4 KiB total
  confirmation required: yes
  sanity top-1 accuracy: 0.00%
  region area consistency: FAIL (max deviation 0.00%, tolerance 5.00%)
  advisory [sanity_partial_failures]: Clean sanity check contained failed or invalid outputs.
  advisory [area_sanity_partial_failures]: Some region geometries could not be evaluated for area consistency.
  advisory [profile_partial_failures]: Perturbed profile contained terminal item failures.
```

How to read this:

- **20 clean samples / 80 perturbed items**: the fixture manifest has 20 images; the config's `grid2x2` region (4 cells) times one `constant_fill` perturbation gives 4 perturbed items per sample, 80 total.
- **`sanity top-1 accuracy: 0.00%`**: expected here — the model has random, untrained weights (`weights: null` in the config), so its predictions carry no signal. A real audit should see this near the trained model's known accuracy; a near-zero or wildly wrong value on a real checkpoint usually means a preprocessing or class-index mismatch, not a genuinely broken model.
- **`region area consistency: FAIL`**: the [preflight area sanity check](CONFIG_REFERENCE.md#preflight-area-sanity) compares each region's intended area fraction against its actual (post-preprocessing) area fraction. This particular FAIL is expected too: the quickstart manifest intentionally references two missing image files to exercise failure recording, and those missing files show up here as geometries the check could not evaluate — not as an area miscalculation. A crop that shrinks edge regions relative to center ones on real data would fail this check for a different, real reason; see the same section for how to interpret that case.
- **Three `advisory` lines**: all three trace back to the same two intentionally-missing files, and all three are non-fatal advisories, not hard errors — they only mean `run` will ask for confirmation before creating a dump the caller has not been warned about (below).

## 2. `ssat run` — create the dump

```bash
ssat run configs/examples/quickstart.yaml --output /tmp/ssat-quickstart --yes
```

`--yes` accepts the advisories/FAIL above explicitly instead of prompting; omit it to get an interactive confirmation prompt reproducing the same estimate output. This runs the same preflight step again (so its output repeats what step 1 showed) and then executes:

```text
SSAT run completed
  output: /tmp/ssat-quickstart
  records written: 100
  OOM events: 0
  final batch size: 4
```

`100 = 20 clean + 80 perturbed` rows written to a new Parquet-backed dump at `/tmp/ssat-quickstart`. Running the exact same command again does not repeat the work — `run` detects the existing dump, recognizes the configuration and code version have not changed, and resumes (0 additional records written); this is what makes a killed or interrupted run safe to retry.

## 3. `ssat inspect` — check what actually happened

```bash
ssat inspect /tmp/ssat-quickstart
```

```text
SSAT dump
  path: /tmp/ssat-quickstart
  schema/code: 1.0.0 / 0.1.0
  model: torchvision:squeezenet1_0:weights=none:init_seed=0
  clean rows: 20
  perturbed rows: 80
  statuses: ok=90, load_failed=10, prepare_failed=0, predict_failed=0, skipped_oom=0
  resumes: 0
  finished: ...
  manifest counts match: yes
```

`statuses` is a per-item-status count across all 100 rows, not per-sample: `load_failed=10` is the two missing images (2 clean rows + 2×4 perturbed rows they generate = 10), and `ok=90` is everything else (18 clean + 72 perturbed) — a real audit with no missing files reports `ok` for every row. A per-item failure like this is durably recorded, not silently dropped: it excludes that item from metrics/analysis rather than crashing the run or corrupting the counts. `manifest counts match: yes` confirms the row counts in the dump's manifest agree with what is actually on disk — a `no` here would indicate a corrupted or partially-written dump.

## 4. `ssat metrics` — score every item

```bash
ssat metrics /tmp/ssat-quickstart
```

```text
SSAT metrics computed
  dump: /tmp/ssat-quickstart
  metrics dir: /tmp/ssat-quickstart/metrics
  primary metric: margin_drop
  metrics: flip_correct_to_wrong, flip_wrong_to_correct, gt_logit_drop, gt_prob_drop, gt_rank_worsening, loss_increase, margin_drop, pred_changed, topk_exit
  item-metric rows: 648
  computed at: ...
```

Every registered metric (nine built-in ones by default) is computed for every `ok`-status perturbed item: `72 ok perturbed items × 9 metrics = 648` rows. `margin_drop` (the drop in the ground-truth class's logit margin) is the default **primary metric** — the one downstream analysis and reporting treat as the headline degradation signal; override it with `--primary-metric NAME` when a different one fits the study better. See [Configuration Reference §Metrics](CONFIG_REFERENCE.md#metrics) for what each built-in metric measures and how to register a custom one.

## 5. `ssat analyze` — control and stability evidence

```bash
ssat analyze /tmp/ssat-quickstart
```

```text
SSAT control/stability analysis computed
  ...
  reliability grades: low=648
  anchors: 80
  conditions insufficient: 80
  controls unmatched: 0
  area mismatch warnings: 0
```

Every row grades `low` reliability here, and every one of the 80 perturbed-item "anchors" is `conditions insufficient` — this is the expected outcome for the minimal quickstart config, not a bug: reliability grading needs repeated seeds, multiple fill strategies, or `controls` to compare against, and the quickstart config has exactly one fill strategy, one seed, and no `controls` block. A study design meant to produce non-`low` grades needs `seed_salts` with more than one value, more than one perturbation `op`, and/or a `controls` section — see [Configuration Reference §Perturbations](CONFIG_REFERENCE.md#perturbations) and [§Area-matched controls](CONFIG_REFERENCE.md#area-matched-controls).

## 6. `ssat report` — the human-readable output

```bash
ssat report /tmp/ssat-quickstart
```

```text
SSAT report generated
  ...
  report dir: /tmp/ssat-quickstart/report
  secondary report (question-driven): /tmp/ssat-quickstart/report/report_question_driven.html
  samples: 18
  regions: 4
  reliability grades: low=648
```

Open `/tmp/ssat-quickstart/report/report.html` in a browser. `samples: 18` (not 20) because the two `load_failed` samples are excluded from the report the same way they were excluded from metrics. Alongside the two HTML reports, `report/data/` holds the same information as flat files for programmatic use — `report_model.json` (the full report data model) and five CSVs (`sample_rankings.csv`, `region_summary.csv`, `semantic_summary.csv`, `class_semantic_matrix.csv`, `flagged_items.csv`). Building a custom downstream view (a dashboard, a paper figure) should read these rather than re-deriving them from the raw dump.

## 7. `ssat export-labels` — optional downstream risk labels

```bash
ssat export-labels /tmp/ssat-quickstart/report
```

```text
SSAT risk labels exported
  labels file: /tmp/ssat-quickstart/report/labels/labels.jsonl
  ...
  labels: 0
  negative: n/a (continuous primary metric)
```

`labels: 0` is expected for this fixture, not an error: by default this command only labels items whose *clean* prediction was correct, and the quickstart's random, untrained model gets essentially nothing right on clean input (recall the 0.00% sanity accuracy from step 1). On a trained model, this command turns per-item metric values into a binary/continuous risk label file suitable for feeding into another tool; pass `--include-non-clean-correct` to label every item regardless of clean correctness, and `--csv` to also write a flat CSV alongside the JSONL.

## Next steps

- Swap in a real source, adapter, and region/perturbation design — the [configuration reference](CONFIG_REFERENCE.md) documents every built-in kind, and `configs/examples/` has runnable configs for video, skeleton-tracked, area-matched-control, and multi-fill-strategy audits.
- Extend SSAT with a new model, dataset, or metric without forking it — see [Application API §Extension points](APPLICATION_API.md#extension-points).
- Reproduce a specific published result end to end, including exact expected numbers — see the [reproducibility demo](REPRODUCIBILITY_DEMO_v1.md).
- Reuse this same pipeline from Python instead of the CLI — see [Application API](APPLICATION_API.md).
