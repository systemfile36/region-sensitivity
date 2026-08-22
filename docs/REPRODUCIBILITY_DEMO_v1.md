# Reproducibility Demo (Q1-Q5)

## What this reproduces

The pre-registered Q1-Q5 verdicts from
`docs/internal/L3_Synthetic-Shortcut Experiment Report.md` (crop-free
preprocessing): a `squeezenet1_0` classifier trained on CIFAR-10 with a
synthetic patch shortcut baked into one class (`M_shortcut`), audited
region-by-region and compared against a clean control model (`M_normal`),
to check whether region-sensitivity auditing actually finds the shortcut.

This demo uses the two pretrained checkpoints committed at
`experiments/synthetic_shortcut/pretrained/{m_shortcut,m_normal}.pt`
(5.7MB total) instead of training from scratch. CIFAR-10 itself downloads
automatically via `torchvision.datasets.CIFAR10(download=True)` -- no
license gate, no manual data preparation -- so the whole demo is
self-contained from a fresh clone. Only Phase 4's synthetic-shortcut
result is covered here; see [What is NOT covered](#what-is-not-covered)
below for why the real-dataset case study is scoped out.

## Quick start

The repository's supported development environment is Docker Compose:

```bash
docker compose up -d --build region-sensitivity-workspace
docker compose exec region-sensitivity-workspace bash
```

Inside the container:

```bash
cd experiments/synthetic_shortcut
bash reproduce_demo.sh
```

which runs, step by step (see `reproduce_demo.sh` for the exact commands):

```bash
python3 prepare_data.py                    # CIFAR-10 auto-download; no CLI overrides
python3 verify_crop_free_parity.py         # pre-flight sanity check, no args
python3 run_audit.py --preprocessing crop_free --checkpoint-dir pretrained --results-dir results_crop_free_demo
python3 evaluate_accuracy.py --preprocessing crop_free --checkpoint-dir pretrained --results-dir results_crop_free_demo
python3 evaluate.py --results-dir results_crop_free_demo
python3 check_demo_verdicts.py --results-dir results_crop_free_demo
```

No `train.py` step runs -- that is what makes this the fast path. On a
clean checkout (no cached `data/`), `prepare_data.py` downloads the ~170MB
CIFAR-10 tarball and renders on the order of 66,000 PNG files (both the
demo's and the full retrain path's manifests are built together, since the
script is preprocessing- and training-agnostic). The download dominates:
measured locally against a slow/throttled mirror connection, a clean
`prepare_data.py` run took about 28 minutes end to end, almost entirely
download time (PNG rendering itself took well under a minute once the
tarball was in hand). On a typical broadband or CI-runner connection this
should be well under five minutes. Every step after `prepare_data.py` is
CPU-only inference over a 200-image audit subset across 7 (model, dataset,
fill-strategy) runs and, measured end to end on the same machine with
`data/` already populated, completes in under a minute -- so a rerun (or a
cache hit in CI) finishes the whole sequence in under a minute.

## Why not `ssat run examples/reproduce_q1_q5.yaml`

The original high-level submission plan
(`docs/internal/IMPLE_PLAN_SOFTWAREX_SUBMISSION_v1.md`, Phase 4) sketched a
single-YAML `ssat run examples/reproduce_q1_q5.yaml --output <dir>`
invocation before this experiment's actual architecture was known. The
real Q1-Q5 audit is a 7-run sweep across (model, dataset, fill-strategy)
combinations driven by `run_audit.py`'s `AuditApplication`/`RunRequest`
usage, plus a separate top-1 accuracy check (`evaluate_accuracy.py`) and a
judging pass (`evaluate.py`) -- not expressible as one YAML config: `ssat`'s
`Aggregator` groups `RegionMetrics`/`SpatialProfile` rows by
`(region_key, metric_name)` only, with no `perturb_op` axis (see
`ssat/metrics/aggregate.py` and `run_audit.py`'s own module docstring), so
packing all 5 fill strategies into a single audit config would average
their per-region degradation together -- exactly what Q4's
per-fill-strategy region ranking needs to keep apart. This demo therefore
lives under `experiments/synthetic_shortcut/` as a script pipeline,
following that directory's existing pattern, rather than as another
`configs/examples/*.yaml`.

## What "success" looks like

`check_demo_verdicts.py` asserts these five pre-registered questions all
have `"pass": true` in `verdicts.json`, judged against the thresholds in
`thresholds.json`:

| Question | Criterion | Reference value (crop-free, from the L3 report) |
| --- | --- | --- |
| Q1 identifies patch region | Patch region ranks #1 by mean `margin_drop` under `M_shortcut`/A/`constant_fill` | rank 1 |
| Q2 separated from baseline | Patch region's `margin_drop` is at least `q2_multiplier` (3.0) times the mean of the other regions | multiplier ≈ 175.65 |
| Q3 distinguishes `M_normal` | Patch region does *not* rank #1 under `M_normal`/A/`constant_fill` | rank 16 (of 16) |
| Q4 robust to fill strategy | Patch region ranks #1 in at least `q4_min_fill_strategies` (2) of the 5 fill strategies | 5/5 |
| Q5 predicts generalization gap | `M_shortcut`'s A→C accuracy drop exceeds `M_normal`'s by at least `q5_min_margin_points` (10.0) points | margin ≈ 95.75 points |

These reference values are what the pre-registered run (and this demo,
re-run against the same checkpoints and manifests) produced; treat them as
the expected order of magnitude, not a byte-exact requirement -- the pass/
fail booleans and integer ranks are what `check_demo_verdicts.py` actually
asserts, since floating-point details can vary slightly across
CPU/GPU/BLAS-thread environments while remaining far on the correct side of
every threshold above.

## How to verify automatically

```bash
python3 experiments/synthetic_shortcut/check_demo_verdicts.py \
  --results-dir experiments/synthetic_shortcut/results_crop_free_demo
```

Exit code 0 means all five questions passed; a non-zero exit prints which
question(s) failed and their recorded values.

## What is NOT covered

Phase 3's real-dataset case study (ImageNet-1k validation and NTU60 XSub)
is out of scope for this demo: NTU-RGB+D requires a data-use license and
cannot be redistributed, and ImageNet-1k validation requires a large
separate download that a third party must obtain themselves. See
[Real dataset case study](REAL_DATASET_CASE_STUDY_v1.md) for that protocol
and its results instead.

## Full from-scratch retraining path

To additionally verify that training itself reproduces an equivalent
checkpoint (not just the audit/judgment step covered here), see
`experiments/synthetic_shortcut/README.md`'s "A1. Full retrain path"
section.
