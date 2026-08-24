# Runtime/Storage Performance Benchmark

## What this measures

Wall-clock runtime, throughput (items/sec), peak memory, on-disk dump size,
and the effect of cache/resume, measured on the *real* `ssat` CLI end to
end (`run` -> `metrics` -> `analyze` -> `report`) rather than predicted.
This is a different thing from `ssat estimate`'s preflight report: that
command samples a small bounded subset of pending work before a real run
starts and extrapolates a prediction from it (see
`ssat/core/estimate/profiler.py`/`cost_model.py`), which is useful for
deciding whether a config is safe to run but is not a measurement of an
actual full run. Everything in this document is measured, not predicted,
using `experiments/benchmark_runtime_storage/run_benchmark.py`.

Two scales are measured:

- **Quickstart** -- the committed `configs/examples/quickstart.yaml`
  (CPU, `squeezenet1_0`, 20-image fixture). Cheap enough to run in CI on
  every push as an advisory job.
- **Real dataset** -- the Phase 3 case study's
  `experiments/real_dataset_case_study/configs/imagenet_mnv2_050_exact.yaml`
  (CUDA, `timm` `mobilenetv2_050.lamb_in1k`, ImageNet-1k val, 10,000
  samples, 4x4 grid regions). Of the six Phase 3 configs, only this one is
  re-measured here (see [Known limitations](#known-limitations)); it
  requires a local CUDA GPU and the ImageNet-1k validation data, so it is
  measured locally only, not in CI.

A third, smaller CPU scale is used only to demonstrate the cache/resume
effect (see [Cache/resume effect](#cacheresume-effect)); it is not one of
the two headline scales above.

## How to reproduce

Quickstart scale (no external data, runs in well under a minute):

```bash
docker compose exec region-sensitivity-workspace bash
python3 experiments/benchmark_runtime_storage/run_benchmark.py quickstart
```

Real-dataset scale (needs a local CUDA GPU and the ImageNet-1k validation
data prepared exactly as described in
[Real dataset case study](REAL_DATASET_CASE_STUDY_v1.md); took about 2h14m
end to end when measured -- see
[Known limitations](#known-limitations)):

```bash
python3 experiments/benchmark_runtime_storage/run_benchmark.py real-dataset
```

Cache/resume scale (CPU only, no external data; generates its own larger
synthetic fixture first):

```bash
python3 experiments/benchmark_runtime_storage/prepare_resume_fixture.py
python3 experiments/benchmark_runtime_storage/run_benchmark.py resume
```

Each subcommand writes a machine-readable
`experiments/benchmark_runtime_storage/results/<scale>.json` with every
number in the tables below plus the measured environment. Pass `--dry-run`
to any subcommand to print the commands it would run without executing
them.

## Quickstart-scale results

Measured on: Linux, 32 CPU cores, ~125.6 GiB RAM, Python 3.11.13,
torch 2.8.0+cu129, no GPU used (config sets `device: cpu`).

| phase | wall time (s) | peak RSS (MiB) |
| --- | ---: | ---: |
| `run` | 2.91 | 972.4 |
| `metrics` | 0.74 | 232.8 |
| `analyze` | 0.73 | 229.7 |
| `report` | 1.46 | 241.2 |

- Throughput: 34.3 items/sec (100 clean+perturbed items total, over the
  `run` phase's wall time).
- Raw dump size (`clean/` + `perturbed/` + `index/` + `run_manifest.json`,
  snapshotted right after `run`): 495.6 KiB.
- Total artifact size (after `metrics`/`analyze`/`report` also ran):
  1.17 MiB.

## Real-dataset-scale results (`imagenet_mnv2_050_exact`)

Measured on: Linux, 32 CPU cores, ~125.6 GiB RAM, 1x NVIDIA GeForce RTX
4090, CUDA 12.9, Python 3.11.13, torch 2.8.0+cu129. 10,000 ImageNet-1k
validation samples, 4x4 grid regions, `--minimum-accuracy 0.50` (matches
Phase 3's gate for this config; the preflight sanity check measured 70.00%
clean top-1 accuracy, so the gate passed).

| phase | wall time | peak host RSS (GiB) |
| --- | ---: | ---: |
| `run` | 1h 46m 28s | 19.43 |
| `metrics` | 12m 46s | 25.73 |
| `analyze` | 10m 30s | 30.40 |
| `report` | 3m 56s | 22.94 |
| total pipeline | 2h 13m 40s | -- |

- Throughput: 252.0 items/sec (1,610,000 clean+perturbed items, over the
  `run` phase's wall time).
- Peak GPU memory (sampled via `nvidia-smi` polling during `run`
  only): 3,152 MiB. See [Known limitations](#known-limitations) for why
  this is an approximation, distinct in rigor from the exact host-RSS
  figures above.
- Raw dump size (`clean/` + `perturbed/` + `index/` + `run_manifest.json`,
  snapshotted right after `run`): 5.64 GiB.
- Total artifact size (after `metrics`/`analyze`/`report` also ran):
  7.11 GiB. This is consistent with the independently-measured, untimed
  Phase 3 dump for the same config
  (`experiments/real_dataset_case_study/results/imagenet_mnv2_050_exact/`,
  ~7.2GB on disk), a useful cross-check that this re-run behaved
  equivalently to the original Phase 3 run.
- `metrics`/`analyze` peak RSS (25.7-30.4 GiB) is the most memory-hungry
  part of the whole pipeline, not `run` itself: `analyze` builds an
  item-context table with 1,440,000 reliability rows over 320,000 anchors
  in a single pandas process, which dominates over the GPU-inference `run`
  phase's own working set.

## Cache/resume effect

Measured with a 300-image synthetic fixture (generated by
`prepare_resume_fixture.py`, not the committed 20-image test fixture) and
`configs/resume_bench.yaml` (CPU, `squeezenet1_0`, same shape as
quickstart): 302 clean samples, 1,208 perturbed items, 1,510 items total.

| scenario | wall time (s) | items completed |
| --- | ---: | ---: |
| baseline (fresh, full run) | 9.64 | 1,510 |
| interrupted (SIGTERM at ~50% target) | 4.90 | 760 |
| resumed (same output dir, to completion) | 5.47 | 1,510 (750 more written) |

`interrupted` was stopped by polling `run_manifest.json`'s
`counts_by_status` until it reached roughly half of the plan's total items,
then sending SIGTERM; `ssat` recorded a clean `returncode == -15`
(terminated) rather than crashing. Re-running the identical `ssat run`
command against the same, now partially-populated output directory
auto-detected resume mode (`dump mode: resume` in its own preflight
output, `clean samples: 2 pending / 302 total`,
`perturbed items: 748 pending / 1,208 total`) and completed only the
remaining ~750 items rather than redoing the ~760 already-written ones.
`ssat inspect --json` on the finished dump reports `resume_count: 1`,
confirming exactly one resume event was recorded.

At this small CPU scale, `interrupted + resumed` (4.90s + 5.47s = 10.37s)
was not faster than `baseline` (9.64s): each of the two `ssat run`
invocations pays its own process-startup and preflight-estimate cost
(importing torch, resolving the config, running the bounded sanity/profile
sample), which dominates over the few seconds of actual audit work at this
scale. The evidence that resume works is not this wall-clock comparison --
it is that the item counts above never restarted from zero and
`resume_count` is exactly 1: the ~760 items completed before the
interruption were not redone. Resume's wall-clock benefit becomes visible
at scales where per-item work is large relative to per-invocation
overhead, i.e. the real-dataset scale, where losing hours of completed GPU
work to a crash and having to redo it would be the alternative.

## Known limitations

- **Peak memory** is host RSS (`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`)
  for every phase above; it does not include GPU VRAM. GPU VRAM for the
  real-dataset scale is reported separately, sampled via `nvidia-smi`
  polling (default every 200ms) during the `run` phase only -- an
  approximation that can miss sub-interval spikes and would include any
  other process sharing the same GPU.
- **Only one of Phase 3's six configs** (`imagenet_mnv2_050_exact`) is
  re-measured here, to bound GPU time cost; it is not repeated across the
  other model/preprocessing combinations or the NTU60 video modality. See
  [Real dataset case study](REAL_DATASET_CASE_STUDY_v1.md) for the full
  six-run matrix's correctness results (not timing).
- **Single-trial measurement**, not repeated-trial statistics: each number
  above is one run, not a mean/variance over several. The real-dataset run
  took about 2h14m end to end (1h46m of that in `run` alone); repeating it
  for variance was judged not worth the added GPU cost for what this
  document needs to show.
- **Timing includes `ssat run`'s own preflight estimate/confirmation
  pass**, not just the audit loop -- this is deliberate, since it matches
  what a real user experiences when they run `ssat run ... --yes`.
- **The `imagenet_mnv2_050_exact` config uses "exact" (non-crop-free)
  preprocessing**, which Phase 1's area-sanity check correctly flags as
  `region area consistency: FAIL` (a large max deviation from the nominal
  grid-cell area) -- this is the known preprocessing artifact Phase 3
  already documents for the "exact" variant, not a defect introduced by
  this benchmark; see
  [Real dataset case study](REAL_DATASET_CASE_STUDY_v1.md) and
  `docs/internal/CONTROL_STABILITY_DESIGN_v1.md` for why both "exact" and
  "crop_free" variants exist.
