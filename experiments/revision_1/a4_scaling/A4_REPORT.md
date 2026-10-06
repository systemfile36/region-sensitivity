# A4 report: computational scaling

Pre-registered settings: `protocol.json` (commit `2792c33`, before any
measurement). Reference workload: mobilenetv2_050, exact preprocessing,
ImageNet validation images, run settings as in the case study.
Measurements ran from 2026-09-29 05:48Z to 2026-09-30 02:17Z:
- one component profile;
- 41 sweep measurements (15.0 M items);
- four warm-ups;
- 15 `ssat estimate` calls.

Nothing else ran in the container meanwhile. The host's desktop session
(browser, VS Code) stayed open. Deviations: `../deviations.md` D-008 (budget
and mechanics) and D-009 (host OOM, meaning of the run-phase RSS, kept
outliers, workload-table correction).

Host: 32 CPU cores, 125 GiB RAM, RTX 4090 (24 GiB), `num_workers` 12,
batch 128, fp32. "Items" are clean plus perturbed model evaluations:
`N * (1 + R * V * (1 + K))`.

## 1. What dominates the cost (`components.csv`, `components.json`, `measurements.csv`)

| Stage (N=200, 4x4, V=5, K=3) | ms per item | Ceiling (items/s) |
|---|---|---|
| (b) chunk preparation in the 12-worker pool (decode, masks, perturbation) | 1.94 | 516 |
| (b) same, one process (CPU cost per item) | 6.29 | 159 |
| (c) adapter preprocessing (resize, crop, normalize; main process, serial) | 2.19 | 456 |
| (c) model forward incl. host-device transfer (GPU) | 0.14 | 7,335 |
| effective-area mask transform (main process) | 0.15 | 6,606 |
| main process total, excluding dump write | 2.48 | 403 |
| measured audit loop (pooled fit) | 3.80 | 263 |

- **`ssat run` is CPU-bound in the main process, not GPU-bound.**
  - The model forward pass is 4 % of the per-item time.
  - Mean GPU utilization during `run` is 2.4-4.3 % in every measurement.
  - The single-threaded model preprocessing in the main process takes
    58 % of the per-item time.
  - The remaining ~1.3 ms per item (dump writing, transfer of prepared
    arrays from the workers, batching) was not separated (D-008).
  - The worker pool is not the limit: it prepares items about twice as fast
    as the main process consumes them.
- **GPU memory is flat.** Peak GPU memory is 3.3-4.2 GiB in every
  measurement, from 32 k to 1.28 M items, because the batch size is fixed.
  This is the nvidia-smi device total, which also counts other host
  processes. The desktop session held about 2.2 GiB when the idle GPU was
  checked on 2026-09-30, so the audit's own share is roughly 1-2 GiB.
- **Phase shares.** For the 4,000-sample run, `run` is 80 % of the
  end-to-end time, `metrics` 10 %, `analyze` 7.5 %, and `report` 2.5 %.

## 2. Scaling with samples, regions, controls, and perturbations (`fits.csv`, `settings.csv`, `fig_a4_*.pdf`)

Pooled fit over all 41 measurements, `time = a + b * items`:

| Phase | a (s) | b (ms / item) | R^2 |
|---|---|---|---|
| `ssat run` | 22 | 3.84 | 0.979 |
| audit loop only | 12 | 3.80 | 0.979 |
| `metrics` | 1 | 0.46 | 0.999 |
| `analyze` | 2 | 0.34 | 0.998 |
| `report` | 5 | 0.11 | 0.990 |
| all four phases | 30 | 4.76 | 0.986 |

- **Time is linear in items along every axis.**
  - `run` costs 3.75-4.00 ms per item on the samples, regions, controls, and
    perturbations axes (per-axis R^2 0.95-0.997). Throughput therefore
    depends on items, not on which factor produced them.
  - Throughput does not change with N: the audit loop runs at 239-270
    items/s from N=250 to 4,000 (`fig_a4_samples.pdf`). The lowest value is
    the N=500 mean, which includes its slow repeat.
  - The fixed cost is small: preflight (config, plan, sanity check, profile)
    takes 8-60 s, growing slightly with the plan size.
- **Regions, controls, and variants act only through the item count**
  (`fig_a4_regions.pdf`, `fig_a4_controls.pdf`, `fig_a4_perturbations.pdf`).
  Going from 4x4 to 8x8 multiplies the items per sample by 4.0 and the run
  time by 3.9. Time per sample is linear in 1 + K (0.34 s at K=0, 6.8 s at
  K=20).
- **Operators differ modestly in cost.** With mean_fill only (V=1) the
  loop runs at 326 items/s. Adding blur (V=2) gives 279 items/s, adding
  noise (V=5) 266-270 items/s at N=1,000 and 4,000, and adding constant
  fill and patch shuffle (V=7) 282 items/s.
- **Storage is linear.** The raw dump takes 3.6 kB per item (3.4-3.8 by
  axis), dominated by the 1,000 stored logits per item; with metrics,
  analysis, and report it is 3.9 kB. The fit through the origin has
  R^2 >= 0.998.
- **Repeatability.** The coefficient of variation across repeats is at most
  2.3 % for 10 of the 13 repeated settings, and 6.7 % for 2x2. The other two
  settings each have one slow repeat, most likely from other activity on the
  host desktop (kept, D-009):
  - N=500: 815 s vs 624 / 653 s (CV 15 %);
  - 6x6: 4,039 s vs 2,555 / 2,436 s (CV 30 %).

## 3. Cross-checks against separately measured runs (`reference_points.csv`, `estimate_accuracy.csv`)

- **The fit extrapolates correctly to the paper setting (10,000 samples,
  3.21 M items).**
  - Audit loop: 12,223 s predicted vs 11,925 s in the stored baseline's
    manifest (+2.5 %).
  - `metrics` and `analyze`: 1,467 s and 1,108 s predicted vs 1,492-1,660 s
    and 1,140-1,172 s in the A3 runs, which use the same item count.
  - Raw dump: 11.5 GB predicted vs 12.1 GB (-5 %).
- **The A2 K=20 exact run (3.36 M items)** took 12,830 s vs 12,949 s
  predicted (-0.9 %). On the controls axis it is 6.4 s per sample, against
  6.8 s for N=500 at K=20.
- **`ssat estimate` is a usable preflight.** Its predicted run time is
  0.76-1.42x the measured one (median 0.91; within +-25 % for 14 of 15
  settings). It uses 20 sampled chunks and takes 8-62 s.

## 4. Memory: the practical limit (`fits.csv`, `workload_table.csv`, `fig_a4_memory.pdf`, D-009)

- **metrics / analyze / report (single processes)** grow linearly:
  - `analyze` at 17.8 GiB per million items (R^2 0.9996);
  - `metrics` at 15.8 GiB per million items;
  - `report` at about 12 GiB per million items.

  Linear extrapolation puts `analyze` at the 125 GiB host limit near 7.0 M
  items: about 21,800 samples at 4x4 / V=5 / K=3, or 5,500 samples at 8x8.
  This is an extrapolation beyond the measured 1.28 M items. The paper
  setting (3.21 M items) is predicted at 58 GiB, consistent with the ~60 GiB
  observed in earlier runs.
- **The `ssat run` worker pool is the larger risk, and it is not captured by
  the recorded run-phase RSS.**
  - The recorded value is the largest single process (Linux `ru_maxrss` of
    children). It rises with items per sample (9 GiB at 65, 23 GiB at 321,
    34-43 GiB at 1,281-1,681) and levels off in N from N=500 on. Its
    pooled linear fit is poor (R^2 0.60) and is not extrapolated.
  - The total was observed once. During the 8x8 N=1,000 run (1,281 items
    per sample), the host ran out of memory (2026-09-30 02:27 KST). The
    kernel's task dump shows the 12 workers at 109.5 GiB in total, and the
    kernel killed the desktop's VS Code process.
  - The audit itself completed, and its time matched its repeats.
  - With the default 12 workers, settings at or above 8x8 x V=5 x K=3 per
    sample can therefore exhaust 125 GiB in the run phase. Fewer workers
    would presumably reduce this but were not measured.

## 5. Workload table (`workload_table.csv`) and coarse-to-fine (`coarse_to_fine.csv`)

Predictions from the pooled fits; "measured" values are from section 3.

| Workload | Items | `ssat run` | All phases | Raw dump | `analyze` RSS |
|---|---|---|---|---|---|
| Small: 1,000 x 4x4, V=5, K=3 | 0.32 M | 21 min (measured 20 min) | 26 min | 1.2 GB | 6.4 GiB |
| Medium: 10,000 x 4x4, V=5, K=3 (paper) | 3.21 M | 3.4 h (measured 3.3 h) | 4.2 h | 11.5 GB | 58 GiB |
| 10,000 x 8x8, V=5, K=3 | 12.8 M | 13.7 h | 16.9 h | 46 GB | 229 GiB (exceeds host) |
| Large: 10,000 x 8x8, V=5, K=10 | 35.2 M | 37.6 h | 46.5 h | 126 GB | 627 GiB (exceeds host) |

- **The practical limits are memory and storage, not GPU.** Exhaustive
  audits at the paper's scale are practical in hours. Finer grids and larger
  K multiply the cost linearly. Beyond about 7 M items per analysis
  (`analyze`), or at per-sample item counts like 8x8 x K=3 (`run` worker
  pool), this 125 GiB host is exceeded.
- **Coarse-to-fine** (analytic, V=5, K=3): scanning 2x2 and refining only
  the top 2x2 cell to 4x4 sub-cells gives 8x8 resolution there with 401
  items per sample instead of 1,281 for exhaustive 8x8 (31 %). For 10,000
  samples that is 4.3 h and 72 GiB `analyze` RSS instead of 13.7 h and
  229 GiB. Subset sampling reduces every figure in proportion to N.

## 6. Post hoc: re-measurement after the planner fix (D-013, `summary_plan_cache/`)

This section was added after sections 1-5 and the C1 re-measurement (D-012)
were seen. The sweep ran before the planner fix (`d0fcca8`), which removed a
per-chunk recomputation of each sample's items in the main process. Four
settings were measured again with the same scripts, configs, inputs, settings,
warm-ups, and estimates, on 2026-10-02 from 05:54Z to 12:13Z at `620054c`
(`ssat/` clean). `n500_g4_v5_k0` has one chunk per sample, which the fix
cannot change, and serves as the drift control between the two sessions. The
interpretation rule was fixed in D-013 before the run.

| Setting | Chunks per sample | Repeats | Audit loop, ms per item: sweep | After the fix | Ratio | Ratio / drift control |
|---|---|---|---|---|---|---|
| `n500_g4_v5_k0` (drift control) | 1 | 3 | 3.90 | 3.96 | 1.015 | 1.000 |
| `n1000_g4_v5_k3` (reference) | 3 | 3 | 3.70 | 3.66 | 0.988 | 0.974 |
| `n4000_g4_v5_k3` | 3 | 1 | 3.76 | 3.53 | 0.940 | 0.927 |
| `n500_g4_v5_k20` | 14 | 3 | 3.99 | 3.67 | 0.920 | 0.906 |

- **The sessions are comparable.** The drift control changed by 1.5 %. Its
  repeat CV rose to 5 % because its first repeat was slower (170 s vs
  155-156 s).
- **The fix removes the extra cost of many chunks per sample.** Before the
  fix, K=20 items cost 8 % more than 4x4 K=3 items (3.99 vs 3.70 ms).
  Afterwards they cost the same (3.67 vs 3.53-3.66 ms). The gain is 9 % at
  K=20 and 3 % at the reference setting. K=0 remains the most expensive
  per item, because the fixed cost per sample is shared by only 81 items.
- **The cost stays linear, with a lower slope.** Over these four settings
  (10 measurements each), `time = a + b * items` gives:

  | Fit (`summary_plan_cache/fits.csv`) | Sweep | After the fix |
  |---|---|---|
  | Audit loop, ms per item (R^2) | 3.88 (0.9972) | 3.58 (0.9984) |
  | `ssat run`, ms per item | 3.92 | 3.62 |
  | End to end, ms per item | 4.83 | 4.52 |
  | Sec. 3.2 setting (3.21 M items), audit loop | 12,454 s | 11,520 s (3.2 h) |
  | Sec. 3.2 setting, end to end | 4.3 h | 4.0 h |

  The subset fit of the sweep (3.88 ms) is above the pooled sweep fit
  (3.80 ms, section 2) because the subset includes K=0 and K=20. The stored
  Sec. 3.2 run (11,925 s) ran before the fix, so it does not check the
  post-fix prediction.
- **Nothing else changed.** `metrics`, `analyze`, and `report` times are
  within 3 %. Stored bytes are identical. Peak RSS is unchanged (run phase
  12-39 GiB largest process, `analyze` 1.3-23.6 GiB). Mean GPU utilization
  rose from 2.9-3.8 % to 3.3-4.3 %.
- **`ssat estimate`** predicted 0.95-1.31x the measured `ssat run` for these
  settings (3 of 4 within +-25 %), against 0.83-1.13x in the sweep.
- **Memory.** During `n500_g4_v5_k20` repeat 0 the host ran out of memory
  (worker pool about 104 GiB plus 10.5 GiB main process). The kernel killed
  the desktop browser and VS Code, and the measurement completed (D-013).
  The fix does not change memory: the largest run-phase process was
  38.7 GiB before and after.

**Decision under the D-013 rule.** The K=20 adjusted ratio (0.906) is
outside 0.95-1.05. The manuscript therefore quotes the post-fix per-item
time and Sec. 3.2 prediction (3.6 ms per evaluation for `run`, 4.5 ms end
to end, about 4 h for the Sec. 3.2 setting). The pre-registered sweep
(sections 1-5) stays as recorded and is reported next to it in the
Response.

## Interpretation (experiment plan section 8.6)

1. **What dominates the cost.** The number of model evaluations. Per item,
   the time goes to single-threaded model preprocessing in the main process
   and the dump path, not to the GPU (4 % of the item time, <5 % mean
   utilization). Faster hardware for the model would not help; parallel
   preprocessing would (a possible software follow-up after the revision;
   `ssat` is not changed now).
2. **Is the growth predictable?** Yes. Time and storage are linear in
   `N * (1 + R * V * (1 + K))`. In the sweep (before the planner fix), one
   per-item constant (3.8 ms for `run`, 4.8 ms end to end on this host)
   predicts runs 2.5x larger than any measured setting to within 3 %. After
   the fix the constant is 3.6 ms for `run` and 4.5 ms end to end
   (section 6). `ssat estimate` gives a +-25 % preflight
   on any machine.
3. **When is exhaustive auditing burdensome?**
   - The paper setting (3.2 M items) takes about 4 h end to end.
   - An exhaustive 8x8 audit of 10,000 images takes about 17 h and exceeds
     125 GiB in `analyze`.
   - 8x8 per-sample workloads already reached 110 GiB in the run-phase
     worker pool.
4. **Why coarse-to-fine or subsets are needed.** Cost is proportional to
   regions x variants x (1 + K). A 2x2 -> 4x4 refinement cuts an 8x8-
   resolution audit to 31 % of the items, and memory falls with it.

## Recommended wording and paper locations

- **Computational cost / scalability paragraph** (new, or an extension of
  the BENCHMARK_v1 statement):
  - Linear cost in the item count, 3.6 ms per model evaluation for `run`
    and 4.5 ms end to end on one RTX 4090 host, CPU-bound. These are the
    post-fix values of section 6 (D-013 rule); the sweep's 3.8 / 4.8 ms and
    its +2.5 % cross-check go to the Response.
  - GPU memory constant at ~4 GiB device total (roughly 1-2 GiB for the
    audit itself; section 1).
  - Storage 3.9 kB per item.
  - Figures: `fig_a4_samples.pdf` and `fig_a4_regions.pdf` (required
    figures 1 and 2), `fig_a4_controls.pdf` (3), `fig_a4_memory.pdf` (4).
- **Workload table.** The small / medium / large rows above, labelling the
  extrapolated entries.
- **Limitations and practical guidance.**
  - Memory is the scaling limit (`analyze` about 18 GiB per million items;
    run worker pool up to 110 GiB at 8x8 per-sample workloads).
  - Recommend coarse-to-fine or subset sampling beyond a few million items.
  - Recommend running `ssat estimate` first.

## Deviations

D-008 (budget ~20 h instead of ~10 h; own GPU poller; `--log-file` run split;
full-pipeline warm-ups; dump write not timed; grid ids) and D-009 (host OOM;
run-phase RSS semantics; kept outliers; run-phase RSS removed from the
workload predictions after the first fit output). All summaries were
regenerated from a clean tree at `a310768`. `components.json` was produced
from a clean tree at `2792c33` before the sweep. D-013 (section 6): the post
hoc re-measurement ran at `620054c` (`ssat/` clean), and
`summary_plan_cache/` was generated from a clean tree at `b22aeb9`.
