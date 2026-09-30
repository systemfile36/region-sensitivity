# experiments/revision_1

Additional experiments for the SSAT minor revision (reviewer requests on
reliability-threshold sensitivity, matched-control count, architecture
coverage, computational scaling, ImageNet preprocessing confound, NTU
semantic validation, and the Captum comparison). Code for all of them lives
here; the `ssat/` package keeps its v1.0.0 defaults and behavior throughout
the revision.

**All commands run inside the `region-sensitivity-workspace` Docker Compose
container** (`docker compose exec region-sensitivity-workspace bash -lc '...'`),
never on the host. Scripts are run from the repository root.

## Layout

| Path | Tracked | Content |
|---|---|---|
| `common/` | yes | Baseline run registry, column-selective loaders, agreement statistics, deterministic subsets, provenance, matrix runner |
| `phase0/` | yes | Baseline freeze and verification scripts (P0-1 to P0-7) |
| `phase0/summary/` | yes | Small JSON outputs of Phase 0, each with a `*.provenance.json` |
| `deviations.md` | yes | Every departure from the pre-registered plan, with timing |
| `<experiment>/protocol.json` | yes | Pre-registered settings, committed before the experiment runs |
| `results/` | no (`results*/`) | Dumps, recomputed stores, logs, and other large intermediates |

## Baseline runs

`common/runs.py` is the single source of truth for which stored outputs the
revision reads (`BASELINE_RUNS`):

| Run | Stores read | K (controls/target) |
|---|---|---|
| `imagenet_mnv2_{050,100}_{exact,crop_free}_k3` | `experiments/real_dataset_case_study/results_control_3/<run>/` | 3 |
| `ntu60_tsm_{exact,crop_free}` | dump in `experiments/real_dataset_case_study/results/<run>/`; metrics/analysis recomputed by P0-7 | 0 |
| `synthetic_{shortcut,normal}` | dump in `experiments/synthetic_shortcut/results_crop_free/dumps/`; metrics/analysis recomputed by P0-7 | 2 |

`K1_PARITY_RUNS` (`experiments/real_dataset_case_study/results/imagenet_*`,
K=1) are read only because the paper's accuracy / mean `margin_drop` table
came from them.

## Phase 0: baseline freeze

| ID | Command | Output (`phase0/summary/`) | Time |
|---|---|---|---|
| P0-1 | `python experiments/revision_1/phase0/build_baseline_manifest.py` | `baseline_manifest.json` | ~2 min |
| P0-2 | `python experiments/revision_1/phase0/verify_paper_numbers.py` | `paper_numbers.json` | ~1 min |
| P0-3 | `python experiments/revision_1/phase0/verify_target_parity.py` | `target_parity.json` | ~13 min, ~15 GB RAM |
| P0-4 | (doc edit) | `docs/BENCHMARK_v1.md` notes the benchmark ran with K=1 | – |
| P0-5 | `SSAT_CONTAINER_IMAGE_ID=<id> python experiments/revision_1/phase0/capture_environment.py --note ...` | `environment.json` | seconds |
| P0-6 | `python -m pytest -q tests/unit/test_rev1_common.py` | – | seconds |
| P0-7 | `python experiments/revision_1/phase0/recompute_small_baselines.py` | `recomputed_baselines.json` (+ stores in `results/phase0/recomputed/`) | ~3 min |
| Images | `python experiments/revision_1/phase0/verify_imagenet_images.py` | `imagenet_images.json` | seconds |

P0-7 must run before any experiment that reads the NTU or synthetic
baselines. Findings are summarized in `phase0/PHASE0_REPORT.md`.

To pass the image id from the host:

```bash
docker compose exec -e SSAT_CONTAINER_IMAGE_ID="$(docker inspect --format '{{.Id}}' local/region-sensitivity-workspace:latest)" \
  region-sensitivity-workspace bash -lc 'python experiments/revision_1/phase0/capture_environment.py'
```

## Phase 1A: A1 reliability threshold sensitivity (offline)

Pre-registered in `a1_threshold/protocol.json`; results in
`a1_threshold/A1_REPORT.md`.

| Step | Command | Output | Time |
|---|---|---|---|
| Features | `python experiments/revision_1/a1_threshold/build_features.py --runs all` | `results/a1/features/<run>__margin_drop.{parquet,json}` | ~1 min |
| Parity gate + sweep | `python experiments/revision_1/a1_threshold/run_sweep.py` | `a1_threshold/summary/*.csv`, `parity.json` | ~15 s |
| Tables and figures | `python experiments/revision_1/a1_threshold/summarize.py` | `sensitivity_ranking.csv`, `table_a1.*`, `fig_a1_*.pdf` | seconds |
| Tests | `python -m pytest -q tests/unit/test_rev1_grade_engine.py` | – | seconds |

`build_features.py` fails unless the recomputed 95 % intervals reproduce
`intervals.parquet`, and `run_sweep.py` stops unless the default-setting grades
reproduce `reliability.parquet` exactly.

## Phase 1B: B1 ImageNet preprocessing confound (offline)

Pre-registered in `b1_preprocessing/protocol.json`; results in
`b1_preprocessing/B1_REPORT.md`. B1-4 (effective-area-matched controls) runs
after the A2 K=20 dumps exist.

| Step | Command | Output (`b1_preprocessing/summary/`) | Time |
|---|---|---|---|
| B1-1 areas | `python experiments/revision_1/b1_preprocessing/area_tables.py` | `cell_area.csv`, `within_sample_area_ratio.csv`, `control_area_ratio.csv`, `zero_area_targets.csv` | ~1 min |
| B1-2/3 ranking and grades | `python experiments/revision_1/b1_preprocessing/compare_protocols.py` | `ranking_change.csv`, `cell_rank_change.csv`, `top_region_share.csv`, `grade_*.csv`, `by_cell_type.csv`, `yardstick.csv`, `representative_cases.csv` | ~10 s |
| Figures | `python experiments/revision_1/b1_preprocessing/plot_figures.py --image-root <dir with the 3 example JPEGs>` | `fig_b1_area.pdf`, `fig_b1_rank.pdf`; `results/b1/fig_b1_examples.pdf` (untracked, embeds ImageNet images) | ~1 min |
| Tests | `python -m pytest -q tests/unit/test_rev1_b1.py` | – | seconds |

`area_tables.py` fails unless its anchor-level area ratios reproduce the stored
`area_matched` flag exactly. The example images are checked against the
dumps' `content_hash` (`deviations.md` D-004, D-005).

## Phase 2: A2 matched-control count ablation (GPU)

Pre-registered in `a2_control_count/protocol.json`; results in
`a2_control_count/A2_REPORT.md` (B1-4 in `b1_preprocessing/B1_REPORT.md`). Three K=20 runs
(ImageNet mnv2_050 crop-free = primary, ImageNet mnv2_050 exact, synthetic
shortcut) on 2,000 ImageNet samples (2 per class) and the 200 synthetic
samples; the K ablation is computed offline from them.

| Step | Command | Output | Time |
|---|---|---|---|
| Inputs | `python experiments/revision_1/a2_control_count/make_inputs.py [--check]` | `data/revision_1/imagenet/val_2_per_class_a2.txt`, `configs/synthetic_shortcut_k20.yaml` | ~10 s |
| Runs | `python experiments/revision_1/common/run_matrix.py --matrix experiments/revision_1/a2_control_count/matrix.json --output-root experiments/revision_1/results/a2 --skip-report` | `results/a2/<run>/` | ~3.5 h per ImageNet run (+ metrics/analyze) |
| Nested parity | `python experiments/revision_1/a2_control_count/verify_nested.py` | `summary/nested_parity.json` | minutes |
| Control tensors | `python experiments/revision_1/a2_control_count/build_control_tensor.py --runs all` | `results/a2/tensors/<run>__margin_drop.{npz,json}` | minutes |
| Ablation | `python experiments/revision_1/a2_control_count/run_ablation.py` | `summary/*.csv`, `ablation_parity.json` (features in `results/a2/features/`) | minutes |
| Figures | `python experiments/revision_1/a2_control_count/summarize.py` | `fig_a2_{convergence,high_share,flip}.pdf` | seconds |
| B1-4 | `python experiments/revision_1/b1_preprocessing/eff_area_controls.py` | `b1_preprocessing/summary/eff_area_controls.csv` | minutes |
| Tests | `python -m pytest -q tests/unit/test_rev1_a2.py` | – | seconds |

`build_control_tensor.py` fails unless the full K=20 control statistics
reproduce the stored `control_comparison` bit for bit, and `run_ablation.py`
fails unless prefix K=20 reproduces `reliability.parquet` and prefix K=3
(K=2 and 3 for synthetic) reproduces the ssat analysis path on the filtered
items.

## Phase 3: A3 multi-architecture comparison (GPU)

Pre-registered in `a3_multi_arch/protocol.json`. ConvNeXt-T
(`convnext_tiny.fb_in1k`) and DeiT-S (`deit_small_patch16_224.fb_in1k`), both
protocols, on the same 10,000 samples and settings as the MobileNetV2
baselines (K=3). DeiT-S fails the same-geometry condition (`crop_pct` 0.9),
so the four-model comparison uses the crop-free runs (`deviations.md` D-006).

| Step | Command | Output | Time |
|---|---|---|---|
| Model check | `HF_HOME=/workspace/data/hf_cache python experiments/revision_1/a3_multi_arch/inspect_models.py` | `a3_multi_arch/summary/model_selection.json` | ~2 min |
| Runs | `HF_HOME=/workspace/data/hf_cache python experiments/revision_1/common/run_matrix.py --matrix experiments/revision_1/a3_multi_arch/matrix.json --output-root experiments/revision_1/results/a3 --skip-report` | `results/a3/<run>/` | ~4-5 h per run (+ metrics/analyze) |
| Run checks | `HF_HOME=/workspace/data/hf_cache python experiments/revision_1/a3_multi_arch/verify_runs.py` | `summary/run_checks.json` | minutes |
| Tables | `python experiments/revision_1/a3_multi_arch/compare_models.py` | `summary/*.csv` | minutes |
| Figures | `python experiments/revision_1/a3_multi_arch/summarize.py` | `fig_a3_{profiles,grades,similarity}.pdf` | seconds |
| Operator profiles (post hoc, D-007) | `python experiments/revision_1/a3_multi_arch/operator_profiles.py` | `summary/operator_profile.csv` | ~5 min |
| Tests | `python -m pytest -q tests/unit/test_rev1_a3.py` | – | seconds |

Results are in `a3_multi_arch/A3_REPORT.md`.

`inspect_models.py` fails unless mobilenetv2_050's model-space cell areas,
recomputed from the source image shapes, reproduce the `effective_area_px`
stored in both mobilenetv2_050 baseline dumps. `verify_runs.py` fails unless
each new run has the baseline's items, seeds, and images, the recorded
weights, and the expected effective areas.

## Phase 4: A4 computational scaling (GPU, dedicated time)

Pre-registered in `a4_scaling/protocol.json`. Reference workload
mobilenetv2_050 exact; four axes (samples, regions, controls,
perturbations) around N=1,000 / 500, 4x4, V=5, K=3; 3 repeats (1 for
N >= 2,000). Nothing else may run on the host during the sweep (~20 h,
`deviations.md` D-008).

| Step | Command | Output | Time |
|---|---|---|---|
| Inputs | `python experiments/revision_1/a4_scaling/make_inputs.py [--check]` | `data/revision_1/imagenet/a4_N{50,...,4000}.txt` | seconds |
| Components | `python experiments/revision_1/a4_scaling/profile_components.py` | `summary/components.json` | ~4 min |
| Sweep | `python experiments/revision_1/a4_scaling/run_scaling.py all --repeats 3` (`--dry-run` lists the schedule) | `results/a4/{measurements,warmups,estimates}.jsonl`, `results/a4/logs/` | ~20 h |
| Fits and tables | `python experiments/revision_1/a4_scaling/fit_scaling.py` | `summary/*.csv` | seconds |
| Figures | `python experiments/revision_1/a4_scaling/summarize.py` | `fig_a4_{samples,regions,controls,perturbations,memory}.pdf` | seconds |
| Tests | `python -m pytest -q tests/unit/test_rev1_a4.py` | – | seconds |

`run_scaling.py` skips measurements already in `measurements.jsonl`, so an
interrupted sweep resumes where it stopped. Results are in
`a4_scaling/A4_REPORT.md`. The run-phase RSS is the largest single process,
not the worker pool total (`deviations.md` D-009).

## Phase 6: C1 Captum comparison, resource measurement (GPU)

Pre-registered in `c1_captum/protocol.json` and `c1_captum/command_map.md`
(C1-min). The Captum reference workflow and the SSAT synthetic-shortcut
scripts are each run 3 times in alternating order. Each command is measured
for wall time, peak RSS (largest process and summed process tree), GPU
memory against the idle baseline, and storage. Requires `captum==0.9.0` in
the container (`pip install -e ".[reference]"`).

| Step | Command | Output | Time |
|---|---|---|---|
| Measure | `python experiments/revision_1/c1_captum/measure_resources.py --repeats 3` (`--dry-run` lists the commands) | `results/c1/measurements.jsonl`, `results/c1/{captum,ssat}_fresh/r<repeat>/` | ~70 min |
| Summaries and checks | `python experiments/revision_1/c1_captum/summarize.py` | `summary/{measurements,runs,resources,storage}.csv`, `summary/verification.json` | seconds |
| Tests | `python -m pytest -q tests/unit/test_rev1_c1.py` | – | seconds |

Results and the claim-narrowing draft are in `c1_captum/C1_REPORT.md`.
C1-std (ImageNet second setting) was not run (protocol default).

## Running new audits

`common/run_matrix.py --matrix <matrix.json> --output-root <dir>` runs
`ssat run -> metrics -> analyze -> report` for each entry and appends per-step
wall times to `<dir>/run_matrix_log.jsonl`. Do not run two `analyze` steps
at the same time (a 3.2 M-item analysis peaks around 60 GiB RSS).

## Reviewer comment -> artifact

Filled in as experiments complete.

| Reviewer request | Artifact | Location |
|---|---|---|
| Config / seed / environment / commit for every experiment | `baseline_manifest.json`, `environment.json`, each `*.provenance.json` | `phase0/summary/` |
| Threshold sensitivity figure/table | `fig_a1_z_curve.pdf`, `fig_a1_oat.pdf`, `table_a1.*`, `sensitivity_ranking.csv` | `a1_threshold/summary/` |
| Reliability grade transition summary | `transitions.csv`, `fig_a1_transitions.pdf`, `agreement.csv` | `a1_threshold/summary/` |
| Preprocessing: nominal vs effective area | `cell_area.csv`, `within_sample_area_ratio.csv`, `control_area_ratio.csv`, `fig_b1_area.pdf` | `b1_preprocessing/summary/` |
| Preprocessing: ranking and grade change | `ranking_change.csv`, `grade_transition_exact_to_cf.csv`, `yardstick.csv`, `fig_b1_rank.pdf`, `eff_area_controls.csv` | `b1_preprocessing/summary/` |
| 3/5/10/20 control-count ablation | `grade_vs_k.csv`, `control_mean_convergence.csv`, `effect_convergence.csv`, `ranking_vs_k.csv`, `fig_a2_convergence.pdf`, `fig_a2_high_share.pdf` | `a2_control_count/summary/` |
| Control-count stability / variance, cost | `replicate_variability.csv`, `fig_a2_flip.pdf`, `marginal_gain.csv` | `a2_control_count/summary/` |
| Model info and evaluated samples per architecture | `model_selection.json`, `model_info.csv`, `run_checks.json` | `a3_multi_arch/summary/` |
| Per-architecture sensitivity and reliability | `region_profile.csv`, `top_region_share.csv`, `grade_distribution.csv`, `flag_rates.csv`, `fig_a3_profiles.pdf`, `fig_a3_grades.pdf` | `a3_multi_arch/summary/` |
| Cross-architecture ranking comparison, common vs specific patterns | `cross_model.csv`, `common_patterns.csv`, `strategy_rank_corr.csv`, `operator_profile.csv`, `fig_a3_similarity.pdf` | `a3_multi_arch/summary/` |
| Sample / region count vs wall-clock, throughput, storage | `fig_a4_samples.pdf`, `fig_a4_regions.pdf`, `fits.csv`, `settings.csv` | `a4_scaling/summary/` |
| Control / perturbation count vs cost | `fig_a4_controls.pdf`, `fig_a4_perturbations.pdf`, `reference_points.csv` | `a4_scaling/summary/` |
| Peak GPU / host memory by workload size; cost breakdown | `fig_a4_memory.pdf`, `components.csv`, `workload_table.csv`, `coarse_to_fine.csv`, `estimate_accuracy.csv` | `a4_scaling/summary/` |
| Captum comparison: runtime / memory / storage | `resources.csv`, `runs.csv`, `storage.csv`, `verification.json` | `c1_captum/summary/` |
| Captum comparison: claim narrowing (engineering burden) | `C1_REPORT.md` (draft wording) | `c1_captum/` |
