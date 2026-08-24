# experiments/benchmark_runtime_storage

Measures `ssat`'s real end-to-end runtime, throughput, peak memory, on-disk
dump size, and cache/resume behavior by driving the real CLI (`run` ->
`metrics` -> `analyze` -> `report`) as subprocesses and timing each phase
from the outside. Results and their interpretation are documented in
[`docs/BENCHMARK_v1.md`](../../docs/BENCHMARK_v1.md); this directory holds
only the scripts and configs that produce them.

**All commands below run inside the `region-sensitivity-workspace` Docker
Compose container** (`docker compose exec region-sensitivity-workspace
bash`), never on the host.

This intentionally does not reuse `ssat/core/estimate/profiler.py` or
`cost_model.py`: those measure a small bounded sample before a real run to
produce a *predicted* estimate for `ssat estimate`, without writing a dump.
Here we want a *measured* number for a real, full run, so
`run_benchmark.py` shells out to the actual `ssat` CLI instead, the same
way `experiments/real_dataset_case_study/run_matrix.py` does.

## Script inventory

| Script | Role |
| --- | --- |
| `run_benchmark.py` | Main entry point. Subcommands `quickstart`, `real-dataset`, `resume`, `all`. Also implements an internal `--internal-run-one` mode (see its module docstring) used to keep each phase's `resource.getrusage(RUSAGE_CHILDREN)` reading isolated from the others. |
| `prepare_resume_fixture.py` | Generates the 300-image synthetic fixture used only by the `resume` subcommand, into a gitignored `data/resume_fixture/` -- separate from the committed 20-image test fixture, which is too small for a reliable mid-run interruption. |
| `configs/resume_bench.yaml` | CPU/`squeezenet1_0` config pointed at that fixture. |

`configs/examples/quickstart.yaml` (quickstart scale) and
`experiments/real_dataset_case_study/configs/imagenet_mnv2_050_exact.yaml`
(real-dataset scale) are reused as-is, not copied here.

## Quick start

```bash
docker compose exec region-sensitivity-workspace bash

# Quickstart scale: no external data, well under a minute.
python3 experiments/benchmark_runtime_storage/run_benchmark.py quickstart

# Cache/resume scale: CPU only, generates its own fixture first.
python3 experiments/benchmark_runtime_storage/prepare_resume_fixture.py
python3 experiments/benchmark_runtime_storage/run_benchmark.py resume

# Real-dataset scale: needs a local CUDA GPU and the ImageNet-1k
# validation data prepared as described in
# docs/REAL_DATASET_CASE_STUDY_v1.md. Takes on the order of an hour or
# more -- see docs/BENCHMARK_v1.md's "Known limitations".
python3 experiments/benchmark_runtime_storage/run_benchmark.py real-dataset
```

Each subcommand requires a fresh (non-existent) output directory and
refuses to overwrite one that already exists; pass `--output`/
`--output-root` to point at a different location for a repeat run. Pass
`--dry-run` to any subcommand to print the commands it would run without
executing them. Results are written to `results/<scale>.json` (gitignored)
and are what `docs/BENCHMARK_v1.md`'s tables are transcribed from.
