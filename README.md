# SSAT: Spatial Sensitivity Audit Toolkit

[![CI](https://github.com/systemfile36/region-sensitivity/actions/workflows/ci.yml/badge.svg)](https://github.com/systemfile36/region-sensitivity/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

SSAT is a reproducible audit toolkit for measuring how image- and video-classification models respond to spatial perturbations. It applies controlled deletion-style perturbations to configured regions, stores clean and perturbed logits in a resumable Parquet dump, computes vulnerability metrics, evaluates control and stability evidence, and generates an inspectable HTML report.

The command-line interface and Python API share the same `AuditApplication` service, including configuration validation, bounded preflight estimation, output locking, stale-input checks, cancellation, and resume behavior.

## Key capabilities

- Image and uniformly sampled video inputs through JSON manifests.
- Built-in ImageNet-style file-list and Kinetics-style CSV source providers.
- Torchvision image, Torchvision video, and timm model adapters, with optional local checkpoints.
- Grid, explicit-mask, and frame-dependent skeleton body-part regions.
- Constant fill, dataset-mean fill, blur, Gaussian noise, and patch-shuffle perturbations.
- Deterministic stochastic variants and random area-matched controls.
- Durable raw logits and status records, including recoverable per-item failures.
- Built-in item, sample, region, spatial, and class-level metrics.
- Control/stability analysis, reliability grading, HTML reports, and risk-label export.

SSAT is currently alpha software. The ImageNet and Kinetics providers have been tested with format-compatible synthetic fixtures, not full production-scale dataset distributions.

## Quick start

Python 3.11 or later is required. The repository's supported development environment is Docker Compose:

```bash
docker compose up -d --build region-sensitivity-workspace
docker compose exec region-sensitivity-workspace pip install --no-deps -e .
```

Run the committed CPU-only synthetic example. It uses randomly initialized Torchvision weights and requires no model download:

```bash
ssat estimate configs/examples/quickstart.yaml
ssat run configs/examples/quickstart.yaml --output /tmp/ssat-quickstart
ssat inspect /tmp/ssat-quickstart
ssat metrics /tmp/ssat-quickstart
ssat analyze /tmp/ssat-quickstart
ssat report /tmp/ssat-quickstart
```

Open `/tmp/ssat-quickstart/report/report.html` after the final command. The quickstart manifest intentionally contains two missing image paths to exercise failure recording. A complete run therefore writes 20 clean rows and 80 perturbed rows, of which 18 and 72 respectively have status `ok`.

The randomly initialized model is useful for exercising the software only; its metrics are not scientifically meaningful. Use pretrained weights or a trusted local checkpoint and a representative dataset for an actual audit.

The minimal quickstart has one fill strategy, one seed, and no control regions. `ssat analyze` consequently records those comparison families as unavailable/insufficient; use repeated seeds, multiple fill strategies, jittered regions, or `controls` when the corresponding stability evidence is part of the study design.

## Typical workflow

```text
YAML configuration
      |
      v
estimate -> run -> raw Parquet dump -> metrics -> analysis -> HTML/CSV/JSON report
                    ^
                    +------------ inspect / resume
```

1. Describe the source, model adapter, regions, perturbations, controls, and runtime policy in YAML.
2. Use `ssat estimate` to profile a bounded subset and estimate work, storage, and sanity-check results.
3. Use `ssat run` to create or resume the raw dump. Confirmation is requested only when configured limits or sanity criteria require it.
4. Compute metrics and, when the audit design supports comparisons, control/stability analysis.
5. Generate the report and optionally export downstream risk labels.

By default, the workflow creates this durable layout:

```text
DUMP/
  run_manifest.json
  clean/*.parquet
  perturbed/*.parquet
  index/*.parquet
  metrics/{metrics_manifest.json,*.parquet}
  analysis/{analysis_manifest.json,*.parquet,coverage_report.json}
  report/{report.html,report_question_driven.html,report_manifest.json,data/,assets/}
```

The raw dump remains authoritative. Metrics, analyses, reports, and labels are derived artifacts that can be regenerated into alternate directories.

The full CLI surface is:

```text
ssat run CONFIG --output DUMP [--yes] [--minimum-accuracy FLOAT]
ssat estimate CONFIG [--dump DUMP] [--minimum-accuracy FLOAT] [--json]
ssat inspect DUMP [--json]
ssat rebuild-index DUMP
ssat metrics DUMP [--metrics-dir DIR] [--primary-metric NAME] [--json]
ssat analyze DUMP [--metrics-dir DIR] [--analysis-dir DIR] [--primary-metric NAME] [--json]
ssat report DUMP [--metrics-dir DIR] [--analysis-dir DIR] [--report-dir DIR] [--json]
ssat export-labels REPORT_DIR [--output-dir DIR] [--include-non-clean-correct] [--csv] [--json]
```

Use `ssat COMMAND --help` for the authoritative option list.

## Minimal configuration

```yaml
schema_version: 1.0.0
source:
  kind: image_manifest
  manifest: data/manifest.json
adapter:
  provider: torchvision
  model_name: resnet50
  weights: DEFAULT
  device: auto
regions:
  - region_id: grid_4x4
    kind: grid
    params: {rows: 4, cols: 4}
perturbations:
  - op: constant_fill
    params: {value: [0, 0, 0]}
runtime:
  global_seed: 0
  target_batch_size: 32
```

Relative paths are resolved from the configuration file's directory. See the [configuration reference](docs/CONFIG_REFERENCE.md) for all built-in providers and audit settings, and `configs/examples/` for runnable examples.

## Dataset preparation

Manifest sources expect files to be prepared in advance. For NTU RGB+D, the repository includes a reference preprocessing script that converts RGB clips and `.skeleton` files into `video_manifest.json`, `skeleton_bbox.json`, and an executable configuration:

```bash
python scripts/dataset_prep/ntu_rgb_d.py \
  --rgb-root /path/to/nturgb+d_rgb \
  --skeleton-root /path/to/nturgb+d_skeletons \
  --split xsub --num-frames 16 \
  --out /path/to/output_dir

ssat estimate /path/to/output_dir/config.yaml
```

The script is a reference implementation, not a stable public API. Dataset-specific parsing remains outside the core package; the reusable skeleton bounding-box builder is in `ssat.core.region.skeleton_bbox_builder`.

## Python API

```python
from pathlib import Path

from ssat.application import AuditApplication, RunRequest

application = AuditApplication()
with application.prepare_run(
    RunRequest("configs/examples/quickstart.yaml", Path("/tmp/ssat-run"))
) as prepared:
    result = application.execute_run(
        prepared,
        confirmation_granted=True,
    )

print(result.to_dict())
```

See [Application API](docs/APPLICATION_API.md) for UI integration, progress events, cancellation, custom providers, and post-processing request objects.

## Reproducibility and limitations

SSAT records the resolved configuration, source-manifest hash, adapter identity, checkpoint hash when applicable, schema version, code version, timestamps, status counts, and resume events. Work-item IDs and stochastic perturbations are derived deterministically from the resolved audit specification and seeds. A resumed run is accepted only when its resolved configuration, adapter, and code version match.

These controls do not make every third-party model or GPU kernel deterministic. Set `deterministic: true`, keep `runtime.allow_nondeterministic: false`, pin the software environment, and inspect warnings and per-item statuses. A spatial perturbation audit measures model sensitivity under the configured interventions; it does not by itself establish causal feature use, model fairness, robustness to arbitrary distribution shifts, or deployment safety.

## Documentation

- [Installation and deployment](docs/INSTALLATION.md)
- [Configuration reference](docs/CONFIG_REFERENCE.md)
- [Application API](docs/APPLICATION_API.md)
- [Logging policy](docs/LOGGING_POLICY.md)
- [Contributing](CONTRIBUTING.md)

The previous Korean documentation is retained under `docs/internal/`.

## Testing

```bash
docker compose exec region-sensitivity-workspace pytest -q
```

CI also performs a clean package installation and runs the test suite with CPU-only PyTorch wheels.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). If you use SSAT in research, cite the archived release or paper associated with the version you used.

## License

SSAT is released under the [MIT License](LICENSE).
