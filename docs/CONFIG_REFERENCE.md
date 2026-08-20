# Configuration Reference (schema 1.0.0)

An SSAT configuration is a YAML mapping with `source` and `adapter` sections plus the audit space. Unknown configuration fields are rejected unless this document explicitly states otherwise. Relative paths are resolved from the YAML file's directory; for an in-memory mapping, supply `base_dir` when it contains relative references.

```yaml
schema_version: 1.0.0
source: {...}
adapter: {...}
regions: [...]
perturbations: [...]
controls: []
runtime: {...}
dump: {...}
dataset_stats: null
skeleton_source: null
```

`regions` and `perturbations` must each contain at least one entry. Region IDs must be unique and match `^[A-Za-z0-9][A-Za-z0-9_.-]*$`.

## Sources

Four source kinds are registered by default: `image_manifest`, `video_manifest`, `imagenet`, and `kinetics400`.

### Image manifest

```yaml
source:
  kind: image_manifest
  manifest: ../data/manifest.json
```

The manifest is a JSON object:

```json
{
  "samples": [
    {"sample_id": "image-001", "path": "images/001.jpg", "gt_label": 3}
  ]
}
```

- `samples` must be non-empty.
- `sample_id` must be non-empty and unique. Planning sorts samples by this ID.
- Relative sample `path` values are resolved from the JSON manifest's directory.
- `gt_label` is an optional integer intended to be a zero-based class index. The clean sanity check counts an out-of-range value as invalid, and label-aware metric computation requires it to index the model output.
- Extra fields in the manifest document and sample entries are ignored in schema 1.0.0.
- The resolved manifest path and its SHA-256 digest are stored as source provenance.

Images are decoded as RGB and enter the core as a `uint8` array with shape `(1, H, W, 3)`.

### Video manifest

```yaml
source:
  kind: video_manifest
  manifest: ../data/video_manifest.json
  num_frames: 16
```

The JSON shape and path rules are the same as for `image_manifest`, but each `path` identifies a video file. `num_frames` is a positive integer (default `16`). Decord samples uniformly spaced frame indices; short clips can repeat indices. A loaded clip has shape `(num_frames, H, W, 3)` and dtype `uint8`.

The region, perturbation, and adapter boundary consistently uses `(T, H, W, C)`. A region mask may be `(H, W)`, broadcast over frames, or `(T, H, W)` for frame-dependent selection.

### ImageNet-style file list

```yaml
source:
  kind: imagenet
  root: /path/to/images
  annotation_file: /path/to/train.txt
```

`annotation_file` contains one whitespace-separated `<relative_path> <label>` pair per non-blank line:

```text
n01440764/image_0001.JPEG 0
n01443537/image_0002.JPEG 1
```

Labels must be non-negative integers and paths must not repeat. The annotation file is source provenance; image paths are resolved under `root`. Raw ImageNet distributions do not necessarily include this normalized file list, so producing it from devkit or validation metadata is the caller's responsibility.

### Kinetics-style CSV

```yaml
source:
  kind: kinetics400
  csv_path: /path/to/kinetics400_train.csv
  video_root: /path/to/clips
  num_frames: 16
  split: train
  extension: mp4
  classes: null
```

The CSV must include `label`, `youtube_id`, `time_start`, and `time_end`; `split` is needed only when filtering by `split`. Extra columns such as `is_cc` are ignored. Files are resolved as:

```text
<video_root>/<youtube_id>_<time_start:06d>_<time_end:06d>.<extension>
```

When `classes` is omitted, distinct label strings from the selected CSV rows are sorted alphabetically and assigned zero-based indices. Supply an ordered `classes` list when a model checkpoint uses a different class order. Duplicate generated clip IDs and labels absent from an explicit class list are rejected.

The ImageNet and Kinetics providers are covered by format-compatible synthetic unit fixtures, but have not been validated against complete production-scale downloads. Confirm that your local distribution follows these exact conventions.

## Preflight area sanity

`ssat estimate` and the preflight stage of `ssat run` perform a bounded preprocessing/effective-area check before audit execution. For each selected region they compute:

```text
intended_ratio = intended_area_px / source_plane_px
effective_ratio = mean_per_frame(transformed_mask_area_px) / model_input_plane_px
relative_deviation = abs(effective_ratio - intended_ratio) / intended_ratio
```

When both ratios are zero, the deviation is zero. A zero intended ratio with a positive effective ratio fails the check. The default inclusive tolerance is `0.05` (5%); set `--max-area-relative-deviation FLOAT` on `estimate` or `run` to change it.

The check evenly selects at most three samples. Within each sample it removes perturbation and seed duplicates using `(region_id, region_instance_id, invert_mask, is_control)`, then evenly selects at most 256 geometries. Truncated coverage is reported as an advisory. Python callers can change these bounds with `EstimateOptions.max_area_sanity_samples` and `EstimateOptions.max_area_sanity_regions_per_sample`.

An excessive deviation or a selected geometry that cannot be loaded, resolved, or transformed produces `FAIL` and makes `run` require explicit confirmation (or `--yes`). This remains an advisory rather than a hard estimation error. If `AdapterSpec.mask_transform_available` is false, the result is `UNAVAILABLE` and does not require confirmation.

### Custom source providers

Custom providers are available through the Python API only. A provider validates a strict provider-specific configuration and returns a `SampleSource` plus file-backed `SourceProvenance`.

```python
from pathlib import Path

from ssat.application import AuditApplication
from ssat.core.config import SourceProvenance
from ssat.core.source import (
    ImageFolderSource,
    SourceProvider,
    SourceProviderConfig,
    default_source_provider_registry,
)
from ssat.utils.io import sha256_file


class MySourceConfig(SourceProviderConfig):
    kind: str = "my_source"
    manifest: Path


class MySourceProvider(SourceProvider):
    name = "my_source"
    config_model = MySourceConfig

    def build(self, config, *, base_dir):
        manifest = (base_dir / config.manifest).resolve(strict=True)
        samples = build_sample_metadata(manifest)
        return ImageFolderSource(samples), SourceProvenance(
            kind=config.kind,
            manifest=manifest,
            manifest_hash=sha256_file(manifest),
        )


registry = default_source_provider_registry()
registry.register(MySourceProvider())
application = AuditApplication(source_registry=registry)
```

## Model adapters

Three adapter providers are built in.

### Torchvision image classification

```yaml
adapter:
  provider: torchvision
  model_name: resnet50
  weights: DEFAULT
  checkpoint: null
  device: auto
  deterministic: true
  max_batch_size: 32
  model_id: null
  model_kwargs: {}
  init_seed: 0
  weights_hash: null
  preprocessing: null
  pipeline_config: null
```

`weights` is a Torchvision weights-enum selector. `null` creates randomly initialized weights using `init_seed` and avoids a model-weight download. `device: auto` lets the adapter select a device; values such as `cpu`, `cuda`, and `cuda:0` are passed explicitly. `max_batch_size` is optional and positive.

### Torchvision video classification

```yaml
adapter:
  provider: torchvision_video
  model_name: r3d_18
  weights: null
  device: auto
  deterministic: true
  max_batch_size: 8
  init_seed: 0
  resize_size: 128
  crop_size: 112
  mean: [0.43216, 0.394666, 0.37645]
  std: [0.22803, 0.22145, 0.216989]
```

The shown preprocessing defaults match the built-in Kinetics preset used for models such as `r3d_18`, `mc3_18`, and `s3d`. Adjust them for architectures with different presets, including MViT and Swin3D variants. The remaining identity, checkpoint, device, and determinism fields have the same meaning as in the image adapter.

### timm image classification

```yaml
adapter:
  provider: timm
  model_name: resnet50
  pretrained: false
  checkpoint: null
  device: auto
  deterministic: true
  max_batch_size: null
  model_id: null
  model_kwargs: {}
  init_seed: 0
  weights_hash: null
```

`pretrained: true` can use the framework cache or network. Use `false` for random initialization without a model-weight download.

### Local checkpoints

All adapter providers accept a trusted local PyTorch checkpoint:

```yaml
adapter:
  provider: torchvision
  model_name: resnet50
  weights: null
  checkpoint:
    path: ./weights/model.pt
    state_dict_key: state_dict
    strict: true
```

For Torchvision, `weights` and `checkpoint` are mutually exclusive. For timm, `pretrained: true` and `checkpoint` are mutually exclusive. SSAT computes and records the checkpoint SHA-256; do not load untrusted pickle-based checkpoint files.

### Declarative preprocessing pipeline

Torchvision image and video adapters accept an MMAction2-style `pipeline_config`. The image adapter cannot combine `pipeline_config` with its legacy `preprocessing` list.

```yaml
adapter:
  provider: torchvision
  model_name: squeezenet1_0
  device: cpu
  pipeline_config:
    - type: Resize
      scale: [-1, 256]
    - type: CenterCrop
      crop_size: 224
    - type: ToFloat
    - type: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - type: FormatShape
      input_format: NCHW
```

Video pipelines should end in `FormatShape` with `input_format: NTCHW`:

```yaml
adapter:
  provider: torchvision_video
  model_name: r3d_18
  device: cpu
  pipeline_config:
    - type: Resize
      scale: [-1, 128]
    - type: CenterCrop
      crop_size: 112
    - type: ToFloat
    - type: Normalize
      mean: [0.43216, 0.394666, 0.37645]
      std: [0.22803, 0.22145, 0.216989]
    - type: FormatShape
      input_format: NTCHW
```

Built-in deterministic transforms are:

| `type` | Contract |
| --- | --- |
| `SampleFrames` | Center-slice the decoded time axis to positive `clip_len`; it does not replace source decoding. |
| `Resize` | `scale` is an integer or `[width, height]`; one pair entry may be `-1` for aspect-preserving short-edge resize. |
| `CenterCrop` | Square positive `crop_size`; zero-pads an undersized input. |
| `TenCrop` | Four corners plus center and horizontal mirrors, expanding the batch by 10; mask geometry is unsupported. |
| `ToFloat` | Convert to float32 and multiply by `scale` (default `1/255`). |
| `Normalize` | Apply channel-wise `(x - mean) / std`. |
| `FormatShape` | Produce `NCHW` for images (`T=1`) or `NTCHW` for video. |

Random crop, random flip, and color-jitter transforms are intentionally absent. Custom transforms can be registered through `BaseTransform`, `TransformRegistry`, and `default_transform_registry()` when directly constructing a Python adapter; the stock CLI uses only the default registry.

## Regions

### Grid

```yaml
regions:
  - region_id: grid_4x4
    kind: grid
    semantic_group: null
    params:
      rows: 4
      cols: 4
```

`rows` and `cols` must be positive integers and are the only accepted parameters. A grid expands into `rows * cols` row-major concrete regions per sample. Integer boundaries cover the complete source frame without overlap.

### Explicit mask

```yaml
regions:
  - region_id: foreground
    kind: explicit
    params: {}
    ref: masks/foreground.png
    ref_hash: null
```

`ref` is required and `params` must be empty. The file must decode as a single-frame, single-channel bitmap with the exact source `(H, W)` dimensions; nonzero pixels select the region. If `ref_hash` is omitted, resolution computes it. If supplied, it must be a 64-character hexadecimal SHA-256 and must match the file. The same explicit mask is applied to every sample, so use this kind only when source geometry is consistent.

### Skeleton body part

```yaml
skeleton_source:
  bbox_data: ../data/skeleton_bbox.json
  bbox_data_hash: null

regions:
  - region_id: occlude_left_arm
    kind: skeleton_parts
    semantic_group: upper_body
    params:
      body_part: left_arm
      bbox_scale: 1.15
```

`body_part` is required and non-empty. `bbox_scale` is a positive finite number (default `1.0`) that scales each box around its center. The referenced JSON has this shape:

```json
{
  "sample-001": {
    "frame_size": [640, 480],
    "parts": {
      "left_arm": [[10, 20, 50, 90], null]
    }
  }
}
```

For each sample:

- `frame_size` is `[width, height]` in decoded source coordinates.
- `parts` is non-empty, and all body-part lists have the same nonzero frame count.
- An entry is `null` for an untracked frame or a finite `[x1, y1, x2, y2]` satisfying `0 <= x1 < x2` and `0 <= y1 < y2`.
- The configured body part must exist for every planned sample, including samples whose media later fails to load, because region expansion precedes pixel decoding.

One family expands into one sample-specific concrete region spanning the time axis. Its instance ID includes `sample_id`. Frame count and decoded frame size are checked when the mask is resolved. Area-matched controls do not support the resulting `(T, H, W)` target mask and fail explicitly.

### Semantic groups

Every public region kind accepts optional `semantic_group`. Reporting and aggregation use it to group geometrically different families under a shared meaning; when omitted they fall back to `region_id`. It is descriptive metadata and does not change mask generation or deterministic item IDs.

`bbox_partition`, `gt_bbox`, and direct `random_area_match` values exist as reserved enum values but are not valid built-in user region implementations in schema 1.0.0. Random area matching is created through `controls` instead.

## Perturbations

```yaml
perturbations:
  - op: gaussian_noise
    params: {sigma: 12.5}
    invert_mask: false
    seed_salts: [0, 1, 2]
```

`seed_salts` defaults to `[0]`, must be non-empty, unique, and non-negative. Each salt creates a separate work item. Stochastic operations receive an item-local RNG derived from the global seed and stable work identity. `invert_mask: true` perturbs the complement of the resolved region.

| `op` | Required `params` | Behavior |
| --- | --- | --- |
| `constant_fill` | `value`: scalar or channel-length list in `[0, 255]` | Fill selected source-space pixels. |
| `mean_fill` | `{}` | Fill with `dataset_stats.channel_mean`. |
| `blur` | positive `sigma` | Gaussian-blur each full frame, then composite selected pixels. |
| `gaussian_noise` | positive `sigma` | Add clipped, rounded Gaussian noise using the item-local RNG. |
| `patch_shuffle` | positive integer `patch_size` | Shuffle complete spatial tiles with one permutation shared across frames; partial edge tiles remain unchanged. |

Only the documented keys are accepted for each operation.

## Area-matched controls

```yaml
controls:
  - match_area_of: grid_4x4
    n_samples: 3
```

`match_area_of` must name a configured region family and `n_samples` must be positive. For every concrete target region and each configured perturbation/seed combination, planning adds `n_samples` controls. At runtime, each control crops the target mask to its bounding box and translates that exact shape, including interior holes, to a uniformly sampled in-bounds position. Area and shape are therefore preserved while location changes. These controls are deterministic under `runtime.global_seed` and currently require a two-dimensional target mask.

## Runtime

```yaml
runtime:
  global_seed: 0
  variants_per_chunk: 16
  target_batch_size: 32
  num_workers: 0
  retry_failed: false
  fail_fast: false
  allow_nondeterministic: false
```

| Field | Default | Meaning |
| --- | ---: | --- |
| `global_seed` | `0` | Unsigned 64-bit root seed for deterministic item RNGs. |
| `variants_per_chunk` | `16` | Positive number of perturbed work items per durable planning chunk. |
| `target_batch_size` | `32` | Positive requested inference batch size, capped by adapter limits and reduced after OOM. |
| `num_workers` | `0` | Non-negative source/preparation worker count. |
| `retry_failed` | `false` | On resume, retry terminal failed items instead of treating them as completed. |
| `fail_fast` | `false` | Abort rather than persist and continue after recoverable item failures. |
| `allow_nondeterministic` | `false` | Permit an adapter that declares itself nondeterministic; otherwise resolution rejects it. |

## Dump policy

```yaml
dump:
  flush_every: 1000
  max_classes_for_full_logits: 10000
```

`flush_every` is a positive record-buffer threshold. `max_classes_for_full_logits` is a positive warning threshold or `null`; it does not truncate logits. SSAT's raw dump stores full clean and perturbed logits together with identities, provenance, timings, terminal statuses, and a resumable item index.

## Dataset statistics

```yaml
dataset_stats:
  channel_mean: [123.675, 116.28, 103.53]
```

Values are finite source-space channel means in `[0, 255]`. `mean_fill` requires them. If omitted when needed, configuration resolution scans successfully loaded source samples and computes the means before planning the run. Providing reviewed statistics avoids that scan and fixes the exact fill value in the configuration.

## Complete examples

Runnable examples are under `configs/examples/`:

- `quickstart.yaml`: CPU image audit over the committed synthetic fixture.
- `video_quickstart.yaml`: grid audit with a Torchvision video model.
- `skeleton_quickstart.yaml`: tracked body-part audit over synthetic video.
- `area_matched_control.yaml`: repeated random area-matched controls.
- `grid_fill.yaml`: two fill strategies and supplied channel statistics.
- `ntu_rgb_d_quickstart.yaml`: requires locally prepared NTU RGB+D data.
- `explicit_noise.yaml`: illustrates explicit-mask hashing but contains a placeholder hash; replace it with the actual mask SHA-256 before use.
