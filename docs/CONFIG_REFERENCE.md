# v1 설정 레퍼런스

설정은 `source`, `adapter`, 기존 감사 공간으로 구성된 하나의 YAML mapping입니다.
알 수 없는 필드는 거부됩니다.

## Source

```yaml
source:
  kind: image_manifest
  manifest: ../data/manifest.json
```

manifest는 다음 형식의 JSON object입니다.

```json
{
  "samples": [
    {"sample_id": "image-001", "path": "images/001.jpg", "gt_label": 3}
  ]
}
```

`sample_id`는 고유하고 비어 있지 않아야 합니다. `path`는 manifest 기준이며
`gt_label`은 선택적인 0-based class index입니다. 부가 필드는 허용하지만 v1에서는
무시합니다. manifest 경로와 SHA-256은 run manifest에 기록됩니다.

비디오 입력은 `kind: video_manifest`로 사용합니다. manifest의 `samples` 형식은
image_manifest와 동일하며 `path`가 비디오 파일(mp4 등)을 가리킵니다.

```yaml
source:
  kind: video_manifest
  manifest: ../data/video_manifest.json
  num_frames: 16
```

`num_frames`(기본 16, 양의 정수)은 decord로 클립마다 균등 간격으로 샘플링할
프레임 수입니다. 클립 길이가 `num_frames`보다 짧으면 낮은 인덱스가 반복
샘플링됩니다. 로드된 배열은 `(num_frames, H, W, 3)` uint8이며, 이후의 region·
perturbation·adapter 계층은 이미지(`T=1`)와 동일한 `(T, H, W, C)` 계약을 그대로
사용하므로 별도 처리가 필요 없습니다.

region/perturbation 마스크는 `(H, W)`(전 프레임 공통 브로드캐스트) 또는
`(T, H, W)`(프레임별로 다른 선택)를 모두 지원하도록 코어가 확장되어 있습니다.
`grid`/`explicit`/`random_area_match`는 여전히 `(H, W)`만 반환하고, 프레임마다
달라지는 마스크는 아래 `skeleton_parts` region kind가 제공합니다. 자세한
설계 배경은 `docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md`를 참고하세요.

### Skeleton 부위 추적(skeleton_parts)

사전 계산된 프레임별 신체 부위 bounding box를 따라가며 가리려면 최상위
`skeleton_source`와 `regions[].kind: skeleton_parts`를 함께 설정합니다.

```yaml
skeleton_source:
  bbox_data: ../data/skeleton_bbox.json
  bbox_data_hash: null   # 선택: 지정하면 로드 시 SHA-256이 일치해야 함

regions:
  - region_id: occlude_left_arm
    kind: skeleton_parts
    params:
      body_part: left_arm
      bbox_scale: 1.15    # 선택, 기본 1.0. bbox 중심 기준 확대 배율
```

`skeleton_source.bbox_data`가 가리키는 JSON은 다음 형식입니다(전체 스키마와
검증 규칙은 `ssat/core/region/skeleton_store.py` 참고).

```json
{
  "<sample_id>": {
    "frame_size": [width, height],
    "parts": {
      "<part_name>": [[x1, y1, x2, y2] , null, "... 프레임 수만큼"]
    }
  }
}
```

- `sample_id`는 source manifest의 `sample_id`와 일치해야 합니다. source가
  로드하는 모든 sample(로드에 실패하는 손상 파일 포함)에 대해 항목이 있어야
  합니다 — planning 단계는 픽셀을 읽기 전에 region을 확장하기 때문입니다.
- `frame_size`는 `[width, height]`이며 source가 실제로 디코딩하는 프레임
  크기(예: `video_manifest`의 `num_frames`로 샘플링된 이후 크기가 아니라
  원본 디코딩 크기)와 정확히 일치해야 합니다.
- 각 부위의 리스트 길이(프레임 수)는 그 sample의 모든 부위가 동일해야 하며,
  개별 프레임 값은 추적 실패를 표시하는 `null`이거나 `[x1, y1, x2, y2]`
  (`0 <= x1 < x2`, `0 <= y1 < y2`)입니다.
- 이 JSON 생성(원본 skeleton/joint 파싱, 부위별 bbox 계산)은 SSAT 범위 밖의
  오프라인 전처리이며, `ssat.core.region.skeleton_store`는 이미 계산된
  결과만 로드합니다.

region 1개는 `body_part` 1개에 대응하며, 샘플당 정확히 1개의 concrete
region으로 확장됩니다(grid의 셀별 확장과 달리 부위 하나가 이미 시간축 전체를
포괄하는 단위이기 때문). 여러 부위를 가리려면 `region_id`가 다른 여러
`skeleton_parts` region을 나열하세요. `random_area_match`의 대조군은
`skeleton_parts` target에 대해서는 아직 지원하지 않습니다 — target이
`(T, H, W)`로 resolve되면 명시적인 오류가 발생합니다.

`skeleton_parts`는 `region_instance_id`에 `sample_id`를 포함하므로(사람을
프레임마다 추적해야 하기 때문), 같은 `region_id`라도 샘플마다 다른 concrete
region으로 취급됩니다. 여러 부위(예: `occlude_left_arm`과
`occlude_left_hand`)를 리포트/집계 단계에서 "상체"처럼 하나의 의미 단위로
묶고 싶다면, 아래 `semantic_group`을 사용하세요.

```yaml
regions:
  - region_id: occlude_left_arm
    kind: skeleton_parts
    semantic_group: upper_body
    params: {body_part: left_arm}
  - region_id: occlude_left_hand
    kind: skeleton_parts
    semantic_group: upper_body
    params: {body_part: left_hand}
```

동작하는 전체 예시는 `configs/examples/skeleton_quickstart.yaml`(및 함께
커밋된 `tests/fixtures/synthetic_video/skeleton_bbox.json`)을 참고하세요.

### 내장 데이터셋 source provider (imagenet, kinetics400)

`image_manifest`/`video_manifest` 외에, 잘 알려진 공개 분류 데이터셋을
자체 매니페스트 JSON 없이 바로 가리킬 수 있도록 `imagenet`, `kinetics400`
두 `source.kind`가 기본 레지스트리에 내장되어 있습니다. 두 provider 모두
원본 배포물이 흔히 동반하는 텍스트 포맷(파일 리스트, 주석 CSV)을 그대로
읽어 `SampleMeta` 목록으로 변환할 뿐, 이미지/비디오 디코딩 자체는 기존
`image_manifest`/`video_manifest`와 동일하게 `ImageFolderSource`/
`VideoFolderSource`가 수행합니다.

> **주의:** 두 provider 모두 지금까지 실제 대규모 ImageNet/Kinetics
> 다운로드로 검증된 적이 없습니다 — 아래 문서화된 포맷을 그대로 흉내 낸
> 소규모 합성 fixture(`tests/unit/test_imagenet_source.py`,
> `tests/unit/test_kinetics_source.py`)로만 검증되었습니다. 실제 데이터에
> 적용하기 전에 자신의 배포본이 정말 이 포맷과 일치하는지 직접 확인하세요.

**imagenet** — `<relative_path> <label>` 파일 리스트(한 줄에 하나, 공백
구분, `label`은 0-based 음이 아닌 정수)와 그 경로들이 기준으로 삼는 이미지
루트 디렉터리로 구성됩니다. 이 파일 리스트 포맷은 ImageNet 자체보다 오래된
포맷(Caffe `ImageDataLayer`)이지만 지금도 torchvision의 클래스별 폴더
(`ImageFolder`) 대신 이 방식을 쓰는 ImageNet 파생 학습 파이프라인 다수에서
표준처럼 쓰입니다. 원본 ImageNet 배포(devkit, `LOC_val_solution.csv`,
`valprep.sh` 재배치 등)에서 이 파일을 만드는 과정은 SSAT 범위 밖입니다.

```yaml
source:
  kind: imagenet
  root: /path/to/imagenet/images
  annotation_file: /path/to/train.txt   # "n01440764/img.JPEG 0" 형식, 줄당 1개
```

**kinetics400** — DeepMind Kinetics가 배포하는 주석 CSV(헤더
`label,youtube_id,time_start,time_end,split`, `is_cc` 등 추가 컬럼은 무시)와
클립 디렉터리로 구성됩니다. 클립 파일명은 대다수 다운로드 스크립트(예:
`cvdfoundation/kinetics-dataset`)가 만드는 관례인
`<youtube_id>_<time_start:06d>_<time_end:06d>.<extension>`을 그대로
기대합니다. CSV는 클래스 인덱스를 담지 않으므로 `classes`를 지정하지 않으면
CSV에 등장하는 `label` 문자열을 알파벳순으로 정렬해 인덱스를 매깁니다(공개
Kinetics `label_map.txt`/`mmaction2` 등 통용 관례). 체크포인트가 다른 순서를
기대하면 `classes`로 직접 순서를 고정하세요.

```yaml
source:
  kind: kinetics400
  csv_path: /path/to/kinetics400_train.csv
  video_root: /path/to/clips
  num_frames: 16
  split: train        # 선택, 지정하면 해당 split 행만 사용
  extension: mp4       # 선택, 기본 mp4
  classes: null         # 선택, 지정하면 CSV의 알파벳 정렬 대신 이 순서 사용
```

### 커스텀 source provider 등록

`source.kind`는 기본적으로 `image_manifest`/`video_manifest`/`imagenet`/
`kinetics400` 네 가지가 등록되어 있습니다. 다섯 번째 kind가 필요하면(예:
자체 매니페스트 포맷, 이미 메모리에 있는 목록으로 소스를 구성하는 경우)
`ssat.core.source`의
`SourceProvider`/`SourceProviderRegistry`에 직접 등록해
`AuditApplication(source_registry=...)`로 주입할 수 있습니다 — 어댑터 쪽
`AdapterProvider`/`AdapterProviderRegistry`/`AuditApplication(adapter_registry=...)`와
동일한 패턴입니다.

```python
from pathlib import Path

from ssat.application import AuditApplication, RunRequest
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
    manifest: Path  # provenance용 실제 파일 경로는 여전히 필요


class MySourceProvider(SourceProvider):
    name = "my_source"
    config_model = MySourceConfig

    def build(self, config, *, base_dir):
        samples = ...  # 자체 로직으로 만든 SampleMeta 목록
        manifest_path = (base_dir / config.manifest).resolve(strict=True)
        provenance = SourceProvenance(
            kind=config.kind,
            manifest=manifest_path,
            manifest_hash=sha256_file(manifest_path),
        )
        return ImageFolderSource(samples), provenance


registry = default_source_provider_registry()
registry.register(MySourceProvider())

application = AuditApplication(source_registry=registry)
application.prepare_run(
    RunRequest({"source": {"kind": "my_source", "manifest": "..."}, ...}, Path("/tmp/out"))
)
```

이 확장 지점은 **Python API 전용**입니다 — CLI(`ssat run`/`ssat estimate`)는
항상 기본 레지스트리만 사용하며 커스텀 provider를 등록할 방법이 없습니다
(어댑터 쪽도 CLI에는 동일한 비대칭이 있습니다). 어떤 provider를 등록하든
`SourceProvenance`(실제 파일 경로 + SHA-256)를 채워야 하는 재현성 계약은
완화되지 않습니다 — "이 실행이 정확히 어떤 데이터로 감사됐는지"를 보장하는
핵심 계약이기 때문입니다.

## Adapter

Torchvision:

```yaml
adapter:
  provider: torchvision
  model_name: resnet50
  weights: DEFAULT       # null이면 다운로드 없는 무작위 초기화
  device: auto           # auto, cpu, cuda, cuda:0 등
  deterministic: true
  max_batch_size: 32
  init_seed: 0
  model_kwargs: {}
```

Torchvision video (action recognition, `T>1` 클립 입력):

```yaml
adapter:
  provider: torchvision_video
  model_name: r3d_18      # torchvision.models.video의 모델명
  weights: null            # null이면 다운로드 없는 무작위 초기화
  device: auto
  max_batch_size: 8
  resize_size: 128          # 기본값은 r3d_18/mc3_18/s3d의 Kinetics-400 preset
  crop_size: 112
  mean: [0.43216, 0.394666, 0.37645]
  std: [0.22803, 0.22145, 0.216989]
```

`resize_size`/`crop_size`/`mean`/`std`는 아키텍처별 preprocessing 통계가 다른
모델(예: `mvit_v2_s`, `swin3d_t`)을 사용할 때 조정합니다. 나머지 필드
(`checkpoint`, `init_seed`, `model_kwargs` 등)는 `torchvision` provider와
동일한 의미입니다.

Timm:

```yaml
adapter:
  provider: timm
  model_name: resnet50
  pretrained: false
  device: auto
```

로컬 checkpoint는 두 provider에서 공통으로 지원합니다.

```yaml
  checkpoint:
    path: ./weights/model.pt
    state_dict_key: state_dict  # payload 자체가 state dict면 생략
    strict: true
```

torchvision의 `weights`, timm의 `pretrained: true`와 checkpoint는 상호 배타적입니다.
checkpoint는 신뢰할 수 있는 파일만 사용하며 SHA-256이 자동 기록됩니다.

### 선언적 전처리 파이프라인(pipeline_config)

`torchvision`/`torchvision_video` provider는 `weights` preset이나 (기존) 평평한
`preprocessing` op 목록 대신, MMAction2/MMCV 스타일의 `{type: ..., ...}` 스텝
목록으로 전처리를 선언하는 `pipeline_config`도 받을 수 있습니다. `preprocessing`과
`pipeline_config`는 상호 배타적입니다(config 로드 시점에 검증). 아래 두 예시는
[`test_model_adapter.py`](../tests/unit/test_model_adapter.py)에서 어댑터 생성부터
`predict()`/`transform_mask()`까지 실제로 검증된 값과 동일합니다(문서와 코드가
갈라지지 않도록 값을 일부러 그대로 옮겼습니다 — 실제 감사에서는 `Resize.scale`/
`CenterCrop.crop_size`를 모델에 맞는 값으로 키워서 쓰면 됩니다).

```yaml
adapter:
  provider: torchvision
  model_name: squeezenet1_0
  device: cpu
  pipeline_config:
    - type: Resize
      scale: [-1, 40]     # [w, h], -1인 축은 짧은 변 기준 리사이즈
    - type: CenterCrop
      crop_size: 32
    - type: ToFloat
    - type: FormatShape
      input_format: NCHW   # 이미지 어댑터는 NCHW만 지원(T=1 squeeze)
```

```yaml
adapter:
  provider: torchvision_video
  model_name: r3d_18
  device: cpu
  pipeline_config:
    - type: Resize
      scale: [-1, 40]
    - type: CenterCrop
      crop_size: 32
    - type: ToFloat
    - type: Normalize
      mean: [0.43216, 0.394666, 0.37645]
      std: [0.22803, 0.22145, 0.216989]
    - type: FormatShape
      input_format: NTCHW   # NCHW는 안 됨 -- predict()가 (B,T,C,H,W)를 기대함
```

`torchvision_video`의 `pipeline_config`는 마지막 스텝이 반드시
`FormatShape(input_format: NTCHW)`여야 합니다 — `predict()`가 내부적으로
`(B,T,C,H,W)`를 가정해 고정된 `permute`를 적용하기 때문입니다. `NCHW`로 끝내면
config 로드 시점이 아니라 `predict()` 호출 시점에 명확한 에러로 실패합니다(이
비대칭은 의도적입니다 — 자세한 배경은 구현 계획서의 리스크 표를 참고하세요).

v1이 제공하는 내장 스텝은 다음 7종입니다(모두 결정론적이며, 무작위 증강용
RandomCrop/RandomFlip 같은 스텝은 제공하지 않습니다 — 감사 재현성과 충돌하기
때문입니다):

| `type` | 의미 |
| --- | --- |
| `SampleFrames` | `clip_len`만큼 시간축을 중앙 기준으로 재슬라이스(디코드 후 슬라이스이며, 디코드 시점 프레임 샘플링과는 별개) |
| `Resize` | `scale: [w, h]`(mmaction 관례), 한 축을 `-1`로 두면 짧은 변 기준 리사이즈 |
| `CenterCrop` | 정사각 `crop_size`만큼 중앙 크롭, 입력이 더 작으면 0-padding |
| `TenCrop` | 4개 모서리+중앙 크롭과 각각의 좌우 반전, 총 10개 뷰로 배치를 10배 확장 — 마스크 기하 변환은 지원하지 않음(`mask_transform_available: false`) |
| `ToFloat` | `scale`(기본 `1/255`)을 곱해 float32로 변환 |
| `Normalize` | 채널별 `(x - mean) / std` |
| `FormatShape` | 최종 레이아웃 변환. `NCHW`(이미지, `T=1` 필요) 또는 `NTCHW`(비디오)만 지원 |

### 커스텀 transform 등록

`pipeline_config`의 스텝 타입은 기본적으로 위 7종만 등록되어 있습니다. 새 스텝이
필요하면(예: 커스텀 photometric 변형) `ssat.core.adapter`의
`BaseTransform`/`TransformRegistry`/`default_transform_registry()`에 직접 등록해
`TorchvisionAdapter(pipeline_config=..., transform_registry=...)`로 주입할 수
있습니다 — 위 "커스텀 source provider 등록"과 동일한 패턴입니다.

```python
from ssat.core.adapter import BaseTransform, TorchvisionAdapter, default_transform_registry


class MySharpen(BaseTransform):
    type_name = "MySharpen"
    mask_supported = False  # 광도 변형이라 마스크 지오메트리엔 영향 없음 -- 명시적으로 선언

    def apply_batch(self, batch):
        ...  # 커스텀 numpy 연산


registry = default_transform_registry()
registry.register(MySharpen)

adapter = TorchvisionAdapter(
    "resnet50",
    pipeline_config=[{"type": "Resize", "scale": [-1, 256]}, {"type": "MySharpen"}],
    transform_registry=registry,
)
```

이 확장 지점도 **Python API 전용**입니다 — CLI(`ssat run`/`ssat estimate`)는 항상
`default_transform_registry()`만 사용하며 커스텀 transform을 등록할 방법이
없습니다(source/adapter provider 확장과 동일한 비대칭입니다).

## 감사 공간

- `regions`: `grid`, `explicit`, `skeleton_parts` region family. grid는
  `rows`, `cols`를 사용하고, `skeleton_parts`는 위 "Skeleton 부위 추적" 절을
  참고하며 최상위 `skeleton_source`가 함께 있어야 합니다. 모든 region kind는
  선택적으로 `semantic_group`(region_id와 같은 문자 제약)을 가질 수 있습니다
  — 지정하지 않으면 리포트/집계 단계는 `region_id` 자체를 의미 단위로
  취급합니다. `semantic_group`은 순수 메타데이터로, 마스크 생성이나 결정론적
  item ID 해시에는 전혀 영향을 주지 않습니다(자세한 사용법과 설계 근거는
  `docs/IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md` 참고).
- `perturbations`: `constant_fill`, `mean_fill`, `blur`, `gaussian_noise`,
  `patch_shuffle`과 params, `invert_mask`, `seed_salts`를 정의합니다.
- `controls`: `match_area_of`와 `n_samples`로 random area-matched control을 요청합니다.
- `runtime`: `global_seed`, `variants_per_chunk`, `target_batch_size`, `num_workers`,
  `retry_failed`, `fail_fast`, `allow_nondeterministic`를 설정합니다.
- `dump`: `flush_every`, `max_classes_for_full_logits`를 설정합니다.
- `dataset_stats.channel_mean`: mean fill에 필요한 원본 uint8 채널 평균입니다. 없으면
  ConfigResolver가 source를 스캔해 계산합니다.

상세 region/perturbation 예시는 `configs/examples/`를 참고하세요. 경로는 YAML 파일
디렉터리를 기준으로 해석되며 schema version은 `1.0.0`입니다.
