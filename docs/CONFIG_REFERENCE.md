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
다만 v1에 내장된 region kind(`grid`, `explicit`, `random_area_match`)는 아직
`(H, W)`만 반환합니다 — 실제로 프레임마다 달라지는 마스크(예: skeleton 부위
추적)를 만들어내는 provider는 향후 확장입니다. 자세한 내용은
`docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md`를 참고하세요.

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

## 감사 공간

- `regions`: `grid` 또는 `explicit` region family. grid는 `rows`, `cols`를 사용합니다.
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
