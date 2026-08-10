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
