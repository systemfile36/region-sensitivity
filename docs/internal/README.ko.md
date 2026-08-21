# Spatial Sensitivity Audit Toolkit (SSAT)

SSAT는 이미지·비디오 분류 모델의 예측이 공간 영역별 교란에 얼마나 민감한지
감사하고, 재현 가능한 raw logits dump를 생성하는 도구입니다. CLI, Python 코드,
향후 WebUI가 같은 `AuditApplication` 계층을 사용합니다.

## 빠른 시작

Dev Container 또는 Compose 워크스페이스를 빌드한 뒤 다음을 실행합니다.

```bash
pip install --no-deps -e .
ssat estimate configs/examples/quickstart.yaml
ssat run configs/examples/quickstart.yaml --output /tmp/ssat-quickstart
ssat inspect /tmp/ssat-quickstart
```

quickstart는 committed synthetic fixture와 CPU torchvision 모델의 무작위 초기화
가중치를 사용하므로 네트워크 다운로드가 없습니다. 비디오 입력(`source.kind:
video_manifest`)과 action recognition 모델(`adapter.provider: torchvision_video`)도
같은 방식으로 동작하며 `configs/examples/video_quickstart.yaml`을 참고하세요.

Python에서도 같은 실행 정책을 사용할 수 있습니다.

```python
from pathlib import Path
from ssat.application import AuditApplication, RunRequest

application = AuditApplication()
with application.prepare_run(
    RunRequest("configs/examples/quickstart.yaml", Path("/tmp/ssat-run"))
) as prepared:
    # 실제 UI에서는 confirmation_required일 때 사용자 승인을 받습니다.
    result = application.execute_run(
        prepared,
        confirmation_granted=True,
    )
print(result.to_dict())
```

자세한 내용은 [설치 문서](docs/INSTALLATION.md),
[설정 레퍼런스](docs/CONFIG_REFERENCE.md),
[애플리케이션/WebUI 연동](docs/APPLICATION_API.md)을 참고하세요.

## 내장 지원 데이터셋 (source provider)

아래 데이터셋은 오프라인 전처리 스크립트 없이 `source.kind`만 지정하면
바로 감사할 수 있도록 SSAT 기본 레지스트리에 provider가 내장돼 있습니다.

- **ImageNet** (`source.kind: imagenet`): 파일 리스트(`<relative_path>
  <label>`)와 이미지 루트 디렉터리.
- **Kinetics-400** (`source.kind: kinetics400`): DeepMind Kinetics 주석
  CSV(`label,youtube_id,time_start,time_end,split`)와 클립 디렉터리.

두 provider의 정확한 입력 포맷은 [설정 레퍼런스](docs/CONFIG_REFERENCE.md)의
"내장 데이터셋 source provider" 절을 참고하세요.

> **검증 상태:** NTU60 XSub의 `video_manifest`와 native TSM 경로는 Phase 3
> 준비 및 smoke test를 완료했습니다. ImageNet 사례는 다운로드 후 전체 audit
> summary가 생성되어야 검증 완료로 표시하며, Kinetics는 여전히 포맷 호환 합성
> fixture에서만 검증되었습니다.

## 지원 데이터셋 레시피

`ssat estimate`/`ssat run`은 `source.kind: image_manifest`/`video_manifest`
매니페스트와 (skeleton 부위 추적을 쓴다면) `skeleton_source.bbox_data` JSON이
이미 만들어져 있다는 전제로 동작합니다. 원본 데이터셋 파일(비디오, `.skeleton`
파일 등)에서 이 형식을 만드는 전처리는 데이터셋마다 근본적으로 달라 SSAT
패키지 자체가 대신 해주지 않지만, 이 저장소가 다뤄본 대표 데이터셋에 대해서는
`scripts/dataset_prep/` 아래 예시 스크립트를 제공합니다.

- **NTU-RGB+D**: [`scripts/dataset_prep/ntu_rgb_d.py`](scripts/dataset_prep/ntu_rgb_d.py)

  ```bash
  python scripts/dataset_prep/ntu_rgb_d.py \
    --rgb-root /path/to/nturgb+d_rgb \
    --skeleton-root /path/to/nturgb+d_skeletons \
    --annotation-file /path/to/ntu60_xsub_test.txt \
    --samples-per-class 20 --num-frames 8 --sampling segment_center \
    --out /path/to/output_dir

  ssat estimate /path/to/output_dir/config.yaml
  ```

  `.skeleton` 파일과 실제 RGB 프레임 해상도로부터 `video_manifest.json`,
  `skeleton_bbox.json`, 바로 실행 가능한 `config.yaml`을 생성합니다. skeleton
  파일이 없거나 파싱에 실패한 샘플은 건너뛰고 이유를 stderr에 남깁니다.

> **이 스크립트는 참고 구현이며 SSAT의 안정된 API가 아닙니다.** 다른
> 데이터셋을 감사하려면 이 파일을 복사해 원본 포맷 파싱 부분만 새로 작성하되,
> 관절 좌표를 `skeleton_bbox.json`으로 바꾸는 부분(`ssat.core.region.
> skeleton_bbox_builder`)은 관절 세트에 무관하게 설계돼 있어 그대로 재사용할
> 수 있습니다.

## 주요 명령

```text
ssat run CONFIG --output DUMP [--yes] [--minimum-accuracy FLOAT]
ssat estimate CONFIG [--dump DUMP] [--minimum-accuracy FLOAT] [--json]
ssat rebuild-index DUMP
ssat inspect DUMP [--json]
```

`run`은 항상 bounded preflight를 수행합니다. 한도나 sanity 기준을 넘을 때만 확인하며
`--yes`는 확인 질문만 건너뜁니다. 기존 유효 dump를 출력으로 지정하면 자동으로
재개합니다.
