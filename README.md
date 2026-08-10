# Spatial Sensitivity Audit Toolkit (SSAT)

SSAT는 이미지 분류 모델의 예측이 공간 영역별 교란에 얼마나 민감한지 감사하고,
재현 가능한 raw logits dump를 생성하는 도구입니다. CLI, Python 코드, 향후 WebUI가
같은 `AuditApplication` 계층을 사용합니다.

## 빠른 시작

Dev Container 또는 Compose 워크스페이스를 빌드한 뒤 다음을 실행합니다.

```bash
pip install --no-deps -e .
ssat estimate configs/examples/quickstart.yaml
ssat run configs/examples/quickstart.yaml --output /tmp/ssat-quickstart
ssat inspect /tmp/ssat-quickstart
```

quickstart는 committed synthetic fixture와 CPU torchvision 모델의 무작위 초기화
가중치를 사용하므로 네트워크 다운로드가 없습니다.

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
