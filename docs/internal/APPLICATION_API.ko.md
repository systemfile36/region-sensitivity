# 애플리케이션 API와 WebUI 연동

`ssat.application`은 Typer, stdin/stdout, prompt에 의존하지 않습니다. CLI와 WebUI는
같은 두 단계 실행 흐름을 사용합니다.

```python
from pathlib import Path
from ssat.application import AuditApplication, RunRequest

application = AuditApplication()
prepared = application.prepare_run(
    RunRequest("audit.yaml", Path("/dumps/run-001"))
)
try:
    # WebUI는 prepared.estimate.to_dict()를 표시하고 사용자의 결정을 받습니다.
    result = application.execute_run(
        prepared,
        confirmation_granted=user_approved,
        event_sink=publish_event,
        cancellation=job_cancellation_token,
    )
finally:
    prepared.close()
```

`PreparedRun`은 process-local one-shot 세션입니다. 실행 직전에 config/source/dump가
preflight 이후 바뀌지 않았는지 검사하며, 같은 dump에 대한 동시 writer를 거부합니다.
분산 worker에서는 향후 직렬화 가능한 job specification을 queue에 저장하고 각 worker가
자체 `PreparedRun`을 만드는 방식으로 확장합니다.

`ApplicationEvent`는 `kind`, `phase`, 선택적 `completed`, `total`만 전달하며 logits나
전체 설정은 포함하지 않습니다. `CancellationToken.cancel()`은 다음 안전한 runtime
경계에서 실행을 멈추고 이미 작성된 fragment를 flush하므로 같은 설정으로 재개할 수
있습니다.

## 사용자 provider

`AdapterProvider` 구현을 registry에 명시적으로 등록합니다. 자동 module/entry-point
검색은 수행하지 않습니다.

```python
from ssat.cli import create_app
from ssat.core.adapter import default_adapter_provider_registry

registry = default_adapter_provider_registry()
registry.register(MyProvider())
app = create_app(registry)
app()
```

WebUI도 같은 registry를 `AuditApplication(registry)`에 주입합니다.
