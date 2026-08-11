# 구현 계획서 (v1 지표 엔진)
## Spatial Sensitivity Audit Toolkit — Metrics Engine

> 본 계획서는 [METRIC_ENGINE_DESIGN_v1.md](METRIC_ENGINE_DESIGN_v1.md)의 설계를 현재 저장소 구현 상태와 대조하여 작성한 실행 계획이며, 방향과 틀을 정하기 위한 것이다. 세부 사항은 구현 과정에서 조정한다.
> 전제: 코어 v1([IMPL_PLAN_CORE_v1.md](IMPL_PLAN_CORE_v1.md))이 이미 구현되어 있다. 본 문서는 그 결과물(`DumpReader`, `DumpWriter`, `RegionResolver`, `ResolvedConfig` 등)을 재사용하는 후속 단계를 다룬다.
> 패키지 위치: 지표 엔진은 `ssat/metrics/`라는 **신규 최상위 패키지**로 구현한다(코어와 형제 관계). 근거는 §3.2 참조.

---

## 1. 현재 구현 상태 대비 격차

설계서(N0~N5)와 현재 저장소를 대조한 결과, **지표 엔진은 아직 한 줄도 구현되어 있지 않다.** `ssat/` 아래에는 `core/`, `application/`, `utils/`만 존재하며 `metrics/`, `analysis/`, `report/`에 해당하는 디렉터리나 테스트는 없다(`tests/unit/`, `tests/integration/`을 통틀어 metrics 관련 테스트 0건).

다만 코어 v1이 지표 엔진이 재사용할 수 있는 자산을 이미 여럿 제공한다. 아래 표는 설계서의 각 모듈이 놓인 실제 상태를 정리한다.

| 설계 항목 | 현재 상태 | 비고 |
|---|---|---|
| N0. DumpReader | **부분 재사용 가능.** `ssat/core/dump/reader.py`에 스키마 검증·chunk 스트리밍 `DumpReader`가 이미 존재 | 개념 인터페이스(`clean()`/`items()`/`joined()`/`summary()`)와 다름. `joined()`·`summary()`는 없음 → 지표 엔진이 이를 감싸는 **상위 계약을 새로 작성**해야 한다(§5 단계 1) |
| N1. OutputNormalizer | **미구현** | `AdapterSpec.output_kind`가 v1 코어에서 `Literal["logits"]`로 **고정**되어 있다(`ssat/core/adapter/types.py:73`). 설계서가 전제하는 `probs` 분기는 코어가 확장되기 전까지 **실도달 불가능한 방어 코드**로만 존재하게 된다 |
| N2. MetricRegistry (1·2순위 지표) | **미구현** | 플러그인 인터페이스, built-in 지표 모두 신규 작성 |
| N3. Aggregator | **미구현** | — |
| N4. MetricsStore | **부분 재사용 가능.** parquet atomic write·manifest 패턴은 `ssat/core/dump/manifest.py`, `ssat/utils/io.py`(`write_json_atomic`, `sha256_file`)에 이미 구현되어 있어 그대로 재사용 | 지표 전용 스키마·manifest는 신규 |
| N5. DebugViz | **미구현** | `matplotlib`, `pillow`는 `requirements.txt`에 이미 포함되어 있어 새 의존성 없이 시작 가능 |
| 검증 실험 (L1/L2/L3) | **미구현** | fixture, 실험 스크립트 모두 없음. 코어의 `tests/fixtures/synthetic_classification/`(이미지 10장 + 손상 파일 2개 + manifest)은 L2/L3 재사용 후보 |

**결론.** 이 계획서는 설계서 §1~§6 전체를 다루는 **신규 구현 계획**이며, "이미 있는 것을 고친다"가 아니라 "코어가 이미 제공하는 계약 위에 새 패키지를 쌓는다"는 성격을 가진다.

---

## 2. 기술 스택과 의존성 방침

### 2.1 신규 의존성 없음

지표 엔진에 필요한 라이브러리는 코어가 이미 `requirements.txt`에 확보해 두었다.

| 용도 | 라이브러리 | 상태 |
|---|---|---|
| 결합·집계 프레임 | pandas | 이미 포함 (코어는 dump에 pyarrow만 쓰고 pandas는 아직 미사용이지만 의존성엔 있음) |
| 저장 포맷 | parquet (pyarrow) | 이미 포함, 코어 dump와 동일 |
| 시각화 | matplotlib, pillow | 이미 포함 |

**결정.** `ssat/metrics/`는 pandas `DataFrame`을 `JoinedFrame`/`ItemMetrics` 등의 내부 표현으로 사용한다. `requirements.txt`, `scripts/install_deps.sh`, `.devcontainer/Dockerfile`은 변경하지 않는다.

### 2.2 새 CLI 표면 (v1 범위 밖으로 미뤘다가 후속 도입함)

설계서는 CLI 명령을 전혀 언급하지 않는다(`ssat run`/`estimate`/`inspect`만 코어 CLI에 존재). 이 계획 역시 처음에는 지표 엔진을 **라이브러리 API + 스크립트**로만 제공하고 `ssat metrics ...` 같은 Typer 명령 추가를 v1 범위 밖으로 두었다(§8 잔여 결정 사항에 "미정"으로 기록).

**후속 결정(단계 9 완료 후).** `experiments/synthetic_shortcut/run_audit.py`와 `tests/fixtures/synthetic_dump_builder.py`가 각자 `DumpHandle`을 직접 열어 동일한 지표 계산 로직을 중복 구현하고 있는 것이 확인되어, 이 CLI 표면을 실제로 추가했다. 코어의 단계 10과 같은 방식(Application 계층에 메서드를 먼저 추가하고 CLI는 그 위의 얇은 Typer 래퍼로만 구현)을 그대로 따랐다:

- `AuditApplication.compute_metrics(ComputeMetricsRequest) -> ComputeMetricsResult`(`ssat/application/application.py`) — 기존 `inspect`/`rebuild_index`처럼 이미 존재하는 dump 디렉터리 하나를 입력으로 받아, 지표 계산·저장 책임을 dump 생성(`execute_run`)과 분리했다. 항상 `default_metric_registry()`(내장 9개 전부)를 쓴다 — 지표 선택 플래그는 이번에도 범위에 넣지 않았다.
- CLI: `ssat metrics <dump> [--metrics-dir DIR] [--primary-metric NAME] [--json]`(`ssat/cli.py`) — `inspect`/`rebuild-index`와 동일한 모양.
- 테스트: `tests/integration/test_application_api.py`, `tests/integration/test_cli.py`에 Application/CLI 경로로 지표를 계산하는 케이스를 추가(기존에는 두 파일 모두 지표 엔진을 전혀 다루지 않았다).

---

## 3. 디렉터리 구조

```
ssat/
├── core/                              # 기존 (변경 없음)
├── application/                       # 기존 (변경 없음)
├── metrics/                           # ← v1 지표 엔진 구현 범위 (신규)
│   ├── __init__.py
│   ├── types.py                       # ItemMetrics·SampleMetrics·RegionMetrics·
│   │                                  #   ClassMetrics·SpatialProfile·ExclusionReason
│   ├── errors.py                      # MetricsError, MetricsSchemaError 등
│   ├── dump_reader.py                 # N0  DumpHandle / JoinedFrame
│   ├── normalize.py                   # N1  OutputNormalizer
│   ├── registry.py                    # N2  Metric protocol + MetricRegistry
│   ├── builtin_metrics/               # N2  1·2순위 built-in 지표 구현체
│   │   ├── __init__.py
│   │   ├── flips.py                   # 오류 전환 (binary)
│   │   └── continuous.py              # 연속 변화량
│   ├── aggregate.py                   # N3  Aggregator
│   ├── store.py                       # N4  MetricsStore
│   └── viz/                           # N5  DebugViz
│       ├── __init__.py
│       ├── mask_check.py              # V1
│       ├── heatmap.py                 # V2
│       └── ranking.py                 # V3
├── utils/                             # 기존 (변경 없음)
└── ...
tests/
├── unit/
│   ├── test_metrics_dump_reader.py
│   ├── test_metrics_normalize.py
│   ├── test_metrics_registry.py
│   ├── test_metrics_flip_metrics.py
│   ├── test_metrics_continuous_metrics.py
│   ├── test_metrics_aggregate.py
│   └── test_metrics_store.py
├── integration/
│   ├── test_metrics_synthetic_dump.py  # L2: 코어 미실행, 합성 dump 직접 주입
│   └── test_debug_viz.py
├── fixtures/
│   ├── synthetic_classification/      # 기존, 코어와 공유
│   └── synthetic_dump_builder.py      # L2 전용: DumpWriter를 직접 호출해
│                                       #   합성 clean/perturbed 레코드를 굽는 헬퍼
└── ...
experiments/
└── synthetic_shortcut/                 # L3 (단계 9에서 산출)
```

### 3.1 구조 설계 의도

**`metrics/`가 N0~N5에 1:1 대응한다.** 코어의 `core/` 하위가 M0~M11에 대응한 것과 동일한 원칙이다. 설계 문서 절 번호와 코드 파일을 오가기 쉽다.

**`registry.py`와 `builtin_metrics/`를 분리한다.** `MetricRegistry`는 플러그인 등록·순회·부호 정규화만 담당하는 얇은 계층이고, 실제 지표 계산식은 `builtin_metrics/`에 있다. 3순위 이후 지표(KL/JS distance, 엔트로피 변화, 대조군 상대 지표, 안정성 지표)를 추가할 때 `builtin_metrics/`에 파일만 늘리면 되고 `registry.py`는 손대지 않는다.

**`viz/`를 세 파일로 나눈다.** V1(마스크 검증)·V2(히트맵)·V3(랭킹)는 서로 다른 실패 모드를 갖는다. 하나가 깨져도 나머지 디버깅 도구는 계속 쓸 수 있어야 한다.

**`dump_reader.py`가 지표 엔진 내부의 유일한 코어 dump 접근 지점이다.** 코어의 `dump/reader.py`가 "코어와 후단의 유일한 계약 지점"(§CORE_DESIGN)이라면, `metrics/dump_reader.py`는 그 계약을 소비하는 **유일한 지점**이다. `registry.py`·`aggregate.py`·`store.py`·`viz/` 중 어느 것도 `ssat.core.dump`를 직접 import하지 않고, `metrics.dump_reader`가 만든 `JoinedFrame`/`DumpHandle.summary()`만 다룬다. 이렇게 하면 코어 dump 스키마가 바뀌어도 파급 범위가 `dump_reader.py` 한 파일로 좁혀진다.

### 3.2 패키지 위치 근거 — `core/` 밖에 둔다

코어의 [IMPL_PLAN_CORE_v1.md](IMPL_PLAN_CORE_v1.md) §2.1은 "`metrics/`, `analysis/`, `report/`를 지금 비워두되 자리는 만든다"고 서술했으나 실제로는 아직 생성되지 않았다. 이번 구현에서 `ssat/metrics/`를 신규 최상위 패키지로 만들어 그 취지를 완성한다.

- 코어의 §2.2 의존 방향 규칙은 명시적으로 단방향(`config → adapter/source/plan/perturb`, `runtime → plan/source/region/perturb/adapter/dump` 등)이며 **후단으로의 역참조는 규칙 위반**이다. `core/metrics/`처럼 core 하위에 두면 이 규칙을 "core 내부 규칙"과 "core→후단 규칙"으로 이중 관리해야 한다.
- 코어는 스스로를 "프레임워크 비의존 실행 코어"로 좁게 정의했다(IMPL_PLAN_CORE_v1 §1.1). 지표 엔진은 실행이 끝난 뒤의 **해석 계층**이라는 다른 책임을 가지므로, 디렉터리 경계를 책임 경계와 일치시킨다.
- `dump/reader.py`가 "코어와 후단의 유일한 계약 지점"이라는 기존 설계 원칙이 디렉터리 구조로도 강제된다: `core.*` 어디에서도 `ssat.metrics`를 import할 수 없고, `ssat.metrics.*`만 `core.dump`(및 §3.3의 명시적 예외)를 import한다.

### 3.3 의존 방향 규칙

```
metrics.types            → (없음)
metrics.errors            → (없음)
metrics.dump_reader        → core.dump, core.types, utils
metrics.normalize           → metrics.types, core.adapter.types (AdapterSpec.output_kind만)
metrics.registry            → metrics.types, metrics.normalize
metrics.builtin_metrics      → metrics.registry, metrics.types
metrics.aggregate             → metrics.types
metrics.store                 → metrics.types, utils
metrics.viz                    → metrics.types, core.region (RegionResolver 재호출), core.config (contracts)
```

`core/*` → `ssat.metrics` 역참조는 전면 금지한다(코어 §2.2 규칙의 확장). `metrics.viz`가 `core.region`을 직접 import하는 것은 **의도된 유일한 예외**다(§5 단계 7에서 근거 설명). CI에서 import-linter로 강제하는 방식은 코어와 동일하게 따른다.

---

## 4. 개발 환경

기존 Dev Container / Docker Compose 워크스페이스 이미지를 그대로 사용한다. §2.1에서 확인한 대로 신규 시스템·Python 의존성이 없으므로 `.devcontainer/Dockerfile`, `compose.yaml`, `scripts/install_deps.sh` 변경이 필요 없다. 테스트는 기존과 동일하게 컨테이너 안에서 `pytest`로 실행하며, `.github/workflows/ci.yml`의 단일 `pytest -q` 잡에 지표 엔진 테스트가 자연히 합류한다(L3 실험은 §5 단계 9, §6에서 별도로 다룬다).

---

## 5. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행하며, 조건 미충족 시 다음 단계로 넘어가지 않는다(코어와 동일한 원칙).

---

### 단계 0. 스캐폴딩 + 계약 타입

> **가장 먼저 하는 이유:** 코어의 단계 1과 같은 이유다. 이후 모든 모듈이 이 타입을 참조한다.

**작업.**
- `metrics/types.py`: `ItemMetrics`(item_id, sample_id, metric_name, value_clean, value_perturbed, degradation, available, excluded_reason), `SampleMetrics`, `RegionMetrics`, `ClassMetrics`, `SpatialProfile`, `ExclusionReason` enum을 dataclass로 정의
- `metrics/errors.py`: `MetricsError`, `MetricsSchemaError`(코어 `DumpSchemaError`와 동일한 성격), `MetricsCorruptionError`
- `metrics/__init__.py`에 공개 API 확정, 각 빈 모듈(`registry.py`, `aggregate.py`, `store.py`, `viz/*`)에 인터페이스 stub만 배치

**설계 결정.** 3순위 이후 지표(KL/JS distance, 엔트로피 변화, 대조군 상대 지표, 안정성 지표)는 v1 전제(설계서 §0)대로 **인터페이스만** 확보하고 구현체는 만들지 않는다.

**테스트.**
- `python -c "import ssat.metrics"` 성공
- 각 dataclass의 필수/선택 필드 유효성 검증(음수 area, 빈 문자열 등 코어 스타일 검증과 동일한 엄격도)

**성공 조건.**
- 패키지 import 성공
- 타입 유효성 단위 테스트 통과

---

### 단계 1. N0. DumpReader (`DumpHandle` / `JoinedFrame`)

**작업.** `metrics/dump_reader.py`의 `DumpHandle`이 `core.dump.DumpReader`를 감싼다.
- `DumpHandle.manifest` → `core.dump.DumpReader.read_manifest()` 그대로 노출
- `DumpHandle.clean()` / `.items()` → `read_clean()`/`read_perturbed()`를 pandas `DataFrame`으로 변환
- `DumpHandle.joined()` → `sample_id` 기준 병합, clean 컬럼에 접미사(`_clean`) 부여. 설계서 §N0의 status 처리 규칙 5개 조합을 그대로 구현:

  | 조합 | 처리 |
  |---|---|
  | clean=ok, perturbed=ok | 정상 계산 |
  | clean=ok, perturbed≠ok | 지표 null, `excluded_reason` 기록 |
  | clean≠ok | 해당 샘플 전체 제외, 별도 집계에 보고 |
  | clean 레코드 없음(고아 perturbed row) | `MetricsCorruptionError` — dump 무결성 문제이지 지표 엔진이 조용히 넘길 결측이 아니다 |

- `DumpHandle.summary()` → status별 개수, 결측/제외 현황 dict
- 스키마 버전 거부는 `core.dump.DumpReader` 생성자가 이미 수행하므로 별도 구현 없이 그대로 전파되는 것을 회귀 테스트로 고정한다(코어 쪽 동작이 바뀌면 즉시 발견되도록)

**테스트(L1 대응).**
- status 조합 5가지 표 전부 검증
- `summary()`가 status별 개수·결측 현황을 정확히 보고
- 코어의 committed fixture(`tests/fixtures/synthetic_classification/`)로 실제 `DumpWriter` 실행 결과를 만들어 `joined()`의 행 수·컬럼이 기대와 일치
- 스키마 버전 불일치 시 `core.dump`가 던지는 예외가 `DumpHandle` 생성 시점에 그대로 전파되는지 확인

**성공 조건.**
- 5가지 status 조합 테스트 전부 통과
- `summary()` 수치가 수동 계산과 일치

---

### 단계 2. N1. OutputNormalizer

**작업.** `metrics/normalize.py`가 `prob`/`logit`/`top1_index`/`top1_prob`/`gt_prob`/`gt_logit`/`gt_rank`/`margin`/`entropy`를 파생한다. clean·perturbed에 동일 함수를 적용한다. 수치 안정성을 위해 softmax는 log-sum-exp 기반으로 구현한다.

**v1 명확화.** `AdapterSpec.output_kind`는 코어 v1에서 `Literal["logits"]`로 고정되어 있다(`ssat/core/adapter/types.py:73`, `IMPL_PLAN_CORE_v1.md` §1 v1 계약 명확화). 즉 설계서가 전제하는 "`output_kind=probs`일 때 logit 복원 불가 → `available=False`" 분기는 **현재 코어가 만들어낼 수 없는 경로**다. 이 단계에서는 그 분기를 방어적으로 구현하되(코어가 후속 버전에서 `probs`를 지원하면 즉시 동작하도록), 실제 도달 가능성이 없다는 점을 코드 주석과 테스트 모두에 명시한다.

**잔여 결정 처리.** 설계서 §5 "확률 입력에서 logit 복원 허용 여부"는 이 단계에서 **"현재는 허용하지 않음(코어가 지원하지 않으므로 논의 자체가 유보)"**으로 확정한다.

**테스트.**
- 손으로 계산 가능한 고정 로짓 벡터로 각 파생값 비교(설계서 §3.2)
- `gt_rank`가 1-indexed임을 확인
- `margin` 부호(정답이 1위면 양수)
- `output_kind=probs`를 모의(mock) `AdapterSpec`으로 강제 주입했을 때 logit 파생 지표가 전부 `available=False`

**성공 조건.**
- 파생값 손계산 전부 일치
- probs 분기 unavailable 동작 확인(모의 입력 기준)

---

### 단계 3. Metric 인터페이스 + 1순위(오류 전환) 지표

**작업.** `metrics/registry.py`에 `Metric` Protocol(`name`, `requires`, `higher_is_better`, `kind`, `available_when`, `compute`)과 `MetricRegistry`(등록·순회·부호 정규화)를 구현한다. `builtin_metrics/flips.py`에 `flip_correct_to_wrong`/`flip_wrong_to_correct`/`pred_changed`/`topk_exit`을 구현한다.

**binary 지표의 degradation 인코딩.** 이진 지표는 `value_clean`/`value_perturbed`를 0/1로, `degradation`을 전환 발생 여부(0/1)로 인코딩해 `ItemMetrics`의 long-form 스키마(§단계 0)를 continuous 지표와 통일한다.

**필수 보고 규칙.** clean에서 이미 오답인 샘플과 정답인 샘플을 반드시 분리 집계한다(`ItemMetrics`에 `clean_correct` 플래그 동반).

**잔여 결정.** `topk_exit`의 k 기본값을 이 단계에서 확정한다(설계서 §5). 별도 요청이 없으면 **k=5**를 기본값으로 제안한다(ImageNet류 top-5 관례와의 정합성, 클래스 수가 5 미만인 데이터셋은 `min(5, num_classes)`로 자동 축소).

**테스트(L1).**
- 각 지표 손계산 일치
- 부호 정규화 방향
- clean=오답 샘플에서 `flip_correct_to_wrong=False` 고정
- 결측(status≠ok) 시 null 반환 및 `excluded_reason` 기록

**성공 조건.**
- L1 1순위 지표 테스트 전부 통과

---

### 단계 4. 2순위(연속 변화량) 지표

**작업.** `builtin_metrics/continuous.py`에 `gt_prob_drop`/`gt_logit_drop`/`margin_drop`(higher_is_better=True), `loss_increase`/`gt_rank_worsening`(higher_is_better=False)을 구현한다.

**테스트(L1 확장).**
- 각 지표 손계산 일치
- `higher_is_better` 조합별 degradation 부호 검증

**성공 조건.**
- L1 확장 테스트 통과, 부호 정규화 검증 완료

---

### 단계 5. N3. Aggregator

**작업.** `metrics/aggregate.py`가 `ItemMetrics`를 `SampleMetrics`/`RegionMetrics`/`ClassMetrics`/`SpatialProfile`로 집계한다.
- `clean_correct`로 층화, `is_control` 아이템은 기본 집계에서 분리
- region 집계에 `intended_area_px`·`effective_area_px`·`n_valid` 병기
- `vulnerability_score`는 설정된 `primary_metric`(기본 `margin_drop`)의 유효 아이템 degradation 단순 평균

**L2 테스트 인프라 신설.** `tests/fixtures/synthetic_dump_builder.py`를 추가한다. 설계서 §3.3이 요구하는 "**코어를 실행하지 않고** 의도적으로 구성한 dump"를 정확히 구현하기 위해, `core.plan.PlanBuilder`/`core.runtime.run_audit`/어댑터는 전혀 사용하지 않고 `core.dump.DumpWriter` + `CleanDumpRecord`/`PerturbedDumpRecord`를 **직접 호출**해 원하는 로짓 패턴을 가진 합성 dump를 만든다. 이렇게 하면 L2가 실패했을 때 코어 실행 경로(PlanBuilder·runtime·adapter)는 확실히 배제할 수 있다(설계서 §3.3 설계 의도).

**테스트(L2, 설계서 §3.3 표 5개 시나리오 전부).**

| 시나리오 | 구성 | 기대 결과 |
|---|---|---|
| 완전 무감각 | 모든 perturbed 로짓 = clean 로짓 | 모든 degradation = 0, flip_rate = 0 |
| 단일 영역 취약 | region_3만 로짓 급변, 나머지 동일 | region_3만 높은 degradation |
| 특정 샘플 취약 | sample_A의 전 region 급변 | sample_A만 높은 vulnerability_score |
| 결측 혼재 | 일부 status=predict_failed | 해당 아이템 제외, n_valid 감소, 집계에 반영 |
| clean 실패 | 일부 clean status≠ok | 해당 샘플 전체 제외 및 보고 |

**성공 조건.**
- L2 5개 시나리오 전부 통과 — 이 통과 이후 발생하는 실패는 코어 쪽으로 원인을 좁힐 수 있다(설계서 §3.3)

---

### 단계 6. N4. MetricsStore

**작업.** `metrics/store.py`가 `item_metrics.parquet`/`sample_metrics.parquet`/`region_metrics.parquet`/`class_metrics.parquet`/`spatial_profile.parquet`/`metrics_manifest.json`을 `<run_dir>/metrics/`에 저장한다. `core.dump.manifest`의 atomic write·fsync 패턴과 `utils.io.sha256_file`을 재사용해 `source_run_manifest_hash`를 계산한다.

**잔여 결정.** `SpatialProfile`의 저장 형식(long vs wide, 설계서 §5)은 이 단계에서 **long-form으로 확정**한다. `ItemMetrics`와 동일한 이유(§N2 설계 의도: 지표 추가 시 스키마 불변)가 그대로 적용되고, 다른 산출물과 저장 형식을 통일하는 편이 `MetricsStore` 구현을 단순하게 만든다.

**테스트.**
- 저장 → 재로드 시 값 일치
- `source_run_manifest_hash`가 실제 `run_manifest.json` 파일 해시와 일치
- dump가 바뀌면(재실행 등) hash mismatch가 감지됨
- `metrics_schema_version` 불일치 거부(코어의 스키마 버전 정책과 동일한 엄격도)

**성공 조건.**
- 저장·재로드 일치, manifest 무결성 테스트 통과

---

### 단계 7. DebugViz V1 (마스크 검증)

**7단계를 5·6단계 직후에 두는 이유.** 설계서 §4의 근거를 그대로 계승한다: 지표가 나오자마자 눈으로 확인해야 코어 오류(좌표계 반전, region 인덱스 매핑 오류, `invert_mask` 반전, crop 불일치)를 조기에 발견할 수 있다.

**작업.** `viz/mask_check.py`가 원본 이미지 + 마스크 오버레이 + 실제 교란된 이미지를 3-패널 PNG로 저장한다. 소수 샘플(예: 5개) × region 1개.

**region 기하 복원과 의존 방향 예외.** 절차적 region(grid 등)은 `region_params_json`으로부터 `core.region.RegionResolver`를 재호출해 마스크를 재생성한다(코어의 결정론 덕분에 동일 마스크가 나온다). explicit region은 `ref_hash`로 원본 마스크 파일을 참조한다. 이것이 §3.3에서 `metrics.viz → core.region`을 유일한 명시적 예외로 둔 이유다 — 마스크 비트맵을 dump에 저장하지 않기로 한 코어의 결정(저장 효율)이 여기서 비용을 발생시키지만, 재생성이 결정론적이므로 문제가 없다.

**테스트.**
- 실제 committed fixture dump에서 마스크 위치가 grid 인덱스로부터 계산한 좌표와 일치하는지 자동 assert(육안 확인은 별도 매뉴얼 체크리스트로 문서화, CI는 좌표 assert만 검증)

**성공 조건.**
- PNG 생성 확인
- 좌표 assert 통과

---

### 단계 8. DebugViz V2·V3 (히트맵·랭킹)

**작업.**
- `viz/heatmap.py`: `SpatialProfile`을 이미지 위에 히트맵으로 오버레이(V2)
- `viz/ranking.py`: `vulnerability_score` 상위/하위 N개를 히트맵과 함께 나열(V3)

**테스트.**
- 히트맵 픽셀 강도 순서가 `SpatialProfile` 수치 순서와 일치하는지 자동 assert
- 랭킹 뷰의 나열 순서가 `vulnerability_score` 정렬 순서와 일치

**성공 조건.**
- 두 자동 assert 전부 통과

---

### 단계 9. L3 합성 Shortcut 실험

설계서 §3.4를 그대로 계승한다. 이 단계는 pytest collection에 포함하지 않고 **별도 실험 스크립트**(`experiments/synthetic_shortcut/`)로 수행한다 — 코어의 재현성 회귀 테스트(단계 8)와 달리, L3는 실제 모델 두 개를 학습시키는 연구 프로토콜이지 CI 게이트가 아니다.

**작업.**
- 데이터셋 A(오염)/B(무관)/C(무패치) 구성, M_shortcut/M_normal 학습
- Q1~Q5 판정 기준을 **사전 등록**하고 결과를 본 뒤 변경하지 않는다(설계서 §3.4)
- §3.5 마스크 방식 민감도(fill strategy별 region 순위 상관)를 같은 단계에서 축소판으로 수행

**잔여 결정.** 데이터셋·모델 선택, 합성 패치의 크기·위치·개수, Q2의 배수 k, Q5의 임계는 설계서 §5와 동일하게 **이 단계 착수 직전**에 확정한다.

**성공 조건.**
- Q1~Q5 판정 결과 확보 및 보고. 기준 미달 시 설계서 §3.4의 4단계 원인 분리 절차(L2 통과 확인 → DebugViz 마스크 검증 → shortcut 의존성 직접 확인 → 도구 한계로 기록)를 그대로 따른다

---

## 6. 단계 간 의존과 병렬화

```
0 ──> 1 ──> 2 ──> 3 ──> 4 ──> 5 ──> 6 ──┬──> 7 ──> 8 ──> 9
```

기본적으로 순차 진행을 권장한다(설계서 §4의 구현 순서를 그대로 계승). 다만 다음은 필요시 앞당길 수 있다.

- **단계 7(DebugViz V1)은 기술적으로는 단계 1(DumpReader) 이후 바로 시작할 수 있다.** V1은 `SpatialProfile`이 아니라 원본 dump와 `RegionResolver`만 있으면 되기 때문이다. 설계서와 이 계획이 5·6 직후에 두는 것은 순수히 워크플로 상의 이유(지표가 나온 시점에 바로 육안 검증)이지 하드 디펜던시가 아니다. 인력 여유가 있다면 단계 2~4와 병행할 수 있다.
- **단계 8(V2·V3)은 단계 6(MetricsStore)까지 끝나야 시작 가능하다.** `SpatialProfile`과 `vulnerability_score`가 있어야 하기 때문이다.

**단계 0이 병목이다.** 코어의 단계 1과 같은 이유로, 여기서 `ItemMetrics`/`SpatialProfile` 등의 스키마를 잘못 잡으면 이후 전 단계에 파급된다.

---

## 7. 테스트 전략

| 계층 | 범위 | 대응 단계 | 실행 방식 |
|---|---|---|---|
| L1. 단위 정확성 | 각 지표 계산식, 부호 정규화 | 단계 2~4 | pytest, 손계산 고정 입력 |
| L2. 파이프라인 무결성 | dump → 지표 → 집계 | 단계 1, 5 | pytest, `synthetic_dump_builder`로 코어 미실행 합성 dump 주입 |
| L3. End-to-end 타당성 | 코어+지표 전체 | 단계 9 | 별도 실험 스크립트, CI 밖 |

L1·L2는 코어의 `unit`/`integration` 계층과 동일하게 기본 `pytest` collection에 포함되어 `.github/workflows/ci.yml`의 단일 `pytest -q` 잡에 자연히 합류한다. GPU가 필요한 테스트는 없다(지표 엔진은 numpy/pandas 연산만 수행).

---

## 8. 잔여 결정 사항의 처리 시점

| 항목 | 결정 시점 | 결정 내용(확정된 경우) |
|---|---|---|
| 지표 엔진 패키지 위치 | 이 계획서 작성 시 확정 | `ssat/metrics/` (§3.2) |
| `topk_exit`의 k 기본값 | 단계 3 | 기본 5, `min(5, num_classes)`로 자동 축소 (제안값, 변경 가능) |
| 확률 입력에서 logit 복원 허용 여부 | 단계 2 | 현재는 허용하지 않음 — 코어가 `probs`를 지원하지 않으므로 논의 유보 |
| `SpatialProfile`의 저장 형식(long vs wide) | 단계 6 | long-form |
| CLI 표면(`ssat metrics ...`) 추가 여부 | v1 범위 밖 → 단계 9 완료 후 후속 도입(§2.2) | 구현됨 — 독립 명령 `ssat metrics <dump>`, 항상 내장 9개 전부 계산 |
| Q2의 배수 k, Q5의 임계 | 단계 9 착수 전(사전 등록) | 미정 |
| 합성 패치의 크기·위치·개수 | 단계 9 설계 시 | 미정 |
| L3에 사용할 데이터셋·모델 | 단계 9 설계 시 | 미정 |

---

## 9. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| 코어의 기존 `DumpReader`와 지표 엔진의 `DumpHandle`(N0) 개념이 이름·역할 모두 유사해 혼동됨 | §1, §3.1에 "감싸는 관계"임을 명시하고, `metrics.dump_reader` 외 어떤 모듈도 `core.dump`를 직접 import하지 않도록 import-linter로 강제 |
| `output_kind=probs` 분기가 실제로는 코어가 만들어낼 수 없어 실전 검증이 안 됨 | 모의(mock) `AdapterSpec` 기반 단위 테스트로만 커버하고, 코어가 `probs`를 지원하는 시점에 실통합 테스트를 추가해야 한다는 사실을 이 문서와 코드 주석에 남김 |
| `vulnerability_score`의 `primary_metric` 설정 오류(오탈자, 미등록 지표 지정) | `MetricRegistry`에 등록된 이름에 대해서만 허용하는 명시적 스키마 검증, 기본값(`margin_drop`) 문서화 |
| `metrics.viz`가 `RegionResolver` 재호출로 코어 내부 세부에 결합됨 | §3.3 의존 방향 표에 유일한 명시적 예외로 문서화해 향후 코어 리팩터링 시 영향 범위를 즉시 인지 가능하게 함 |
| L2 fixture가 코어 실행 경로를 몰래 재사용해 "코어 미실행" 전제가 깨짐 | `synthetic_dump_builder.py`가 `PlanBuilder`/`run_audit`/adapter를 import하지 않는지 코드 리뷰·정적 검사로 확인 |
| L3 실험의 데이터셋/모델 학습 리소스·시간 | 클래스 수가 적고 학습이 빠른 소규모 공개 데이터셋 우선, GPU 없이도 재현 가능한 규모로 설계 |
| 범위 확대 | 3순위 이후 지표, 대조군 비교, 안정성 분석, 클러스터링, 정식 HTML 리포트는 v1에서 손대지 않음(설계서 §0 제외 범위) |
