# 코드 구조·모듈화 분석 (v1)

## 0. 문서 범위와 방법론

**목적.** 코드를 변경하지 않고, (1) 책임 분리·객체지향 패턴 관점의 리팩터링 여지와 (2) 새 데이터셋·모델을 최소한의 코드로 연결할 수 있는 모듈화 수준 및 변경에 취약한 구현 지점을 분석한다.

**범위.** `ssat/` 패키지 전체. `experiments/`는 이번 분석에서 제외한다(추후 별도 작업 예정).

**방법.** `ssat/`를 5개 서브시스템으로 나누어 각각을 기존 설계 문서와 대조하며 코드를 직접 읽었다.

| 서브시스템 | 대상 디렉터리 | 대조한 설계 문서 |
|---|---|---|
| A. 데이터 소스·모델 어댑터 | `core/source/`, `core/adapter/`, `application/config.py` | `CORE_DESIGN_v1.md` §M3, §M8 |
| B. 리전·교란·플랜 | `core/region/`, `core/perturb/`, `core/plan/` | `CORE_DESIGN_v1.md` §M1, §M4, §M5 |
| C. 런타임·비용추정·애플리케이션·설정 | `core/runtime/`, `core/estimate/`, `application/`, `core/config/`, `core/resume/`, `cli.py` | `CORE_DESIGN_v1.md` §M0, M2, M6~M9, M11 |
| D. 분석·지표·리포트 | `analysis/`, `metrics/`, `report/`, `presentation.py` | `METRIC_ENGINE_DESIGN_v1.md`, `REPORT_LAYER_DESIGN_v1.md` |
| E. 덤프·재개·유틸·CLI 인프라 | `core/dump/`, `utils/`, `cli.py`(교차) | `CORE_DESIGN_v1.md` §M2, §M10, §4 |

모든 findings는 실제 코드를 읽고 확인한 `file:line` 근거를 가진다(§5 부록). 이 문서는 진단 문서이며, 구체적 수정 코드는 제시하지 않는다.

---

## 1. 총평

**강점.** 이 코드베이스는 설계 문서를 먼저 쓰고 그에 맞춰 구현하는 방식을 취하고 있고, 실제로 핵심 파이프라인(런타임 M6~M9)은 설계 문서와 거의 1:1로 대응한다. 어댑터(`ModelAdapter`)와 데이터 소스(`SampleSource`)는 이름 기반 provider/registry 패턴이 대칭적으로 적용되어 있어, 등록만으로 새 모델·데이터셋을 코드 차원에서 연결할 수 있다. 계층 간 순환 의존은 발견되지 않았고, 전역 mutable 싱글턴도 로거를 제외하면 거의 없어 테스트·병렬 실행에 유리한 구조다. 실패 처리(`status` 필드), 재현성(item_id 해시 기반 시드), 원자적 쓰기 등 핵심 불변식들은 설계 의도대로 준수되고 있다.

**반복적으로 나타나는 리스크 유형.** 5개 서브시스템을 독립적으로 분석했음에도 동일한 유형의 문제가 서로 다른 위치에서 반복적으로 발견되었다. 이는 우연이 아니라 이 코드베이스의 구조적 습관에 가깝다.

| 유형 | 설명 | 발견된 위치(대표 예) |
|---|---|---|
| **A. 3중 보일러플레이트 복제** | 동일한 구조의 factory/dispatch/storage 클래스가 서로 다른 서브패키지에 독립적으로 재구현됨 | region/expander/perturb factory 3종, dump/metrics/analysis의 `_storage.py` 3종 |
| **B. 컨벤션 의존 동기화 계약** | 두 곳에서 같은 "키 집합"이나 "포맷 문자열"을 하드코딩하고, 타입 시스템이나 테스트가 아닌 사람의 기억으로만 동기화됨 | region param 스키마, `region_id::region_instance_id` 키, `WorkItem.identity_payload()` |
| **C. 파사드/모듈의 책임 과다** | 하나의 클래스·파일이 오케스트레이션과 실제 계산/렌더링을 함께 수행 | `AuditApplication`(976줄, 12개 이상 책임), `ReportDataAssembler`(1241줄) |
| **D. 이름 문자열 일치에만 의존하는 미검증 계약** | Pydantic `Literal` 필드 값과 registry 등록 이름이 정적으로 검증되지 않음 | adapter/source provider의 `Literal[...]` vs `name = "..."` |
| **E. 확장 포인트의 비대칭성** | 같은 "확장 지점"이라는 개념 안에서도 실제 확장 난이도가 크게 다름 | adapter/source=개방형 레지스트리, region kind/perturb op=폐쇄형 enum, skeleton region provider=하드코딩된 유일 사례 |
| **F. 설계 문서가 예견했지만 미실행된 리팩터링** | 설계 문서 자체가 "중복이 확인되면 추출하라"고 명시했고 그 조건이 이미 충족되었는데도 추출이 이루어지지 않음 | `ModelLoader`/`ModelRunner` 공통 계약 (4개 어댑터에서 동일 보일러플레이트 확인됨) |

아래 2~4절은 사용자가 요청한 두 축(책임 분리/OOP, 모듈화/확장성) 순서로 재구성한 요약이며, §5 부록에 서브시스템별 원본 상세 findings를 수록한다.

---

## 2. 책임 분리와 객체지향 패턴 분석

### 2.1 잘 적용된 패턴

- **Provider/Registry 전략 패턴**: `AdapterProvider`/`AdapterProviderRegistry`(`ssat/core/adapter/provider.py`)와 `SourceProvider`/`SourceProviderRegistry`(`ssat/core/source/provider.py`)는 거의 동형으로 설계되어 있고, `ConfigResolver`·`application/config.py`·`application.py`의 오케스트레이션 코드 어디에서도 provider 종류에 대한 `isinstance`/`kind==` 분기가 없다. 이름으로 parse하고 이름으로 build하는 원칙이 실제로 지켜지고 있다.
- **정렬된 strategy 목록 + `supports()` first-match 디스패치**: region mask(`mask_dispatch.py`), region family 확장(`region_expander_dispatch.py`), 교란 연산자(`perturb/dispatch.py`) 세 곳 모두 이 패턴을 일관되게 사용한다. `ConfigResolver`도 런타임과 동일한 순서 목록을 재사용해, "설정 검증"과 "실제 확장"이 같은 구현체 하나에 귀속되도록 되어 있다.
- **런타임 파이프라인의 단일 책임 분리**: `core/runtime/pipeline.py`(정책 무관 iterator) / `execution.py`(fail-fast·dump 기록 정책) / `processors.py`(워커 진입점) / `batching.py`(Rebatcher·BatchSplitter)는 서로 기능 중복 없이 명확히 분리되어 있으며, `core/estimate/`가 이 동일한 `pipeline.py` 함수들을 재사용해 "실제 실행과 동일한 코드 경로를 프로파일링한다"는 설계 원칙(M11)을 지킨다.
- **덤프 계층의 저수준/스키마 분리**: `core/dump/schema.py`는 `writer.py`/`reader.py`에 대한 의존이 전혀 없고, `_storage.py`는 순수 I/O만 담당한다.
- **`report/charts.py`, `html_renderer.py`는 `ssat.metrics`/`ssat.analysis`를 직접 import하지 않도록 강제**되어 있어(테스트로 보장), 이미 조립된 `ReportModel`만 소비하는 순수 변환 계층으로 잘 유지되고 있다.

### 2.2 책임이 과도하게 집중된 지점

**`AuditApplication` (`ssat/application/application.py`, 976줄).** 실행 오케스트레이션(`prepare_run`/`execute_run`), 독립 비용 추정(`estimate`), 덤프 조회(`inspect`), 지표 계산(`compute_metrics`), 통계 분석(`analyze`), HTML 리포트 생성(`generate_report`), 라벨 export(`export_labels`), 인덱스 재구축(`rebuild_index`)까지 서로 무관한 12개 이상의 공개 메서드가 한 클래스에 모여 있다. `CORE_DESIGN_v1.md` §0은 "지표 계산, 집계, 리포트, ... 는 코어에 포함되지 않는다"고 명시하는데, 실제로는 이들을 모두 흡수하는 단일 파사드가 존재한다. 특히 `generate_report()`(574~710행)는 단순 위임이 아니라 차트 디렉터리 생성, SVG 파일 직접 쓰기, `dataclasses.replace()`로 모델 재조립까지 수행하는 실질적 비즈니스 로직을 담고 있다. 실행(run) 경로에 영향을 주려는 의도가 없는 리포트 관련 수정이 같은 파일·같은 임포트 표면에 놓이게 되어, 변경 시 영향 범위 판단이 어려워진다.

**`ReportDataAssembler` (`ssat/report/assembler.py`, 1241줄).** 설계 문서(`REPORT_LAYER_DESIGN_v1.md` §0)는 "리포트는 계산하지 않고 조립한다"를 원칙으로 명시하지만, 실제로는 Shannon 엔트로피·지배적 비중 계산(`_build_spatial_concentration`, `_build_semantic_concentration`), `np.mean`/`percentile` 요약통계, 클래스×semantic 교차표 산출 등 새로운 파생 통계를 직접 계산한다. 이는 모듈 docstring이 스스로 "Gap#6"으로 인정하고 있는 지점이다.

**4종 torch 기반 어댑터의 반복 코드.** `TorchvisionAdapter`/`TorchvisionVideoAdapter`/`TorchvisionTSMAdapter`/`TimmAdapter`는 디바이스 결정 로직, `cleanup_after_oom()` 본문, `init_seed` 검증 및 `fork_rng` 패턴, `predict()`의 예외 래핑 패턴을 각각 독립적으로 거의 동일하게 구현하고 있다. 흥미로운 점은 `CORE_DESIGN_v1.md` §M8이 이미 "빌트인 및 사용자 provider에서 반복 구현이 확인된 경우에만 `ModelLoader`/`ModelRunner` 공통 계약으로 추출한다"고 명시했고, 지금 그 트리거 조건(4개 빌트인 provider 모두에서 확인된 반복)이 충족되어 있는데도 그런 추출이 아직 이루어지지 않았다는 것이다.

**Region/Perturb/Plan의 3중 factory 복제.** `RegionMaskGeneratorFactory`(`region/mask_factory.py`), `RegionFamilyExpanderFactory`(`plan/region_expander_factory.py`), `OperatorFactory`(`perturb/factory.py`)는 등록 시 타입 검사, 중복 등록 거부, 인스턴스 생성 로직을 거의 동일하게 각각 재구현한다. 제네릭 `StrategyRegistry[T]` 하나로 통합 가능한 구조이나 지금은 3곳에서 독립적으로 유지보수해야 한다. 메서드 이름조차 일관되지 않다(`build()` vs `build_operators()`).

**3종 저수준 스토리지 헬퍼 복제.** `atomic_write_parquet`/`fsync_directory`가 `core/dump/_storage.py`, `metrics/_storage.py`, `analysis/_storage.py`에 바이트 단위로 동일하게 복제되어 있다. 각 파일이 이를 "패키지 경계를 지키기 위한 의도적 트레이드오프"로 docstring에 명시하고 있어 완전히 우발적인 중복은 아니지만, 여전히 `ssat/utils/io.py`처럼 세 패키지가 공통 참조할 수 있는 위치로 옮길 수 있는 코드다.

---

## 3. 모듈화·확장성 분석

### 3.1 새 데이터셋을 추가하려면

**Python API로 직접 사용할 경우** — 확장이 매우 쉽다. `SampleSource` Protocol(`core/source/base.py`)을 구현하고, `SourceProviderConfig`/`SourceProvider` 서브클래스를 작성해(`imagenet.py`/`kinetics.py`가 참고 예시) `registry.register(...)` 후 `AuditApplication(source_registry=registry)`에 전달하면 끝난다. `application/config.py`의 실제 구성 경로(`resolved_source_registry.parse(...)` → `.build(...)`)는 어떤 kind 분기도 없이 순수하게 registry에 위임한다.

**그러나 CLI로는 이 경로가 막혀 있다.** `ssat/cli.py`의 `create_app()`은 `adapter_registry` 파라미터는 받지만 **`source_registry` 파라미터가 없다**. `docs/CONFIG_REFERENCE.md`도 "사용자 정의 source provider는 Python API에서만 사용 가능하다"고 명시적으로 인정하는 제약이다. YAML + 표준 `ssat` CLI만으로는 내장 4종(`image_manifest`, `video_manifest`, `imagenet`, `kinetics400`) 외의 데이터셋을 연결할 방법이 없다.

### 3.2 새 모델을 추가하려면

**가장 쉬운 경로(문서화되지 않음)**: 서브클래싱 없이 `CallableAdapter`(`core/adapter/callable_adapter.py`)로 임의의 Python 함수/ONNX/HuggingFace 모델을 감싸 바로 `AuditApplication`에 전달할 수 있다. 다만 `CONFIG_REFERENCE.md`는 이 경로를 전혀 소개하지 않아, 문서만 보고 확장을 시도하는 사용자는 이 가장 쉬운 길을 발견하지 못할 가능성이 높다.

**YAML 설정 기반 provider로 등록**: `ModelAdapter` 서브클래싱(필요한 경우만) + `ProviderConfig`/`AdapterProvider` 서브클래싱 + `registry.register(...)` 후 `create_app(adapter_registry=registry)`로 CLI에도 주입 가능하다(source와 달리 CLI가 이 파라미터를 노출한다). `core/config/schema.py`에는 adapter 종류를 나열하는 `Literal`이 없어, 새 provider 추가 시 코어 스키마를 건드릴 필요가 없다는 점은 설계 의도대로다.

**MMAction2류 프레임워크 모델은 훨씬 무겁다.** 내장 TSM 어댑터(`mmaction_checkpoint.py`, `torchvision_tsm_adapter.py`)는 ResNet50-TSM 하나에 고정된 state-dict 키 리매핑과 아키텍처 재구현으로 이루어져 있어, 이 패턴은 다른 MMAction2 아키텍처(TSN, SlowFast 등)로 일반화되지 않는다. 재사용 가능한 "MMAction2 체크포인트 어댑터" 추상화는 존재하지 않는다.

### 3.3 새 region kind / perturbation 연산자를 추가하려면

**새 교란 연산자는 순수 가산적(additive)이다.** enum 멤버 추가 → `PerturbationOperator` 서브클래스 1개(`supports`/`validate_config`/`apply`를 한 클래스에 모두 포함) → `_DEFAULT_OPERATOR_TYPES` 튜플에 append. `Perturbator`나 중앙 디스패치 로직은 건드릴 필요가 없다.

**새 region kind는 두 개의 독립된 클래스 계층을 함께 건드려야 하고, 그 둘 사이의 파라미터 스키마 일치는 타입 시스템이 아닌 관례로만 보장된다.** `RegionFamilyExpander`(플래닝 시점, family→concrete recipe)와 `RegionMaskGenerator`(실행 시점, recipe→bitmap)는 서로 다른 파일에 독립 등록되며, 둘 다 동일한 `params` 키 집합(예: grid의 `{rows, cols, row_index, col_index}`)을 각자 하드코딩된 `set(...)` 리터럴로 기대한다. 이 계약이 어긋나면 `ConfigResolver`나 `PlanBuilder.enumerate()` 단계가 아니라, 워커가 실제로 마스크를 만드는 시점에야 `RegionResolutionError`로 드러난다.

**샘플-의존 region kind(skeleton_parts류)는 사실상 순수 가산적이지 않다.** `SampleDependentRegionExpander`(`plan/region_expanders.py`)는 `bbox_partition`/`skeleton_parts`/`gt_bbox` 세 kind를 하나의 클래스가 `if/elif`로 분기하며 소유하고 있어, 다른 세 확장기(grid/explicit/random_area_match)가 따르는 "kind당 클래스 하나" 원칙을 깨고 있다. 또한 `SampleRegionProvider`는 `RegionExpander`/`MaskResolutionContext`/`RegionResolver`에 각각 이름이 박힌 단일 필드(`skeleton_store`)로만 주입 가능해, 두 번째 샘플-의존 kind가 추가되면 제네릭한 "서비스 주입" 지점이 아니라 또 다른 전용 필드를 추가해야 한다. `application.py._build_context`도 `SkeletonRegionProvider`를 registry가 아닌 하드코딩된 조건 분기로 직접 생성한다 — adapter/source가 갖고 있는 이름 기반 registry가 이 확장 지점에는 없다.

### 3.4 새 지표 / 리포트 섹션을 추가하려면

**새 내장 지표는 마찰이 적다.** `Metric` Protocol(`metrics/registry.py`)을 구현하는 클래스 1개 + `default_metric_registry()`에 등록, 총 2개 파일 수정으로 끝난다. `aggregate.py`/`store.py`/`assembler.py`/`charts.py`는 `primary_metric: str` 파라미터화 덕분에 건드릴 필요가 없다 — 어댑터/소스 registry 수준으로 낮은 마찰이다.

**새 차트/리포트 섹션은 5개 이상 파일을 건드려야 하고, 그 조율 지점이 `report/` 패키지 밖(애플리케이션 계층)에 있다.** `ReportModel`(`report/types.py`) 필드 추가 → `assembler.py`에서 데이터 채우기 → `charts.py`에 렌더 함수 추가 → **`application.py`의 `generate_report()`**에서 실제 SVG 파일 쓰기와 asset ref 재조립(설계 문서의 R0~R4 파이프라인 다이어그램에는 없는 5번째 필수 지점) → `html_renderer.py`의 두 템플릿(`_REPORT_TEMPLATE_B` 포함) 중 필요한 곳에 섹션 추가. 지표 확장에 비해 훨씬 무겁고 확산되어 있다.

### 3.5 변경에 취약한 구현 지점 (종합)

사용자가 요청한 "변경에 취약한 부분"에 해당하는 항목을 서브시스템 횡단으로 모으면 다음과 같다.

1. **이름 문자열 일치에만 의존하는 미검증 계약**: `TorchvisionProviderConfig.provider: Literal["torchvision"]`류의 값과 `TorchvisionProvider.name = "torchvision"`이 반드시 일치해야 하지만, 이를 정적으로 검사하는 코드가 없다(`core/adapter/provider.py`, `core/source/provider.py`). 오타가 나면 해당 provider는 영구히 선택 불가능해지고, 실패 메시지도 "등록 버그"가 아니라 혼란스러운 Pydantic literal mismatch로 나타난다.
2. **`WorkItem` identity 필드의 이중 관리 및 사문화(死文化)된 검증 경로**: `WorkItem.identity_payload()`(`plan/types.py`)가 정의하는 해시 대상 필드 목록과, 실제로 해시에 쓰이는 `PlanBuilder._make_item`(`plan/builder.py`)의 리터럴 dict가 서로 다른 코드로 독립 유지되고 있으며, `identity_payload()`는 프로덕션 경로에서 한 번도 호출되지 않는다(테스트에도 없음). 새 필드를 추가할 때 한쪽만 갱신해도 아무 것도 실패하지 않는다.
3. **`region_id::region_instance_id` 키 포맷의 7중 복제**: `metrics/aggregate.py`, `analysis/indexer.py`(2곳), `analysis/interval.py`, `analysis/control.py`, `analysis/stability.py`, `analysis/reader.py`, `metrics/viz/mask_check.py`에 동일한 f-string이 독립적으로 반복된다. `analysis` 패키지가 `metrics.aggregate`에 의존하지 않겠다는 의도적 설계 결정 때문이지만, 결과적으로 포맷을 바꾸면 7곳을 손으로 맞춰야 한다.
4. **`_check_region_geometry` 로직의 이중 구현**: `metrics/aggregate.py`와 `analysis/indexer.py`가 동일한 정책을 독립적으로 재구현하며, `analysis/indexer.py`의 docstring이 이를 의도적 비의존이라고 명시한다 — 그러나 공유 위치(`ssat/core` 또는 `analysis.types`)로 추출하지 않고 전체 복제를 택한 이유는 문서만으로는 설득력이 약하다.
5. **`invert_mask` 처리 로직의 이중 구현**: `core/runtime/processors.py`의 `ChunkProcessor.__getitem__`과 `core/estimate/area_sanity.py`의 `_measure_item`이 마스크 반전 후 면적 재계산 로직을 동일하게 복제하고 있다.
6. **공통 협력자(quintuple)의 장거리 시그니처 전파**: `(plan_builder, sample_source, adapter, region_resolver, perturbator)` 조합이 `run_audit()`, `iter_prepared_work_chunks()`, `ChunkProcessor`, `CostEstimator.estimate()`, `PerturbedProfiler.run()`, `AreaSanityCheck.run()` 등 최소 6곳의 함수/클래스 시그니처에 반복 등장한다. 새로운 횡단 협력자(예: 공유 캐시)를 추가하려면 이 6곳과 `application.py._build_context`까지 함께 수정해야 한다 — `_ExecutionContext`(`application.py`)가 이미 부분적으로 이 문제를 인지하고 만든 번들이지만 `core/runtime`/`core/estimate` 함수 시그니처 자체에는 적용되어 있지 않다.
7. **두 개의 독립된 `RegionFamilyExpander` 목록**: `ConfigResolver.__init__`(검증용, `skeleton_store` 없음)과 `application.py._build_context`(실제 플래닝용, `skeleton_store` 있음)가 별도로 `RegionExpander`를 구성한다. 두 목록이 같은 `_DEFAULT_EXPANDER_TYPES` 순서를 쓰는지는 관례로만 보장되며, 어긋나면 `ConfigResolver`는 통과했지만 `PlanBuilder`가 확장에 실패하는 상황이 생길 수 있다.
8. **두 개의 병렬 전처리 엔진**: 구버전 flat-op 엔진(`preprocessing.py`, `preprocessing` 필드)과 신버전 registry 기반 파이프라인(`transform_registry.py`, `pipeline_config` 필드)이 공존하며, 4개 어댑터가 이 중 서로 다른 부분집합만 지원한다(`TorchvisionAdapter`=둘 다, `TorchvisionVideoAdapter`=pipeline만, `Timm`/`TSM`=둘 다 불가). 새 전처리 연산을 추가하려면 어느 엔진에 구현할지, 혹은 양쪽에 다 구현해야 할지 판단이 필요하다.
9. **덤프 계층의 내부 파일 레이아웃이 실제로는 캡슐화되어 있지 않음**: `core/resume/index.py`가 `core/dump/_index.py`, `core/dump/_storage.py`라는 밑줄 접두사(사실상 private) 모듈을 직접 import하고, fragment 파일명 규칙(`chunk_{ordinal:08d}.parquet`)을 자체적으로 재구현한다. `DumpReader`가 유일한 다운스트림 API라는 설계 원칙이 `resume` 패키지에는 적용되지 않고 있다.
10. **`html_renderer.py`의 카드 키 문자열 하드코딩**: 보조 템플릿(`_REPORT_TEMPLATE_B`)이 `selectattr("key", "equalto", "accuracy")` 같은 방식으로 특정 태스크의 스코어카드 키를 가정한다. 현재는 `ClassificationAdapter`만 구현되어 있어 문제가 드러나지 않지만, 향후 `DetectionAdapter`(현재 `NotImplementedError`로 스텁만 존재)가 `"accuracy"` 키를 내지 않으면 `selectattr(...) | first`가 빈 시퀀스에서 Jinja `UndefinedError`를 던진다 — "리포트는 태스크 종류를 모른다"는 설계 원칙이 이 지점에서 깨진다.
11. **`report/assets.py`가 스스로의 설계 헌장을 위반**: 설계 문서(R3)는 "이미 생성된 자산의 재배치만 하고 신규 렌더링은 하지 않는다"고 명시하지만, 실제 `assets.py`는 `ssat.metrics.viz.heatmap.render_heatmap_panel`을 직접 호출해 리포트 시점에 새로 PNG를 렌더링하며, `ssat.metrics.store`/`viz._shared`/`errors`까지 파고든다. 모듈 자체 docstring이 이를 인정하고 있다.

---

## 4. 우선순위 권고 (방향 제시, 상세 설계는 별도 작업)

구체적인 수정 코드는 제시하지 않으며, 다음 순서로 검토할 것을 권고한다.

1. **`AuditApplication` 분할**: run 실행/추정과 무관한 리포트·지표·분석·라벨 export 책임을 별도의 파사드(또는 얇은 서비스 클래스들)로 분리해, "코어는 판단하지 않는다"는 설계 원칙과 실제 파사드 경계를 일치시킨다.
2. **`ModelLoader`/`ModelRunner` 공통 계약 추출**: 설계 문서가 이미 예견한 대로, 4개 torch 어댑터의 디바이스 결정·OOM cleanup·seed 초기화·예외 래핑 보일러플레이트를 공유 베이스/믹스인으로 추출한다.
3. **`StrategyRegistry[T]` 제네릭화 검토**: region mask / region expander / perturb operator 3개 factory의 중복 보일러플레이트를 통합할지 여부를 결정한다(단, `build()` 시그니처 차이 — context 유무 — 를 어떻게 흡수할지가 핵심 설계 질문).
4. **Provider 이름 일치의 정적 검증 추가**: `Literal[...]` 기본값과 `name = "..."` class attribute가 항상 일치하도록 등록 시점 assertion 또는 테스트를 추가한다(등록 로직 자체는 변경하지 않음).
5. **CLI의 `source_registry` 노출 불균형 해소**: `create_app()`이 `adapter_registry`와 대칭적으로 `source_registry`도 받을 수 있게 하거나, 이 비대칭이 의도적이라면 문서에 그 이유를 명시한다.
6. **`WorkItem.identity_payload()`를 실제 해시 경로의 단일 진실 공급원으로 승격**: 현재 죽은 코드로 남아있는 이 메서드를 `PlanBuilder._make_item`이 실제로 호출하도록 일원화하거나, 필요 없다면 제거해 이중 관리를 없앤다.
7. **`region_id::region_instance_id` 키 포맷과 `_check_region_geometry` 정책의 단일 소스화**: `analysis`↛`metrics` 비의존이라는 설계 제약을 지키면서도, 공유 가능한 위치(`ssat/core` 하위)로 추출하는 방안을 검토한다.
8. **`CallableAdapter` 경로를 `CONFIG_REFERENCE.md`에 문서화**: 코드 변경 없이 가장 빠르게 확장성 체감을 높일 수 있는 항목이다. source 쪽에는 이미 커스텀 provider worked example이 있으므로 동일한 수준으로 모델 쪽도 보완한다.
9. **`html_renderer.py`의 카드 키 하드코딩 제거를 `DetectionAdapter` 구현 착수 이전에 선행**: 두 번째 태스크 어댑터가 실제로 추가되기 전에 이 구조적 위험을 해소해야 나중에 런타임 오류로 발견하지 않는다.

---

## 5. 부록: 서브시스템별 상세 근거

아래는 5개 서브시스템 분석에서 수집한 원본 findings의 요약이다. 각 항목은 실제 파일:라인을 근거로 확인되었다.

### 5.A 데이터 소스·모델 어댑터 (`core/source/`, `core/adapter/`)

- `AuditApplication`이 God object 경향을 보이는 것과 별개로, 어댑터 4종의 `_resolve_weights`가 `TorchvisionAdapter`/`TorchvisionVideoAdapter` 사이에 바이트 단위로 중복된다.
- `SampleSource` 쪽도 `_load_error` 헬퍼가 `ImageFolderSource`/`VideoFolderSource` 사이에 동일하게 중복되고, `_resolve_existing` 계열이 `imagenet.py`/`kinetics.py` 사이에 동일하게 중복된다.
- `mmaction_checkpoint.py`는 core 어댑터 코드 내부에 MMAction2/MMEngine의 pickle 안전목록·`HistoryBuffer` 모듈 경로 등 프레임워크 내부 구현 세부사항을 담고 있다. import는 하지 않지만 "코어는 프레임워크 무의존"이라는 설계 원칙의 뒷문에 해당한다.
- adapter/source provider의 kind 문자열은 스키마(`core/config/schema.py`)에 별도 `Literal`로 중복되어 있지 않다 — 이는 긍정적인 설계 지점으로, provider 자신의 `Literal` 필드 + registry의 `name` 속성, 2곳만 일치하면 된다(다만 이 2곳의 일치 자체는 §3.5 항목 1 참고).

### 5.B 리전·교란·플랜 (`core/region/`, `core/perturb/`, `core/plan/`)

- 디스패치 함수의 구조가 서브시스템마다 다르다: `mask_dispatch.py`는 find+실행이 하나로 합쳐진 함수만 제공하지만, `region_expander_dispatch.py`/`perturb/dispatch.py`는 `find_X()`/`dispatch_X()`로 분리되어 있다. `ConfigResolver`가 설정 검증 시점에 필요로 하는 `find_*` 계열이 mask 쪽에는 없다.
- `PerturbationOperator` 베이스 클래스는 `requires_dataset_stats()`/`resolve_config_params()`라는 실질적인 template method 기본 구현을 제공하는 반면, `RegionMaskGenerator`/`RegionFamilyExpander`는 constructor injection(`self._context = context`) 외에는 공유 동작이 전혀 없다 — 세 베이스 클래스가 "공유 상태/행동" 문제를 서로 다른 방식으로 풀고 있다.
- `PlanBuilder`가 `random_area_match` 대조군을 위해 `RegionSpec`을 직접 조립하며(`plan/builder.py`), `RegionMaskGenerator` 쪽의 파라미터 스키마를 import 연결 없이 관례로만 공유한다. 이는 설계 문서가 명시적으로 허용한 예외이지만, `RegionSpec`에 필드가 추가되면 `region/types.py`, `plan/builder.py`, `region/mask_generators.py` 세 곳을 동시에 고쳐야 한다.
- `RegionKind.EXPLICIT` 하드코딩 분기가 `config/schema.py`(2곳), `config/resolver.py`(2곳), `region/types.py`, `region/mask_generators.py`, `plan/region_expanders.py`까지 7곳에 독립적으로 존재한다.

### 5.C 런타임·비용추정·애플리케이션·설정

- `core/runtime/pipeline.py`/`execution.py`/`processors.py`/`batching.py`는 설계 문서와 잘 대응하며 서로 중복이 없다. 다만 `core/estimate/`의 `measurement.py`/`profiler.py`/`sanity.py`가 `core/runtime`의 내부 전송 타입(`PerturbedInferenceItem`, `BatchSizeState` 등)에 직접 의존해, "코어는 실행 중 판단하지 않는다"는 계층 분리가 `estimate`↔`runtime` 사이에서는 상당히 깊게 결합되어 있다.
- `adapter.describe() == config.adapter_spec` 검증이 `execution.py`(인라인)와 `estimate/measurement.py`(`_validate_provenance`, 4곳에서 재사용)에 각각 독립 구현되어 있다.
- skeleton region provider는 `application.py._build_context`에서 registry가 아닌 하드코딩된 `if/else`로 직접 생성된다 — adapter/source가 가진 개방형 registry 패턴이 이 지점에는 적용되어 있지 않다.
- `ConfigResolver.__init__`(검증용 expander 목록)과 `application.py._build_context`(실행용 expander 목록)가 독립적으로 `RegionExpander`를 구성하며, 둘의 동기화는 관례에 의존한다.

### 5.D 분석·지표·리포트

- `Metric` 레지스트리(`metrics/registry.py`)는 어댑터/소스 provider와 구조적으로 유사하지만 `parse()`/config model이 없어 YAML로 지표를 확장할 수는 없다(코드 레벨 확장만 가능). v1 범위에서는 합리적인 축소이나, 유일한 예외(`topk_exit`의 `k` 값)가 `metrics/store.py`에서 `getattr(metric, "k", None)`으로 문자열 이름을 하드코딩해 다루고 있어, 두 번째 "지표별 설정값"이 필요해지면 확장 지점이 없다.
- `_check_region_geometry`, `region_id::region_instance_id` 키, `_is_binary_primary_metric`, `DEFAULT_TOPK`/`DEFAULT_PRIMARY_METRIC` 등 다수의 로직/상수가 `analysis`↔`metrics`, `assembler.py`↔`labels.py` 경계를 넘어 의도적으로 중복 구현되어 있다(각 파일이 "의존성 방향 제약 때문"이라고 docstring에서 인정).
- `report/assets.py`가 설계 문서(R3)의 "재배치만, 신규 렌더링 금지" 원칙을 위반하고 있는 것이 이 서브시스템에서 가장 명확한 설계-구현 괴리 사례다.

### 5.E 덤프·재개·유틸·CLI 인프라

- `DumpWriter`는 I/O 오케스트레이션, row-스키마 매핑(`_clean_row`/`_perturbed_row`), 쓰기 순서 정책을 한 클래스에서 함께 수행한다. `PERTURBED_SCHEMA` 정의(`schema.py`)와 `_perturbed_row`(`writer.py`)의 필드 목록은 컴파일 타임 연결 없이 사람이 손으로 맞춰야 한다.
- `ResumeIndex.rebuild()`(`core/resume/index.py`)가 사실상 두 번째 "writer"로서 `DumpWriter`의 fragment 명명 규칙과 인덱스 파생 로직을 독립 재구현한다.
- `schema_version` 호환성 체크가 `manifest.py`(JSON 기반)와 `_storage.py`(Arrow 메타데이터 기반)에 서로 다른 메커니즘으로 중복 존재하며, `writer.py`에는 이미 `load_manifest`가 검사한 것을 다시 검사하는 도달 불가능한(`# pragma: no cover`) 코드도 남아있다.
- `write_json_atomic`(`utils/io.py`)은 디렉터리 fsync를 자체적으로 하지 않아, 호출자가 별도로 `fsync_directory`를 불러야 한다. `manifest.py`만 이를 올바르게 수행하고, `analysis/store.py`·`metrics/store.py`·`region/skeleton_bbox_builder.py`·`report/exporter.py`·`report/labels.py`의 호출부는 그렇지 않다 — 크래시 안전성의 잠재적 공백이다.
- `cli.py` 자체는 얇고 비즈니스 로직이 없는 좋은 상태를 유지하고 있다.

---

*이 문서는 2026-08-21 기준 `phase0-softwarex-prep` 브랜치의 코드를 대상으로 작성되었으며, 코드 변경은 포함하지 않는다.*

---

## 6. Phase 6 반영 현황 (2026-08-24 추가)

SoftwareX 제출 준비 Phase 6("문서/테스트/CI 마무리")에서 위 §4 권고 중 문서화만으로 정직하게 다룰 수 없는 두 항목, 그리고 이 문서가 직접 지적하지는 않았지만 같은 조사 과정에서 추가로 발견된 비대칭 하나를 처리했다. 나머지 권고(§4의 1~4, 6~7, 9)는 제출 직전 브랜치에 넣기에는 회귀 위험이 큰 구조 변경이라 판단해 **의도적으로 보류**했으며, 이 문서를 그 후속 작업의 근거 자료로 남겨둔다.

- **처리됨 — §4 항목 5 (CLI의 `source_registry` 노출 불균형)**: `ssat/cli.py`의 `create_app()`이 이제 `AuditApplication`과 대칭적으로 `source_registry` 파라미터를 받아 그대로 전달한다. 회귀 테스트: `tests/integration/test_cli.py::test_cli_create_app_exposes_custom_source_registry`.
- **처리됨 — §4 항목 8 (`CallableAdapter` 미문서화)**: [`docs/CONFIG_REFERENCE.md#callable-adapter`](../CONFIG_REFERENCE.md#callable-adapter)에 `AdapterProvider` 서브클래싱 없이 `CallableAdapter` 하나로 새 모델을 연결하는 최소 예시를 추가했다.
- **추가로 발견·처리됨 (§3.4에서 다루지 않은 비대칭)**: `Metric` 레지스트리는 등록 자체는 쉽지만(§3.4), 그렇게 만든 커스텀 `MetricRegistry`를 실제로 실행에 반영할 방법이 `AuditApplication`에 없었다 — `compute_metrics()`가 내부에서 항상 `default_metric_registry()`를 새로 만들어 썼다. `AuditApplication(metric_registry=...)` 생성자 파라미터를 추가해 `adapter_registry`/`source_registry`와 같은 패턴으로 맞췄다. 회귀 테스트: `tests/integration/test_application_api.py::test_custom_metric_registry_is_used_end_to_end`.
- **의도적으로 보류 — PerturbationOperator 확장**: `OperatorFactory`가 등록은 지원하지만, `AuditApplication`에 이를 주입할 지점이 없다. §3.5 항목 6("공통 협력자의 장거리 시그니처 전파")이 지적한 대로 이 값을 실제로 실행에 반영하려면 `run_audit()` → `execution.py` → `pipeline.py` → `processors.py`의 `ChunkProcessor`까지 최소 4개 파일의 시그니처를 함께 바꿔야 해, "작은 대칭 수정"의 범위를 벗어난다고 판단해 보류했다. 현재 상태와 (`AuditApplication`을 우회하는) 유일한 우회 경로는 [`docs/APPLICATION_API.md#custom-perturbation-operators-not-yet-supported-here`](../APPLICATION_API.md#custom-perturbation-operators-not-yet-supported-here)에 정직하게 문서화했다.
- **의도적으로 보류 — Reporter 확장점 신설**: 리포트 생성에는 애초에 프로토콜/레지스트리 자체가 없다(§3.4). 새로 만드는 것은 문서 정리를 넘어서는 기능 추가이므로, 대신 "확장 불가, 대안은 `report_model.json`을 직접 소비"라고 [`docs/APPLICATION_API.md#reports-no-extension-point`](../APPLICATION_API.md#reports-no-extension-point)에 명시했다.

다음에 이 문서를 다시 참고할 사람을 위한 정리: §4의 나머지 항목(`AuditApplication` 분할, `ModelLoader`/`ModelRunner` 공통화, `StrategyRegistry[T]` 제네릭화, provider 이름 정적 검증, `WorkItem.identity_payload()` 일원화, `region_id::region_instance_id` 키/`_check_region_geometry` 단일 소스화, `html_renderer.py` 카드 키 하드코딩 제거)은 여전히 유효한 권고이며, 별도의 리팩터링 단계에서 하나씩 검토할 것을 권한다.
