# 구현 계획서 (v1 대조군·안정성 분석 모듈)
## Spatial Sensitivity Audit Toolkit — Control & Stability Analysis

> 본 계획서는 [CONTROL_STABILITY_DESIGN_v1.md](CONTROL_STABILITY_DESIGN_v1.md)의 설계를 현재 저장소 구현 상태와 대조하여 작성한 실행 계획이며, 방향과 틀을 정하기 위한 것이다. 세부 사항은 구현 과정에서 조정한다.
> 전제: 코어([IMPL_PLAN_CORE_v1.md](IMPL_PLAN_CORE_v1.md))와 지표 엔진([IMPLE_PLAN_METRIC_DESIGN_v1.md](IMPLE_PLAN_METRIC_DESIGN_v1.md))이 이미 구현되어 있다. 본 문서는 그 두 결과물(`ssat/core/*`, `ssat/metrics/*`)을 재사용하는 후속 단계를 다룬다.
> 패키지 위치: 대조군·안정성 모듈은 `ssat/analysis/`라는 **신규 최상위 패키지**로 구현한다(코어·지표 엔진과 형제 관계). 근거는 §3.2 참조.

---

## 1. 현재 구현 상태 대비 격차

설계서(A0~A7)와 현재 저장소를 대조한 결과, **본 모듈은 아직 한 줄도 구현되어 있지 않다.** `ssat/` 아래에는 `core/`, `application/`, `metrics/`, `utils/`만 존재하며 `analysis/`에 해당하는 디렉터리나 테스트는 없다.

다만 이번에는 지표 엔진 착수 시점과 달리 **선행 계층 두 개(코어, 지표 엔진)가 모두 이미 완성되어 있다.** 그래서 이 계획의 성격은 "무엇을 재사용할 수 있는가"보다 "선행 계층이 이 모듈이 필요로 하는 정보를 실제로 갖고 있는가"를 확인하는 쪽에 가깝다. 코드를 직접 대조한 결과, 설계서 §5 "잔여 결정 사항"의 코어 확인 필요 항목 두 개는 답이 나왔고, 설계서가 암묵적으로 전제한 부분에서 **설계서에 없던 격차 두 개**가 새로 발견되었다. 이 네 가지가 이 계획서 전체의 구조를 결정한다.

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| 1 | 대조군 참조 관계의 dump 기록 형식 (설계서 §5, 코어 확인 필요) | **있음 — 정확 매칭 가능** | `PlanBuilder._enumerate_items_for_sample`이 control `RegionSpec.params`에 `target_region`(대상 region의 `region_id`/`region_instance_id`/`kind`/`params`/`ref`/`ref_hash` 전체를 담은 자기완결적 recipe)을 그대로 심는다(`ssat/core/plan/builder.py:266-274`, `_region_recipe`). 이 값이 `region_params_json`으로 그대로 직렬화되어 dump에 저장된다(`ssat/core/dump/writer.py:320`). 면적 허용 오차 매칭(§A1)은 이 참조가 없는 예외 상황에 대한 방어적 폴백으로만 남는다. |
| 2 | jitter 조건의 코어 지원 여부 (설계서 §5, 코어 확인 필요) | **미지원 — A3(b)는 인터페이스만** | `RegionKind`(`ssat/core/types.py`), config, plan, region 어디에도 "jitter"라는 개념이 없다(`grep -rn jitter ssat/` 결과 0건). 설계서 자신이 이미 이 경우의 처리 방침(§5 "미지원이면 A3(b)는 인터페이스만")을 정해 두었으므로 그대로 따른다. |
| 3 | **(신규 발견) 저장된 지표에는 AnchorKey/ConditionKey를 재구성할 컬럼이 없다** | 격차 — A0/A1이 직접 메워야 함 | `metrics/types.py`의 `ItemMetrics`는 `item_id, sample_id, metric_name, value_clean, value_perturbed, degradation, available, excluded_reason`만 갖는다. `region_key`, `perturb_op`, `perturb_params`, `seed`, `invert_mask`, `is_control` 중 어느 것도 없다. `spatial_profile.parquet`는 `region_key`를 갖지만 이미 조건(모든 perturb_op·seed)에 걸쳐 평균낸 뒤라 조건축이 사라져 있다. 즉 **`ssat/metrics/`의 영속 산출물(parquet 다섯 개) 중 어느 것도 이 모듈의 입력으로 충분하지 않다.** 유일하게 조건축을 보존한 원천은 `ssat.metrics.dump_reader.DumpHandle.items()`(원본 dump)이며, 이 값에는 `region_id`, `region_instance_id`, `region_kind`, `region_params_json`, `perturb_op`, `perturb_params_json`, `invert_mask`, `is_control`, `seed_used`이 전부 있다(`ssat/metrics/dump_reader.py:45-65`). **결론: A0는 `load_metrics()`(값)와 `DumpHandle.items()`(조건 맥락)를 반드시 함께 열어 `item_id`로 조인해야 하며, 이후 모든 A-모듈은 `region_metrics.parquet`/`sample_metrics.parquet`/`class_metrics.parquet`를 입력으로 쓰지 않는다.** (§3.1, §5 단계 1) |
| 4 | **(신규 발견) `seed_salt`는 어디에도 저장되지 않는다** | 격차 — ConditionKey의 seed 성분 재정의 필요 | dump에 저장되는 것은 `seed_used`(`ssat/core/dump/schema.py:55`)뿐이고, 이는 `derive(global_seed, item_id, seed_salt)`(`ssat/core/perturb/rng.py`)로 **item마다 다르게 파생된 해시**다. `item_id` 자체가 region·op·params·seed_salt·is_control 전체의 해시이므로, 같은 `seed_salt`라도 region이 다르면(target vs control 등) `seed_used`가 완전히 달라져 앵커 간 비교에 쓸 수 없다. 설계서 §2의 `ConditionKey = (perturb_op, perturb_params_hash, seed)`를 문자 그대로 구현할 수 없다는 뜻이다. **해결(확정): ConditionKey는 `(perturb_op, perturb_params_hash)`만으로 정의한다.** seed는 "같은 AnchorKey·같은 ConditionKey를 공유하는 item이 몇 개인가"(반복 시행 횟수)를 세는 데만 쓰고, 서로 다른 AnchorKey 사이에서 seed 값을 비교하거나 매칭 조건으로 쓰지 않는다. 이 정의로도 설계서가 실제로 요구하는 두 가지 — (a) A3(a) seed 안정성(한 AnchorKey 안에서 반복 시행 간 분산), (b) A2 대조군 매칭(target·control 모두 동일 `perturbation` 루프에서 생성되므로 `invert_mask`·`perturb_op`·`perturb_params`가 애초에 동일함, `ssat/core/plan/builder.py:276-288`) — 은 그대로 충족된다. 이 재정의는 설계 의도를 바꾸지 않고 dump가 실제로 노출하는 정보에 맞춘 것이다. |

**부수 확인.** `is_control` 아이템은 `item_metrics.parquet`에 남아 있다 — 집계(N3, `aggregate_item_metrics`)에서만 제외되고(`ssat/metrics/aggregate.py:129`), `save_metrics`에 전달되는 원본 `item_metrics` 리스트에는 필터링 없이 그대로 저장된다(`ssat/application/application.py:411-423`). 즉 **대조군의 지표 값 자체는 이미 계산·저장되어 있다** — 새로 계산할 필요 없이 지표 저장소와 dump를 조인하기만 하면 된다.

**결론.** 이 계획서는 설계서 §1~§6 전체를 다루는 **신규 구현 계획**이며, 코어와 지표 엔진이라는 두 계층 위에 세 번째 해석 계층을 쌓는 작업이다. 핵심 난점은 새 통계량 구현이 아니라 — 그것들은 대부분 손계산 가능한 수준이다 — **지표 엔진이 의도적으로 지워버린 조건축을 dump에서 다시 복원하는 조인 계층(A0/A1)을 올바르게 만드는 것**이다.

---

## 2. 기술 스택과 의존성 방침

### 2.1 신규 의존성 없음

| 용도 | 방법 | 근거 |
|---|---|---|
| Spearman 순위 상관 | `pandas.Series.rank().corr()` (기본이 Pearson이므로 순위로 변환 후 상관) | scipy 없이 이미 `experiments/synthetic_shortcut/evaluate.py:111-124`와 `analyze_section35_sensitivity.py:96-112`에서 검증된 패턴을 그대로 재사용 |
| bootstrap CI | numpy 복원추출 + `np.percentile` | 표준 percentile bootstrap, 추가 라이브러리 불필요 |
| 연산자 쌍 상관 기반 군집화 | numpy로 구현한 단순 임계 기반 connected-components (§5 단계 5) | scikit-learn 등 신규 의존성 회피 |

`requirements.txt`, `scripts/install_deps.sh`, `.devcontainer/Dockerfile`은 변경하지 않는다 — 지표 엔진 계획(§2.1)이 확립한 방침을 그대로 잇는다.

### 2.2 CLI 표면은 v1 범위 밖으로 미룬다

지표 엔진도 처음엔 CLI 없이 시작해 단계 9 완료 후 `ssat metrics` 명령을 추가했다(`IMPLE_PLAN_METRIC_DESIGN_v1.md` §2.2). 이 모듈도 동일한 순서를 따른다: 먼저 라이브러리 API(`ssat/analysis/*`)로 완성하고, 실사용 스크립트(§5 단계 9)에서 반복 호출 패턴이 확인되면 `AuditApplication.analyze(...)` + `ssat analyze <dump> <metrics_dir>` CLI 추가를 후속 결정한다(§8에 미정으로 기록).

---

## 3. 디렉터리 구조

```
ssat/
├── core/                              # 기존 (변경 없음)
├── application/                       # 기존 (변경 없음)
├── metrics/                           # 기존 (변경 없음)
├── analysis/                          # ← v1 대조군·안정성 모듈 구현 범위 (신규)
│   ├── __init__.py
│   ├── types.py                       # AnchorKey · ConditionKey ·
│   │                                  #   각 산출 row 타입 (§3.1)
│   ├── errors.py                      # AnalysisError, AnalysisSchemaError,
│   │                                  #   AnalysisCorruptionError
│   ├── reader.py                      # A0  AnalysisReader
│   ├── indexer.py                     # A1  ComparisonIndexer
│   ├── control.py                     # A2  ControlComparator
│   ├── stability.py                   # A3  StabilityAnalyzer (seed/jitter/strategy)
│   ├── strategy_profile.py            # A4  StrategyProfiler
│   ├── interval.py                    # A5  IntervalEstimator
│   ├── reliability.py                 # A6  ReliabilityScorer
│   └── store.py                       # A7  AnalysisStore
├── utils/                             # 기존 (변경 없음)
└── ...
tests/
├── unit/
│   ├── test_analysis_types.py
│   ├── test_analysis_reader.py
│   ├── test_analysis_indexer.py
│   ├── test_analysis_control.py
│   ├── test_analysis_stability.py
│   ├── test_analysis_strategy_profile.py
│   ├── test_analysis_interval.py
│   ├── test_analysis_reliability.py
│   └── test_analysis_store.py
├── integration/
│   └── test_analysis_synthetic_dump.py  # B2: 코어 미실행, 합성 dump+지표 직접 주입
├── fixtures/
│   └── synthetic_dump_builder.py        # 기존 파일, 필요 시 확장(§5 단계 2)
└── ...
experiments/
└── synthetic_shortcut/
    └── analyze_control_stability.py      # B3 (단계 9에서 산출, pytest 밖)
```

### 3.1 구조 설계 의도

**`analysis/`가 A0~A7에 1:1 대응한다.** 지표 엔진의 `metrics/`가 N0~N5에 대응한 것과 동일한 원칙이다.

**`reader.py`가 이 모듈 내부의 유일한 이중 접근 지점이다.** §1의 항목 3에서 확정했듯, `analysis/reader.py`만이 `ssat.metrics.dump_reader.DumpHandle`(→ dump)과 `ssat.metrics.store.load_metrics`(→ 지표)을 함께 열어 `item_id`로 조인한다. `indexer.py` 이후의 모든 모듈은 `reader.py`가 만든 결합 프레임만 다루고, `ssat.metrics.dump_reader`나 `ssat.core.dump`를 직접 import하지 않는다. 지표 엔진이 `metrics/dump_reader.py`를 "코어와 후단의 유일한 계약 지점"으로 좁힌 것과 같은 이유다 — dump 스키마가 바뀌어도 파급 범위가 `reader.py` 한 파일로 좁혀진다.

**`indexer.py`(A1)와 `reader.py`(A0)를 분리한다.** A0는 "무엇이 존재하는가"(로드 + 가용성 보고)만 답하고, AnchorKey/ConditionKey 분해와 비교 가능 집합 구성은 A1의 책임으로 남긴다. 설계서 §1의 흐름도를 그대로 따른 것이며, 두 책임을 섞으면 "로드 실패"와 "비교 불가능"이라는 서로 다른 실패 모드가 한 함수 안에서 뒤섞인다.

**`stability.py` 하나가 seed/jitter/strategy 세 축을 모두 담는다(설계서 §A3와 동일).** 세 축은 "축을 섞지 않는다"는 원칙(설계서 §A3)을 코드 레벨에서도 지키기 위해 각각 독립 함수(`compute_seed_stability`, `compute_jitter_stability`, `compute_strategy_stability`)로 나누되, 세 함수가 공유하는 것은 A1이 만든 AnchorTable/조건 그룹뿐이다. 파일을 나누지 않는 이유는 세 축이 "같은 AnchorKey를 다른 조건으로 재보는" 동일한 상위 질문의 세 변주이기 때문이다 — 지표 엔진이 `builtin_metrics/flips.py`·`continuous.py`를 나눈 것과 달리, 여기서는 계산 대상(1순위/2순위 지표)이 아니라 축(seed/jitter/strategy)이 다를 뿐 로직 골격이 동일하다.

### 3.2 패키지 위치 근거 — `core/`·`metrics/` 밖에 둔다

지표 엔진 계획의 §3.2가 세운 원칙을 그대로 확장한다.

- 코어의 §2.2 의존 방향 규칙(단방향, 후단으로의 역참조 금지)과 지표 엔진의 §3.3이 세운 "`metrics.*`만 `core.dump`를 통해서 core에 접근한다"는 규칙은 그 자체로 "분석은 core의 하위 개념이 아니다"를 이미 강제하고 있다. `analysis/`를 `metrics/` 하위에 두면 지표 계산(N0~N5)과 결과 해석(A0~A7)이라는 서로 다른 책임이 한 패키지 안에 섞인다.
- 본 모듈은 **지표 엔진과 코어 양쪽 모두를 소비**한다(§1 항목 3) — 즉 코어 하위도, 지표 엔진 하위도 아니고 둘의 형제로 두는 것이 의존 관계를 정확히 반영한다.
- CONTROL_STABILITY_DESIGN_v1.md §0은 이 모듈을 "지표 엔진이 산출한 `ItemMetrics`를 입력으로 하여 ... 판정하는 단계"로 정의한다 — 지표 계산이 끝난 뒤의 해석 계층이라는 별도 책임이다.

### 3.3 의존 방향 규칙

```
analysis.types            → (없음)
analysis.errors            → (없음)
analysis.reader              → ssat.metrics (DumpHandle, load_metrics, verify_source_dump,
                                              ItemMetrics, MetricsManifest), ssat.utils
analysis.indexer               → analysis.types, analysis.reader의 산출물(결합 프레임)
analysis.control                  → analysis.types
analysis.stability                   → analysis.types
analysis.strategy_profile               → analysis.types, ssat.core.types(PerturbationOp; 선언적 속성 표 조회만)
analysis.interval                          → analysis.types
analysis.reliability                          → analysis.types
analysis.store                                   → analysis.types, ssat.utils
```

`ssat.analysis`는 `ssat.core.dump`를 직접 import하지 않고 `ssat.metrics.dump_reader.DumpHandle`을 통해서만 dump에 접근한다(§3.1). `analysis.strategy_profile`이 `ssat.core.types.PerturbationOp`를 import하는 것은 **선언적 속성 표(§5 단계 5)의 키로만 쓰는** 명시적 예외다 — 지표 엔진의 `metrics.viz → core.region` 예외와 같은 성격이다. `core/*`, `metrics/*` → `ssat.analysis` 역참조는 전면 금지한다. CI에서 import-linter로 강제하는 방식은 기존과 동일하게 따른다.

---

## 4. 개발 환경

기존 Dev Container / Docker Compose 워크스페이스 이미지를 그대로 사용한다. §2.1에서 확인한 대로 신규 의존성이 없으므로 컨테이너·CI 설정 변경이 필요 없다. 테스트는 컨테이너 안에서 `pytest`로 실행한다. B3(§5 단계 9)는 `experiments/synthetic_shortcut/results/`(gitignore 대상, §6)에 의존하므로 `.github/workflows/ci.yml`의 `pytest -q` 잡에는 합류하지 않는다 — 지표 엔진의 L3와 같은 위치다.

---

## 5. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행하며, 조건 미충족 시 다음 단계로 넘어가지 않는다.

---

### 단계 0. 스캐폴딩 + 계약 타입

> **가장 먼저 하는 이유:** 지표 엔진의 단계 0과 같은 이유다. 이후 모든 모듈이 이 타입을 참조하며, §1에서 확정한 ConditionKey 재정의가 여기서 코드로 고정된다.

**작업.**
- `analysis/types.py`:
  - `AnchorKey`(frozen dataclass: `sample_id`, `region_key`, `invert_mask`) — `region_key`는 지표 엔진과 동일한 `f"{region_id}::{region_instance_id}"` 관례를 그대로 따른다(`ssat/metrics/aggregate.py:155`, `ssat/metrics/viz/mask_check.py:186`와 동일 포맷; 공유 헬퍼로 추출하지 않고 문자열 그대로 재현한다 — 근거는 §9 리스크 표).
  - `ConditionKey`(frozen dataclass: `perturb_op`, `perturb_params_hash`) — **seed 필드를 두지 않는다**(§1 항목 4). `perturb_params_hash`는 dump에 이미 canonical JSON으로 저장된 `perturb_params_json` 문자열을 그대로 `hashlib.sha256(...).hexdigest()`한 값으로, `ssat.core.plan.hashing.canonical_json`을 다시 호출하지 않는다(이미 canonical하므로 재직렬화가 불필요하고, `ssat.core.plan`에 대한 새 의존을 만들지 않는다).
  - 산출 row 타입: `AnchorRow`, `ControlPairRow`, `ControlComparisonRow`, `SeedStabilityRow`, `StrategyStabilityRow`, `RankCorrelationRow`, `StrategyProfileRow`, `IntervalRow`, `ReliabilityRow`, `CoverageReport` — 설계서 §A7의 parquet 목록(`anchor_analysis`, `control_comparison`, `stability`, `rank_correlation`, `strategy_profile`, `intervals`)에 1:1 대응.
  - `FlagValue` enum(`TRUE`, `FALSE`, `UNAVAILABLE`) — 설계서 §A6 "`unavailable`을 `false`로 취급하지 않는다"를 타입으로 강제한다(3-state를 `bool | None`이 아니라 명시적 enum으로 두어, "계산 안 함"과 "계산했더니 거짓"을 실수로 섞을 수 없게 한다).
  - `ReliabilityGrade` enum(`HIGH`, `MODERATE`, `LOW`, `UNRELIABLE`).
- `analysis/errors.py`: `AnalysisError`(base), `AnalysisSchemaError`, `AnalysisCorruptionError` — 지표 엔진 `errors.py`와 동일한 성격.
- `analysis/__init__.py`에 공개 API 확정, 각 빈 모듈(`reader.py` 외 전부)에 인터페이스 stub만 배치.

**테스트.**
- `python -c "import ssat.analysis"` 성공
- `AnchorKey`/`ConditionKey`가 `frozen=True`이고 해시 가능함(딕셔너리·집합의 키로 쓰이므로 필수) 확인
- `FlagValue`의 `unavailable`이 `bool(FlagValue.UNAVAILABLE)`로 암묵 변환되지 않는지(즉 `if flag:` 같은 실수를 못 하게 하는지) 확인 — enum이므로 자연히 방지되지만, 회귀 테스트로 고정
- 각 row 타입의 필수/선택 필드 유효성 검증(빈 문자열, 범위 밖 값 등 지표 엔진과 동일한 엄격도)

**성공 조건.**
- 패키지 import 성공
- 타입 유효성·해시 가능성 단위 테스트 통과

---

### 단계 1. A0. AnalysisReader

**작업.** `analysis/reader.py`가 dump와 지표 저장소를 함께 연다.
- `AnalysisReader(dump_dir, metrics_dir)` 생성자에서:
  1. `ssat.metrics.store.load_metrics(metrics_dir)`로 `item_metrics`, `AggregationResult`, `MetricsManifest`를 로드
  2. `ssat.metrics.dump_reader.DumpHandle(dump_dir)`를 열고
  3. `ssat.metrics.store.verify_source_dump(manifest, handle.manifest_path)`를 즉시 호출해 **이 지표가 이 dump로부터 계산된 것인지** 확인(이미 구현된 함수를 그대로 재사용 — 새로 만들 필요 없음)
- `item_context()`: `DumpHandle.items()`(§1 항목 3의 결론에 따라 `joined()`가 아니라 `items()` — clean 쪽 정보는 이 모듈에 필요 없다)를 읽어 `item_id` 기준 조인 가능한 컨텍스트 프레임을 만든다. 컬럼: `item_id, sample_id, region_id, region_instance_id, region_kind, region_params_json, intended_area_px, effective_area_px, perturb_op, perturb_params_json, invert_mask, is_control, seed_used`.
- `available_analyses()`: 설계서 §A0의 가용성 보고를 그대로 산출.

  | 키 | 판정 |
  |---|---|
  | `control_comparison` | `item_context()["is_control"].any()` |
  | `fill_strategy_stability` | non-control 아이템의 `perturb_op` 고유값 수 ≥ 2 |
  | `seed_stability` | `(sample_id, region_key, invert_mask, perturb_op, perturb_params_hash)`로 묶었을 때 크기 ≥ 2인 그룹이 하나라도 존재(§1 항목 4의 재정의된 ConditionKey 적용) |
  | `jitter_stability` | 항상 `False` — §1 항목 2에 따라 코어가 이 축을 아예 만들어낼 수 없으므로 고정값. 코드 주석과 `available_analyses` 자체에 이유를 남긴다(설계서 §A0 "왜 이 항목이 비어 있는가"가 드러나야 한다) |

**테스트.**
- 지표 엔진의 committed fixture(`tests/fixtures/synthetic_classification/`)로 실제 dump+지표 조합을 만들어 `item_context()`의 행 수·컬럼이 `DumpHandle.items()`와 일치
- 지표 계산 후 dump를 재실행(다른 `run_manifest.json`)한 상황을 흉내내 `verify_source_dump` 실패가 `AnalysisReader` 생성 시점에 그대로 전파되는지 확인
- `available_analyses()`: 대조군 없음 / 있음, op 1개 / 2개 이상, 조건 반복 있음 / 없음의 조합을 `tests/fixtures/synthetic_dump_builder.py`로 구성해 4가지 불리언이 모두 기대와 일치
- `jitter_stability`가 어떤 입력에서도 `False`로 고정됨을 회귀 테스트로 고정(코어가 나중에 jitter를 지원하게 되면 이 테스트가 먼저 깨져서 갱신 필요성을 알린다)

**성공 조건.**
- `item_context()` 조인이 `DumpHandle.items()`와 행·컬럼 단위로 일치
- `available_analyses()` 4개 키 전부 시나리오별 기댓값과 일치
- 무결성 검증(hash mismatch) 실패가 올바르게 전파

---

### 단계 2. A1. ComparisonIndexer

**작업.** `analysis/indexer.py`가 A0의 결합 프레임을 AnchorKey/ConditionKey로 분해한다.
- `AnchorTable` 구성: `item_context()`를 `(sample_id, region_key, invert_mask)`로 그룹화. `intended_area_px`/`effective_area_px`가 같은 `region_key` 안에서 다르면 지표 엔진의 `_check_region_geometry`(`ssat/metrics/aggregate.py:194-232`)와 같은 정책으로 `AnalysisCorruptionError`를 던진다 — 지표 엔진이 이미 dump 작성 시점에 한 번 검증했더라도, 이 모듈은 dump를 독립적으로 다시 읽으므로 같은 방어를 반복한다.
- `ConditionKey` 부여: `(perturb_op, sha256(perturb_params_json))`. 같은 `(AnchorKey, ConditionKey)`를 공유하는 item 개수가 `n_conditions`(seed 반복 횟수, §1 항목 4).
- `ControlPairs` 구성(대조군 짝짓기, 설계서 §A1 규칙 그대로):
  1. `is_control=True`인 각 item의 `region_params_json`을 파싱해 `target_region` 키를 추출한다. 이 recipe는 `PlanBuilder._region_recipe`(`ssat/core/plan/builder.py:327-338`)가 만든 형식과 정확히 일치하므로, `target_region_key = f"{target_region['region_id']}::{target_region['region_instance_id']}"`로 target AnchorKey를 직접 구성할 수 있다.
  2. control과 target은 같은 `perturbation` 설정 루프에서 만들어지므로(`ssat/core/plan/builder.py:276-288`) `invert_mask`·`perturb_op`·`perturb_params_json`이 항상 동일하다 — 즉 **정상 경로에서는 ConditionKey가 이미 일치가 보장되어 있다.** `area_match_ratio`는 control의 `intended_area_px` / target의 `intended_area_px`로 계산해 부가 정보로 기록한다.
  3. `target_region` 파싱이 실패하거나(방어적 경로 — 현재 코어가 만들어내는 dump에서는 발생하지 않음, §1 항목 1) 참조된 target AnchorKey가 AnchorTable에 없는 경우: `area_match_tolerance`(기본 5%, 설계서 §5) 내에서 같은 `ConditionKey`를 가진 target 후보를 탐색하는 폴백으로 넘어가고, 그래도 없으면 `unmatched`로 기록한다.
- `coverage_report` 누적: `n_anchors`, `n_conditions_insufficient`(n_conditions < 2), `n_controls_unmatched`, `n_area_mismatch_warnings`.
- `n_conditions < 2`인 AnchorKey는 안정성 계산 대상에서 제외하고 `insufficient_conditions`로 표시(설계서 §2 규칙)만 하고 여기서 필터링하지는 않는다 — 필터링은 소비하는 A3/A5의 책임.

**테스트.**
- `tests/fixtures/synthetic_dump_builder.py`로 target region 1개 + control n개(같은 `target_region` recipe 참조)를 구성해 `ControlPairs`가 전량 정확 매칭되는지 확인 — **이것이 정상 경로의 기본 케이스**이므로 가장 먼저 통과해야 한다
- `target_region` 파싱 실패를 흉내낸 malformed 대조군 item으로 면적 허용 오차 폴백 경로 exercise
- 면적이 허용 오차를 벗어난 대조군 → `unmatched` + `coverage_report` 반영
- `n_conditions=1`인 AnchorKey → `insufficient_conditions` 플래그
- region_kind/area 불일치를 흉내낸 조작된 컨텍스트 프레임 → `AnalysisCorruptionError`

**성공 조건.**
- 코어가 실제로 생성하는 형식(§1 항목 1)의 대조군은 100% 정확 매칭 경로로 처리됨을 테스트로 고정
- 폴백·미매칭·불충분 조건 케이스 전부 통과

---

### 단계 3. A2. ControlComparator

**작업.** `analysis/control.py`가 각 (target AnchorKey, ConditionKey, metric_name)에 대해 `control_mean`, `control_std`, `n_controls`, `excess`, `ratio`, `z_vs_control`을 계산한다.
- `ratio`: `|control_mean|`이 임계(설정 가능, 기본 `1e-6`) 미만이면 `None`(설계서 §A2 "분모가 0에 가까울 때 폭발").
- `z_vs_control`: `control_std == 0` 또는 `n_controls < 2`(표준편차 정의 불가)이면 `None` — 0 나눗셈을 조용히 삼키지 않는다.
- 대조군이 전혀 매칭되지 않은 target(A1의 `unmatched`)은 `control_available=FlagValue.UNAVAILABLE`로 표시하고 나머지 필드는 전부 `None`.

**테스트(B1 — 손계산).**
- 고정된 `control` 값 목록으로 `excess`/`ratio`/`z_vs_control` 계산이 손계산과 일치
- `control_std=0`(모든 control 값이 동일) → `z_vs_control=None`
- `control_mean≈0` → `ratio=None`, `excess`는 정상 계산
- target == control 평균 → `excess≈0`
- 대조군 없음 → 전 필드 `None` + `control_available=UNAVAILABLE`(설계서 §4.3 B2 표와 동일 시나리오)

**성공 조건.**
- B1 손계산 케이스 전부 일치
- 0 나눗셈·미정의 케이스가 예외 없이 `None`으로 처리됨

---

### 단계 4. A3. StabilityAnalyzer

**작업.** `analysis/stability.py`에 세 함수를 구현한다(§3.1 설계 의도).

**(a) `compute_seed_stability`.** 같은 `(AnchorKey, ConditionKey)`를 공유하는 item들(§1 항목 4에서 재정의한 "반복 시행")의 `degradation`으로 `seed_mean`, `seed_std`, `seed_cv`(`std/|mean|`, `mean≈0`이면 `None`), `n_seeds`를 계산.

**(b) `compute_jitter_stability`.** **인터페이스만 구현한다**(§1 항목 2). 함수는 존재하되 항상 `jitter_available=FlagValue.UNAVAILABLE`인 결과를 반환하고, 코드 주석에 "코어가 jitter 마스크 변주를 지원하기 전까지 도달 불가능"이라고 명시한다 — 지표 엔진 단계 2가 `output_kind=probs` 분기를 방어적으로만 구현한 것과 같은 패턴.

**(c) `compute_strategy_stability`.**
- per-anchor: 같은 AnchorKey 안에서 `perturb_op`별로 그룹화(하나의 op에 여러 seed 반복이 있으면 먼저 반복 평균을 내어 op당 값 하나로 축약) → `strategy_signs`(부호), `strategy_values`, `sign_agreement_ratio`(최빈 부호 비율), `n_strategies`.
- **데이터셋 수준 순위 상관은 `region_metrics.parquet`를 쓰지 않는다.** §1 항목 3에서 확인했듯 그 파일은 이미 모든 `perturb_op`를 뭉뚱그린 뒤이므로 "op별 region 순위"라는 이 계산의 전제 자체를 만족하지 못한다. 대신 A1의 AnchorTable + item 컨텍스트에서 **op별로** `region_key → mean(degradation)`을 직접 재집계한다(로직은 지표 엔진의 `_aggregate_region_metrics`와 비슷하지만 `perturb_op`로 먼저 층화한다는 점이 다르다).
- 이렇게 만든 op별 region 순위 테이블에 대해 모든 `(op_a, op_b)` 쌍의 Spearman을 `pandas.Series.rank().corr()`로 계산(§2.1). `spearman_excl_top1`은 데이터셋 전체에서 (기준 op 기준) 상위 `k`개 region을 제외하고 재계산 — `k`의 기본값은 1(설계서 §3.5/L3 재분석의 실측 관례, §8에 기록).

**테스트(B2).**
- `synthetic_dump_builder`를 다중 `perturb_op`(≥3) × 다중 region(≥4)을 갖도록 구성해 `sign_agreement_ratio`·`n_strategies`가 수동 계산과 일치
- 부호가 절반씩 갈리는 합성 앵커 → `sign_agreement_ratio=0.5`
- 알려진 순위(예: 4개 region에 대해 한 op만 순서를 뒤집은 값)로 `spearman`/`spearman_excl_top1`을 pandas로 직접 계산한 기댓값과 대조
- 최상위 1개 region이 모든 op에서 압도적으로 커서 상관을 인위적으로 끌어올리는 합성 시나리오 → `spearman`과 `spearman_excl_top1`이 유의미하게 달라짐을 확인(설계서 §A3(c)가 요구하는 핵심 성질)
- `compute_jitter_stability`가 항상 `UNAVAILABLE`을 반환함을 회귀 테스트로 고정

**성공 조건.**
- seed/strategy 두 축의 B2 테스트 전부 통과
- jitter 인터페이스 stub의 고정 동작 확인

---

### 단계 5. A4. StrategyProfiler

**작업.** `analysis/strategy_profile.py`.
- **선언적 속성 표**(설계서 §A4): `PerturbationOp`(코어의 5개 값)마다 `preserves_statistics`, `preserves_local_texture`, `is_global_operation`을 하드코딩한 딕셔너리로 둔다. 설계서가 명시한 값(`preserves_statistics`: mean_fill/blur/patch_shuffle=True, constant_fill/gaussian_noise=False)은 그대로 쓰고, 설계서가 예시로만 언급한 나머지 칸(`preserves_local_texture`, `is_global_operation`의 전체 5개 op)은 이 단계에서 **제안값으로 확정**한다(§8 표에 근거와 함께 기록, 변경 가능).
- **경험적 그룹**: 단계 4가 만든 op×op Spearman 상관 행렬에 **단순 임계 기반 connected components**(설계서 §5 "단순 임계 vs 계층 군집" — v1은 단순 임계를 선택, 계층 군집은 필요 시 후속)를 적용해 클러스터를 만든다. 임계 기본값 0.3(제안값, §8).
- **정합성 보고**: 선언적 그룹(`preserves_statistics` true/false 이분)과 경험적 클러스터를 집합 비교해 `aligned`(완전 일치) / `partial`(Jaccard ≥ 0.5) / `divergent`(그 외)로 분류.
- **부호 그룹 요약**: op별 `mean_degradation_excl_top`(상위 `k`개 region 제외 평균, k는 단계 4와 동일 기본값 1), `sign_ratio_positive`, `n_anchors`.

**테스트.**
- 단계 4의 다중-op 합성 fixture를 재사용해, 상관이 강한 두 op가 같은 클러스터로 묶이고 약한 op가 분리되는지 확인
- 선언과 경험이 완전히 일치하는 합성 시나리오 → `aligned`
- 선언과 경험이 반대로 갈리는 합성 시나리오(예: `preserves_statistics=True`인 두 op를 서로 반대 부호로 만든 fixture) → `divergent`
- `mean_degradation_excl_top`이 상위 region 포함/제외 시 실제로 달라짐을 확인

**성공 조건.**
- 군집화·정합성 분류 테스트 전부 통과

---

### 단계 6. A5. IntervalEstimator

**작업.** `analysis/interval.py`. 리샘플링 단위는 **샘플**(설계서 §A5). region 수준 point estimate(해당 region·조건의 표본 평균)에 대해 샘플을 복원추출 → 재계산 → `n_bootstrap`회 반복 → percentile CI(`ci_low`=2.5%ile, `ci_high`=97.5%ile, 둘 다 설정 가능). `excludes_zero = not (ci_low <= 0 <= ci_high)`. 아이템 수준에는 적용하지 않는다(설계서 §A5, 반복이 seed 개수뿐이라 표본 부족).

**테스트(B1).**
- 고정 RNG 시드로 알려진 분포(예: 평균 5, 분산 기지의 정규 근사 배열)에 대해 CI가 numpy로 직접 계산한 percentile과 일치
- 모든 값이 0인 입력 → CI가 0을 포함(`excludes_zero=False`)
- 평균이 뚜렷하게 0에서 떨어진 입력 → `excludes_zero=True`
- `n_bootstrap`을 늘렸을 때 CI 폭이 안정되는 방향(단조는 아니어도 대략적 수렴) 확인

**성공 조건.**
- CI 손계산 일치, `excludes_zero` 경계 케이스 정확

---

### 단계 7. A6. ReliabilityScorer

**작업.** `analysis/reliability.py`가 A2~A5의 산출물을 모아 (AnchorKey, metric_name) 단위로 플래그·등급·사유를 만든다.
- 플래그 7종(설계서 §A6 표) 전부 `FlagValue`(3-state)로 산출. 각 플래그의 판정 임계(`z_vs_control` 임계, `seed_cv` 임계 등)는 상수로 선언하고 `AnalysisManifest`에 기록(설계서 §A7 "임계를 manifest에 기록하는 이유").
- 등급 부여는 설계서 §A6 표 그대로 구현하되, **`sign_consistent=false`(부호 불일치)를 최하 등급 `unreliable`의 유일한 필요조건**으로 둔다(설계서 §A6 "값이 작은 것이 아니라 방향이 갈리는 것").
- `reliability_reasons`: 각 플래그의 실제 값을 사람이 읽을 문장으로 변환(예: `"sign differs across fill strategies (3 positive, 2 negative)"`).

**테스트(B2 — 설계서 §4.3 표 그대로).**
- 모든 조건에서 동일 값 → `sign_consistent=true`, 변동 0, 등급 `high`
- 조건마다 부호 반대 → `sign_consistent=false`, 등급 `unreliable`
- control과 target 동일 → `excess≈0`, `exceeds_control=false`
- control 아이템 없음 → 대조군 지표 `None`, `control_available=UNAVAILABLE`, 등급이 `false`로 강등되지 않고 `unavailable` 사유로 반영됨(**`unavailable≠false` 원칙의 회귀 테스트**)
- 단일 조건만 존재 → `insufficient_conditions`, 안정성 지표 미계산
- 면적 불일치 control → `area_matched=false` 경고

**성공 조건.**
- 설계서 §4.3 B2 표 6개 시나리오 전부 통과 — 특히 "control 아이템 없음" 케이스에서 등급이 `false`로 오염되지 않는지가 이 단계의 핵심 성공 조건

---

### 단계 8. A7. AnalysisStore

**작업.** `analysis/store.py`가 `anchor_analysis.parquet`, `control_comparison.parquet`, `stability.parquet`, `rank_correlation.parquet`, `strategy_profile.parquet`, `intervals.parquet`, `coverage_report.json`, `analysis_manifest.json`을 `<run_dir>/analysis/`(기본값, `metrics/store.py`의 `<run_dir>/metrics/` 관례와 동일)에 저장한다. `ssat.metrics._storage.atomic_write_parquet`와 `ssat.utils.io.write_json_atomic`/`sha256_file`을 그대로 재사용한다.
- `AnalysisManifest`: `analysis_schema_version`, `source_metrics_manifest_hash`(`sha256_file(metrics_dir / "metrics_manifest.json")` — `MetricsManifest.source_run_manifest_hash`와 같은 체이닝 원칙), `available_analyses`(A0), `thresholds`(A2~A6에서 쓴 임계 전체), `n_bootstrap`, `random_seed`, `computed_at`, `grade_distribution`.
- `verify_source_metrics(manifest, metrics_manifest_path)`: `metrics.store.verify_source_dump`와 동일한 패턴으로, 분석 이후 지표가 재계산되면 즉시 감지.

**테스트.**
- 저장 → 재로드 시 값 일치(모든 산출 row 타입 왕복)
- `source_metrics_manifest_hash`가 실제 `metrics_manifest.json` 파일 해시와 일치
- 지표가 재계산되면(`metrics_manifest.json` 변경) `verify_source_metrics`가 감지
- `analysis_schema_version` 불일치 거부

**성공 조건.**
- 저장·재로드 일치, 두 단계(dump→지표→분석) 전체의 무결성 체이닝 테스트 통과

---

### 단계 9. B3. L3 재분석 (핵심 검증)

설계서 §4.4를 그대로 계승한다. **지표 엔진의 L3(신규 학습·신규 dump 생성)와 달리, 이 단계는 이미 완료된 실험 산출물을 재사용하는 것뿐이라 학습이나 감사 재실행이 필요 없다.** `experiments/synthetic_shortcut/results/{dumps,metrics}/shortcut_A_{constant_fill,mean_fill,blur,gaussian_noise,patch_shuffle}/`가 이미 로컬에 존재하며(`docs/L3_Synthetic-Shortcut Experiment Report.md`에 기록된 실행 결과), 이 5개 dump+지표 쌍에 본 모듈을 그대로 적용하기만 하면 된다.

**주의 — 이 데이터는 git에 커밋되어 있지 않다.** `experiments/synthetic_shortcut/results/`는 `.gitignore`의 `results/` 규칙에 걸려 추적되지 않는다(`git ls-files experiments/synthetic_shortcut/results` 결과 0건). 따라서 이 단계는:
- `experiments/synthetic_shortcut/analyze_control_stability.py`라는 **별도 스크립트**로 작성하고(지표 엔진의 `evaluate.py`/`analyze_section35_sensitivity.py`와 같은 위치),
- 로컬에 해당 디렉터리가 없는 환경(신규 체크아웃, CI)에서는 애초에 실행 대상이 아니라는 점을 스크립트 docstring과 이 문서 모두에 명시한다. pytest collection에도 포함하지 않는다(지표 엔진 L3와 동일한 이유).

**검증 질문(설계서 §4.4 표, 이미 알려진 기대값을 §8에 사전 등록).**

| 질문 | 기대 결과 | 근거 |
|---|---|---|
| 패치 region(0,0)의 등급 | `high` | Q1/Q2/Q4가 이미 PASS로 확인됨(`docs/L3_Synthetic-Shortcut Experiment Report.md`) — 부호 일치·5개 op 재현·대조 초과 조건을 이미 만족 |
| 비패치 15개 region의 등급 | 상당수 `unreliable` | 부호가 fill strategy에 따라 갈림(§Check 1: constant_fill/gaussian_noise 양수 평균 vs mean_fill/blur/patch_shuffle 음수 평균) |
| `spearman_excl_top1` 재현 | `mean_fill=-0.311, blur=-0.843, gaussian_noise=0.729, patch_shuffle=-0.218` (constant_fill 기준) | L3 리포트 "Check 2" 표와 정확히 일치해야 함 — 이미 사람이 수동 계산한 값이므로 이 모듈의 재현 여부를 판가름하는 가장 엄격한 기준 |
| 부호 그룹 도출 | `{constant_fill, gaussian_noise}` / `{mean_fill, blur, patch_shuffle}` | L3 리포트 "Check 1" non-patch mean 부호와 일치해야 함 |
| 선언·경험 정합성 | `aligned` 또는 `partial` | `preserves_statistics` 선언과 위 경험적 그룹이 정확히 대응 |

**성공 조건.**
- 위 5개 검증 질문의 재현 여부를 스크립트 출력으로 확보하고 보고. 특히 `spearman_excl_top1` 수치가 소수점 셋째 자리까지 일치하지 않으면 A1(op별 region 재집계) 또는 A3(c) 로직에 결함이 있다는 뜻이므로 원인을 A1까지 거슬러 올라가 확인한다.
- 기준 미달 시 설계서의 절차(A1 조인 로직 재검증 → A3(c) 재집계 로직 재검증 → L3 리포트 원본 재확인)를 순서대로 따른다.

---

## 6. 단계 간 의존과 병렬화

```
0 ──> 1 ──> 2 ──┬──> 3 ──┐
                 ├──> 4 ──┤
                 └──> ────┤──> 7 ──> 8 ──> 9
                     5(4 이후) ┘
                     6(2 이후) ┘
```

기본은 순차 진행을 권장하되, 다음은 앞당길 수 있다.

- **단계 3(A2 ControlComparator)과 단계 4(A3 StabilityAnalyzer)는 서로 독립적이다** — 둘 다 단계 2(A1)의 산출물만 있으면 되므로 병행 가능.
- **단계 6(A5 IntervalEstimator)은 단계 2(A1) 이후 바로 시작할 수 있다** — bootstrap 대상은 AnchorTable의 region 수준 값이지, A2~A4의 산출물에 의존하지 않는다.
- **단계 5(A4 StrategyProfiler)는 단계 4(A3, 특히 (c) fill strategy 축)가 만든 op×op 상관 행렬을 그대로 재사용하므로 단계 4 완료 후 시작한다.**
- **단계 7(A6 ReliabilityScorer)은 단계 3·4·5·6 전부가 끝나야 시작 가능하다** — 모든 플래그를 취합하는 단계이기 때문이다.

**단계 0~2가 병목이다.** 특히 단계 1(A0)에서 §1 항목 3·4의 조인·재정의를 잘못 구현하면, 그 위에 쌓이는 A2~A7 전부가 잘못된 조건축 위에서 계산된다.

---

## 7. 테스트 전략

| 계층 | 범위 | 대응 단계 | 실행 방식 |
|---|---|---|---|
| B1. 단위 정확성 | 개별 통계량 계산식(excess, z_vs_control, ratio, bootstrap CI, cv) | 단계 3, 6 | pytest, 손계산 가능한 고정 입력 |
| B2. 비교 가능성·집계 | AnchorKey/ConditionKey 분해, 대조군 매칭, 안정성·프로파일·신뢰도 로직 | 단계 1, 2, 4, 5, 7 | pytest, `synthetic_dump_builder`로 코어 미실행 합성 dump+지표 주입 |
| B3. L3 재분석 | 전체 파이프라인의 실사용 검증 | 단계 9 | 별도 스크립트, CI 밖, 로컬 산출물 필요 |

B1·B2는 지표 엔진의 `unit`/`integration` 계층과 동일하게 기본 `pytest` collection에 포함되어 `.github/workflows/ci.yml`의 단일 `pytest -q` 잡에 자연히 합류한다. GPU가 필요한 테스트는 없다(이 모듈은 pandas/numpy 연산과 parquet I/O만 수행).

---

## 8. 잔여 결정 사항의 처리 시점

| 항목 | 결정 시점 | 결정 내용(확정/제안) |
|---|---|---|
| 모듈 패키지 위치 | 이 계획서 작성 시 확정 | `ssat/analysis/` (§3.2) |
| ConditionKey의 seed 성분 | 이 계획서 작성 시 확정 | seed 필드 제외, `(perturb_op, perturb_params_hash)`만 사용(§1 항목 4) |
| 대조군 참조 관계 매칭 방식 | 이 계획서 작성 시 확정 | `region_params_json`의 `target_region` recipe로 정확 매칭이 기본 경로, 면적 허용 오차는 방어적 폴백(§1 항목 1, §5 단계 2) |
| jitter 조건 지원 | 이 계획서 작성 시 확정 | 코어 미지원 — A3(b)는 인터페이스만(§1 항목 2, §5 단계 4) |
| `area_match_tolerance` 기본값 | 단계 2 | 5%(설계서 §A1 기본값 그대로 채택), 변경 가능 |
| `exceeds_control`(z 임계)·`seed_stable`(cv 임계) | 단계 7, B3 재분석 후 재조정 | 제안값: z 임계 2.0, cv 임계 0.2 — 설계서 §5 "B3 재분석 결과를 보고 조정. 단, 조정 후 재보고"를 그대로 따름 |
| 경험적 그룹 군집화 방법 | 단계 5 | 단순 임계 기반 connected components, 임계 0.3(제안값). 계층 군집은 v1 범위 밖 |
| `spearman_excl_top1`/부호 그룹 요약의 제외 개수 `k` | 단계 4·5 | 기본 1(L3 실측 관례와 동일), 설정 가능 |
| `alignment_report`의 aligned/partial/divergent 경계 | 단계 5 | Jaccard 완전 일치→aligned, ≥0.5→partial, 그 외 divergent(제안값) |
| `preserves_local_texture`/`is_global_operation`의 5개 op 전체 값 | 단계 5 | 설계서가 예시로만 든 항목을 이 단계에서 5개 op 전부에 대해 확정(구현 시 표로 명시) |
| `n_bootstrap` 기본값 | 단계 6 | 제안 1,000, 실행 시간 실측 후 확정 |
| CLI 표면(`ssat analyze ...`) 추가 여부 | v1 범위 밖 → 단계 9 완료 후 후속 결정(§2.2) | 미정 |

---

## 9. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| A0가 dump와 지표를 조인하는 유일한 지점이라, 이 조인 로직의 버그가 이후 모든 단계로 조용히 전파됨 | 단계 1의 성공 조건을 "조인 결과가 `DumpHandle.items()`와 행·컬럼 단위로 정확히 일치"로 강하게 못박고, 단계 2~9는 이 결과만 소비하도록 강제(§3.1) |
| `region_key` 조립 공식(`f"{region_id}::{region_instance_id}"`)이 `metrics/aggregate.py`·`metrics/viz/mask_check.py`·`analysis/types.py` 세 곳에 중복됨 | 지금 시점에는 지표 엔진 코드를 건드리지 않기 위해 의도적으로 중복 허용(§5 단계 0). 세 번째 중복이 생긴 시점을 계기로 공유 헬퍼 추출을 지표 엔진 쪽에 별도 제안 |
| ConditionKey에서 seed를 뺀 것이 설계서 §2의 "같은 방식으로 몇 번째 시행" 의도를 놓칠 수 있음 | §1 항목 4에서 이 재정의가 A2(대조군 매칭)·A3(a)(seed 안정성) 두 실제 요구를 모두 충족함을 코드 이전에 이미 검증. 만약 향후 코어가 `seed_salt`를 dump에 직접 노출하도록 바뀌면(스키마 버전업), ConditionKey를 확장하는 것은 `analysis/types.py`와 `indexer.py`만의 국소적 변경 |
| `region_metrics.parquet`를 실수로 재사용해 op가 뭉개진 순위로 안정성을 계산할 위험 | §1 항목 3, §3.1에 "N3 집계 산출물은 이 모듈의 입력이 아니다"를 명시적 규칙으로 못박고, import-linter로 `analysis.*`가 `metrics.aggregate`의 반환 타입(`RegionMetrics` 등)을 소비하지 않는지 코드 리뷰로 확인 |
| B3가 로컬 미존재 데이터(`results/`, gitignore)에 의존해 재현성이 사람에 따라 갈림 | 스크립트와 이 문서 양쪽에 전제 조건을 명시하고, pytest collection에서 제외해 CI가 이 데이터 부재로 실패하지 않도록 함(§5 단계 9) |
| 등급 판정 임계(z, cv 등)가 근거 없는 제안값으로 굳어짐 | `AnalysisManifest`에 임계 전체를 기록해 재현성을 보장하고(§5 단계 8), B3 재분석 결과를 본 뒤 반드시 재조정·재보고하는 것을 §8에 명시적 절차로 남김 |
| 범위 확대 | 정식 HTML 리포트, 클러스터링·slice discovery, 검출 태스크, 외적 타당성 검증은 v1에서 손대지 않음(설계서 §0 제외 범위) |
