# 대조군·안정성 분석 모듈 설계 명세 (v1)
## Spatial Sensitivity Audit Toolkit — Control & Stability Analysis

---

## 0. 문서 범위

본 문서는 지표 엔진이 산출한 `ItemMetrics`를 입력으로 하여, **관측된 민감도가 신뢰할 만한 것인지 판정**하는 단계를 기술한다.

### 이 모듈이 답하는 질문

| 질문 | 담당 |
|---|---|
| 이 민감도가 영역 고유의 것인가, 단순 면적 효과인가? | 대조군 비교 |
| 교란 방식을 바꿔도 같은 결론이 나오는가? | 안정성 분석 |
| 서로 다른 교란 방식이 같은 것을 측정하는가? | strategy 프로파일링 |
| 이 결과를 얼마나 믿어야 하는가? | 신뢰도 라벨링 |

### 포함 범위

동일 면적 무작위 대조군 비교, 조건 간 안정성(fill strategy / seed / jitter), 교란 방식의 성격 분류와 부호 일치 분석, 신뢰구간 추정, 신뢰도 등급 부여.

### 제외 범위 (후속)

정식 HTML 리포트, 클러스터링·slice discovery, 검출 태스크, 외적 타당성 검증.

### 설계의 직접적 계기

L3 합성 실험의 부수 분석에서 다음이 관측되었다.

- 비패치 영역의 fill strategy 간 순위 상관이 −0.843 ~ +0.729로 크게 갈림
- 이 영역들의 degradation은 통계적으로 0과 구별됨 (노이즈가 아님)
- 부호가 두 그룹으로 나뉨: 분포 밖 값 주입(constant_fill, gaussian_noise) vs 원본 통계 보존(mean_fill, blur, patch_shuffle)

**따라서 본 모듈의 핵심 요구는 "평균을 내는 것"이 아니라 "불일치를 드러내는 것"이다.** 조건 간 결과를 뭉뚱그려 하나의 수치로 만들면 위와 같은 구조적 차이가 감춰진다.

### 핵심 원칙

**이 모듈은 신뢰도를 판정하되, 원인을 단정하지 않는다.** "이 결과는 조건에 따라 부호가 갈린다"까지 말하고, 그것이 마스크 아티팩트 때문인지 모델의 실제 성질인지는 사용자의 후속 판단에 맡긴다.

### Addendum (2026-08-13) — 부호 그룹 전제 재검토

위 "설계의 직접적 계기"가 관측된 L3 실행은 squeezenet1_0의 ImageNet 프리셋
(Resize(256)→CenterCrop(224))을 감사 시점 전처리로 썼고, 이 프리셋이
설정으로 오버라이드 불가능했다는 사실이 이후 발견되었다(구 STAGE9 §2, 아래
참고). CenterCrop 때문에 명목상 동일 면적인 4×4 grid 16개 셀의 모델 공간
실면적이 위치에 따라 2304/3072/4096px로 갈렸고, 이 면적 편차가 위 부호 그룹
분할과 강하게 상관되어 있었다(`docs/L3_Synthetic-Shortcut Experiment
Report.md`의 "Sign-Group Premise Re-examination" 절 참고).

전처리를 `Resize([224,224])`(crop 없음, 모든 셀 3136px로 균일)로 바꿔
`M_shortcut`/`M_normal`을 재학습하고 7개 run을 재감사한 결과:

- **부호 그룹 분할이 재현되지 않는다.** blur의 `spearman_excl_top1`이
  −0.843 → +0.004로 사실상 사라지며 부호가 바뀐다. mean_fill/patch_shuffle은
  같은 부호를 유지하지만 크기가 원래의 약 1/3로 줄어든다
  (−0.311→−0.111, −0.218→−0.061).
- **gaussian_noise의 강한 양의 상관은 그대로 유지된다** (+0.729→+0.693) —
  이것만은 면적과 무관한 실질적 효과로 보인다.
- 경험적 군집화 결과도 원래의 깔끔한 2그룹 분할 대신 3개 군집
  (`{blur, mean_fill}`, `{constant_fill, gaussian_noise}`,
  `{patch_shuffle}` 단독)으로 바뀐다.

**결론**: "설계의 직접적 계기"로 인용된 부호 그룹 분할은 상당 부분 CenterCrop
면적 아티팩트였다. 이것이 이 모듈 설계 자체를 무효화하지는 않는다 — crop
제거 후에도 fill strategy 간 부호·크기 불일치는 (약화된 형태로) 남아 있고,
"조건 간 결과를 평균내면 이 불일치가 감춰진다"는 위 핵심 원칙은 여전히
유효하다. 다만 원 관측의 **극적인 크기**(2그룹, −0.843~+0.729의 스프레드)는
과장되어 있었다는 뜻이며, A4 경험적 그룹 분류·A6 등급 판정을 해석할 때
이 축소된 크기를 기준으로 삼아야 한다. 상세 수치는
`docs/L3_Synthetic-Shortcut Experiment Report.md`를 참고.

---

## 1. 전체 흐름

```
[MetricsStore]
  item_metrics.parquet / sample_metrics / region_metrics / spatial_profile
        │
        ▼
┌─ A0. AnalysisReader ────────────────────────────────────────┐
│   metrics 로드 + dump manifest 조회                          │
│   metrics_schema_version 검증                                │
│   가용 조건 축 파악 (어떤 op/seed/control이 존재하는가)      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─ A1. ComparisonIndexer ─────────────────────────────────────┐
│   AnchorKey / ConditionKey 분해 (§2)                         │
│   비교 가능 집합 구성, 불완전 조합 보고                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬────────────────┐
        ▼              ▼              ▼                ▼
┌─ A2. Control ─┐ ┌─ A3. Stability ┐ ┌─ A4. Strategy ─┐ ┌─ A5. Interval ─┐
│  Comparator   │ │   Analyzer     │ │   Profiler     │ │   Estimator    │
│               │ │                │ │                │ │                │
│ 동일 면적     │ │ 조건 간 분산   │ │ 방식 성격 분류 │ │ bootstrap CI   │
│ 무작위 대비   │ │ 순위 상관      │ │ 부호 일치 분석 │ │ 0 포함 여부    │
│ 상대 민감도   │ │ 재현 여부      │ │ 그룹 정합성    │ │                │
└───────┬───────┘ └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        └─────────────────┴──────────────────┴──────────────────┘
                                   ▼
┌─ A6. ReliabilityScorer ─────────────────────────────────────┐
│   플래그 산출 → 등급 부여 (high/moderate/low/unreliable)     │
│   각 등급에 사유 명시                                        │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─ A7. AnalysisStore ─────────────────────────────────────────┐
│   parquet 저장 + analysis_manifest.json                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 기초 개념 — 비교 가능성

이 모듈의 모든 계산은 "무엇과 무엇을 비교할 수 있는가"에 달려 있다. 이를 두 개의 키로 분해한다.

### AnchorKey — 측정 대상

```
AnchorKey = (sample_id, region_key, invert_mask)
```

"어느 샘플의 어느 영역을 (제거 또는 보존) 했는가". **비교의 대상이 되는 단위**이다.

### ConditionKey — 측정 방법

```
ConditionKey = (perturb_op, perturb_params_hash, seed)
```

"그 영역을 어떤 방식으로 몇 번째 시행으로 교란했는가". **변주되는 축**이다.

### 두 종류의 비교

| 비교 | 고정 | 변주 | 담당 |
|---|---|---|---|
| **안정성** | AnchorKey | ConditionKey | A3, A4, A5 |
| **대조군** | ConditionKey | AnchorKey (target ↔ control) | A2 |

이 분리가 설계의 뼈대이다. 안정성은 "같은 것을 다르게 재면 같은 값이 나오는가"이고, 대조군은 "같은 방식으로 다른 것을 재면 구별되는가"이다.

### 불완전 조합의 처리

모든 (AnchorKey × ConditionKey) 조합이 dump에 존재한다는 보장이 없다. 실패, 부분 실행, 설정 변경 등의 이유가 있다.

**규칙.**
- 안정성 계산은 **해당 AnchorKey에 실제로 존재하는 ConditionKey들**로만 수행하고, `n_conditions`를 항상 병기
- `n_conditions < 2`이면 안정성 지표를 계산하지 않고 `insufficient_conditions`로 표시
- 대조군 비교는 target과 control이 **동일 ConditionKey**를 가질 때만 수행. 짝이 없으면 `unmatched`로 표시
- 결측을 조용히 버리지 않고 `coverage_report`에 집계

---

## 3. 모듈 명세

### A0. AnalysisReader

**책임.**
- `metrics_manifest.json` 로드, `metrics_schema_version` 검증
- `source_run_manifest_hash` 대조로 dump–metrics 정합성 확인
- `ItemMetrics` 및 필요한 집계 로드
- **가용 조건 축 파악**: 존재하는 perturb_op 목록, seed 개수, control 아이템 유무, jitter 적용 여부

**가용성 보고.** 분석 시작 전 무엇이 가능하고 무엇이 불가능한지 명시한다.

```
available_analyses:
  control_comparison: false   # is_control 아이템 없음
  fill_strategy_stability: true   # 5개 op 존재
  seed_stability: false       # seed 1개뿐
  jitter_stability: false
```

**설계 의도.** 사용자가 대조군을 요청하지 않았거나 seed 반복을 하지 않았다면, 해당 분석은 불가능하다. 이를 조용히 건너뛰지 않고 **명시적으로 보고**하여, 리포트에서 "왜 이 항목이 비어 있는가"가 드러나게 한다.

---

### A1. ComparisonIndexer

**책임.** `ItemMetrics`를 AnchorKey / ConditionKey로 분해하고 비교 가능 집합을 구성한다.

**산출.**
```
AnchorTable:  anchor_key, sample_id, region_key, invert_mask,
              intended_area_px, effective_area_px, is_control,
              n_conditions, condition_keys[]

ControlPairs: target_anchor_key, control_anchor_key, condition_key,
              area_match_ratio
```

**대조군 짝짓기 규칙.**

코어에서 `random_area_match`로 생성된 control은 특정 target의 면적을 참조한다. 그 참조 관계가 `region_params_json`에 기록되어 있어야 짝짓기가 가능하다.

- 참조 관계가 명시된 경우: 그대로 사용
- 명시되지 않은 경우: **면적 허용 오차 내**에서 매칭 (`area_match_tolerance`, 기본 5%)
- 매칭 실패 시 `unmatched`로 기록하고 제외

**면적 기준.** 매칭은 `effective_area_px`(전처리 통과 후 실제 면적)를 우선 사용한다. 사용 불가 시 `intended_area_px`로 대체하고 플래그를 남긴다.

---

### A2. ControlComparator

**목적.** 관측된 민감도가 영역 고유의 것인지, 단순히 "무언가를 가렸기 때문"인지 구별한다.

**계산.** 각 (target AnchorKey, ConditionKey)에 대해:

| 지표 | 정의 |
|---|---|
| `control_mean` | 짝지어진 control들의 degradation 평균 |
| `control_std` | 동일 표준편차 |
| `n_controls` | 짝지어진 control 개수 |
| `excess` | `target_degradation − control_mean` |
| `ratio` | `target_degradation / control_mean` (control_mean이 0 근처면 미정의) |
| `z_vs_control` | `(target − control_mean) / control_std` |

**권장 주지표는 `excess`와 `z_vs_control`이다.** `ratio`는 분모가 0에 가까울 때 폭발하므로 보조로만 사용한다.

> L3에서 Q2 배수 35.1은 인상적이지만, 이는 비패치 평균이 작았기 때문이기도 하다. 비율 지표의 이런 성질을 문서에 명시하고, 논문에서도 excess를 병기하는 것이 안전하다.

**control 부재 시.** 해당 target의 대조군 지표는 전부 null이며 `control_available=false`로 표시한다. 계산을 생략하되 그 사실이 드러나야 한다.

**면적 불일치 보고.** `area_match_ratio`가 허용 오차를 벗어나면 경고 플래그를 남긴다.

---

### A3. StabilityAnalyzer

**목적.** 같은 AnchorKey를 여러 조건에서 측정했을 때 결과가 일관되는지 본다.

세 축을 분리해서 다룬다. **섞으면 원인을 알 수 없다.**

#### (a) Seed 안정성

동일 op·params에서 seed만 다른 경우. 순수한 확률적 변동을 측정한다.

```
seed_mean, seed_std, seed_cv (변동계수), n_seeds
```

**해석.** 여기서 변동이 크면 교란 자체가 불안정한 것이다. 다른 어떤 분석보다 먼저 확인해야 한다.

#### (b) Jitter 안정성

마스크 경계를 미세 이동시킨 조건 간 비교. 영역 정의의 민감성을 본다.

```
jitter_mean, jitter_std, jitter_range, n_jitters
```

**해석.** 변동이 크면 영역 경계가 결과를 좌우한다는 뜻이며, 영역 크기·정렬을 재고해야 한다.

#### (c) Fill strategy 안정성

서로 다른 교란 연산자 간 비교. **가장 중요하고 가장 복잡한 축**이다.

```
per-anchor:
  strategy_signs        # 각 op에서의 degradation 부호
  sign_agreement_ratio  # 최빈 부호의 비율
  strategy_values       # op별 degradation
  n_strategies
```

**핵심 설계 결정: 여기서 단순 평균을 내지 않는다.**

L3에서 확인되었듯 서로 다른 op는 반대 부호를 낼 수 있다. 평균을 내면 상쇄되어 "민감도 없음"으로 보이지만, 실제로는 "조건에 따라 반대로 반응"이다. 이 둘은 완전히 다른 상황이다.

따라서 A3는 **op별 값을 보존한 채 일치도만 계산**하고, 통합은 A6(신뢰도 판정)에서 판단과 함께 수행한다.

#### 영역 순위 상관 (데이터셋 수준)

각 op 쌍에 대해 region 순위의 Spearman 상관을 계산한다.

```
RankCorrelation: op_a, op_b, spearman, n_regions,
                 spearman_excl_top1, scope
```

**`spearman_excl_top1`을 반드시 함께 산출한다.** L3에서 확인되었듯, 모든 op에서 1위인 강한 신호가 상관을 인위적으로 끌어올린다. 상위 영역을 제외한 상관이 약한 신호의 실제 안정성을 보여준다.

`scope`는 상관 계산 범위(전체 / 상위 제외 / 특정 샘플)를 기록한다.

---

### A4. StrategyProfiler

**목적.** 교란 방식들이 서로 무엇을 다르게 측정하는지 규명한다. L3 통찰의 직접적 산물.

#### 선언적 속성 (연산자 메타데이터)

각 Perturbator에 다음 속성을 선언한다. 이는 **연산자 고유의 성질**이다.

| 속성 | 의미 | 예 |
|---|---|---|
| `preserves_statistics` | 원본의 색·밝기 통계를 보존하는가 | mean_fill, blur, patch_shuffle: true / constant_fill, gaussian_noise: false |
| `preserves_local_texture` | 국소 텍스처를 보존하는가 | patch_shuffle: true / blur: false |
| `is_global_operation` | 전체 프레임 연산 후 합성하는가 | blur: true |

#### 경험적 그룹 (데이터에서 추정)

동일 데이터에서 op 간 순위 상관을 계산하고 군집화하여 **경험적 그룹**을 도출한다.

```
EmpiricalGrouping: op, cluster_id, mean_corr_within, mean_corr_across
```

#### 정합성 보고 — 이것이 핵심 산출물

선언적 속성과 경험적 그룹이 일치하는지 대조한다.

```
alignment_report:
  declared_groups:   {preserves_statistics: [...], not: [...]}
  empirical_groups:  {cluster_0: [...], cluster_1: [...]}
  agreement: "aligned" | "partial" | "divergent"
```

**왜 둘 다 필요한가.**

부호가 갈리는 현상은 **연산자만의 성질이 아니라 모델과의 상호작용**이다. L3에서 비패치 영역이 음수였던 것은 M_shortcut이 그 영역을 무시했기 때문이며, 정상 모델에서는 다른 패턴이 나올 수 있다.

따라서 그룹을 하드코딩하면 잘못된 일반화가 된다. 선언적 속성은 **가설**로 제시하고, 경험적 그룹을 **관측**으로 제시하며, 둘의 정합 여부를 보고하는 것이 정직한 설계이다.

#### 부호 그룹 요약 (데이터셋 수준)

```
per-op: mean_degradation_excl_top,  # 상위 신호 제외한 평균
        sign_ratio_positive,
        n_anchors
```

L3의 §3.5 후속 분석이 산출한 표가 정확히 이 형태이다. 그 분석을 모듈 기능으로 정식화한 것이다.

---

### A5. IntervalEstimator

**목적.** 집계값의 불확실성을 정량화한다.

**Bootstrap 대상.** 리샘플링 단위는 **샘플**이다. region 수준 집계에서 샘플을 복원추출하여 재계산한다.

```
region_key, metric, point_estimate,
ci_low, ci_high, ci_method, n_bootstrap, excludes_zero (bool)
```

**설계 결정.**
- 기본 `n_bootstrap`은 설정 가능하되 기본값을 두고, 실행 시간을 CostEstimator처럼 사전 보고
- `excludes_zero`가 A6의 신뢰도 플래그로 직접 연결됨
- 아이템 수준에는 적용하지 않음 (반복이 seed 개수뿐이라 표본이 부족)

---

### A6. ReliabilityScorer

**목적.** 앞선 분석들을 종합해 각 결과에 신뢰도 등급을 부여한다.

#### 단일 점수가 아니라 플래그 집합

먼저 개별 플래그를 산출한다. 각 플래그는 독립적으로 해석 가능해야 한다.

| 플래그 | 조건 |
|---|---|
| `sign_consistent` | 모든 조건에서 degradation 부호 동일 |
| `exceeds_control` | `z_vs_control`이 임계 초과 |
| `seed_stable` | seed 변동계수가 임계 미만 |
| `jitter_stable` | jitter 범위가 임계 미만 |
| `multi_strategy` | 2개 이상 op에서 재현 |
| `ci_excludes_zero` | bootstrap CI가 0을 포함하지 않음 |
| `area_matched` | 대조군 면적 매칭이 허용 오차 내 |

각 플래그는 `true / false / unavailable` 세 값을 갖는다. **`unavailable`을 `false`로 취급하지 않는다.** 대조군을 요청하지 않은 것과 대조군을 넘지 못한 것은 다르다.

#### 등급 부여

| 등급 | 조건 (초안) |
|---|---|
| `high` | `sign_consistent` + `exceeds_control` + `multi_strategy` + `ci_excludes_zero` |
| `moderate` | 위 중 일부 충족, 부정 플래그 없음 |
| `low` | 부호는 일치하나 대조군 대비 미미하거나 조건 수 부족 |
| `unreliable` | `sign_consistent=false` — 조건에 따라 방향이 뒤바뀜 |

**`unreliable`의 정의가 중요하다.** 값이 작은 것이 아니라 **방향이 갈리는 것**을 최하 등급으로 둔다. L3에서 확인된 상황이 정확히 이것이며, 이런 결과를 평균 내어 제시하면 사용자가 오도된다.

#### 사유 명시

등급만 주고 끝내지 않는다.

```
reliability_grade: "low"
reliability_reasons: [
  "sign differs across fill strategies (3 positive, 2 negative)",
  "control comparison unavailable (no control items in run)"
]
```

**설계 의도.** 등급은 요약이고, 사유가 실질이다. 사용자가 등급에 동의하지 않더라도 근거를 보고 스스로 판단할 수 있어야 한다.

---

### A7. AnalysisStore

**출력 구조.**
```
<run_dir>/analysis/
├── anchor_analysis.parquet      # AnchorKey 단위 종합
├── control_comparison.parquet
├── stability.parquet            # 축별 분리 컬럼
├── rank_correlation.parquet
├── strategy_profile.parquet
├── intervals.parquet
├── coverage_report.json         # 결측·불완전 조합 집계
└── analysis_manifest.json
```

**analysis_manifest.json.**
```
analysis_schema_version
source_metrics_manifest_hash
available_analyses            # A0의 가용성 보고
thresholds                    # 등급 판정에 쓴 임계 전체
n_bootstrap, random_seed
computed_at
grade_distribution            # 등급별 개수
```

**임계를 manifest에 기록하는 이유.** 신뢰도 등급은 임계에 의존한다. 임계가 기록되지 않으면 결과 재현과 비교가 불가능하다.

---

## 4. 검증 계획

### 4.1 층위

| 층위 | 대상 | 방법 |
|---|---|---|
| B1 | 개별 통계량 계산식 | 손계산 가능한 고정 입력 |
| B2 | 비교 가능성 처리 | 합성 metrics 주입 |
| B3 | **L3 재분석** | 기존 실행 결과에 모듈 적용 |

### 4.2 B1 — 단위 테스트

- `excess`, `z_vs_control`, `ratio`의 계산 정확성
- `control_std=0`일 때 `z_vs_control` 처리 (0 나눗셈)
- Spearman 상관과 `spearman_excl_top1`의 차이 검증
- bootstrap CI가 알려진 분포에서 기대 구간을 재현
- 변동계수 계산 시 평균이 0 근처인 경우 처리

### 4.3 B2 — 합성 metrics 주입

| 시나리오 | 기대 |
|---|---|
| 모든 조건에서 동일 값 | `sign_consistent=true`, 변동 0, 등급 `high` |
| 조건마다 부호 반대 | `sign_consistent=false`, 등급 `unreliable` |
| control과 target 동일 | `excess≈0`, `exceeds_control=false` |
| control 아이템 없음 | 대조군 지표 null, `unavailable`, 등급이 `false`로 강등되지 않음 |
| 단일 조건만 존재 | `insufficient_conditions`, 안정성 지표 미계산 |
| 면적 불일치 control | `area_matched=false` 경고 |

### 4.4 B3 — L3 재분석 (핵심 검증)

**이미 완료된 L3 실행 결과에 본 모듈을 적용**하여, 수동 후속 분석이 도출한 결과를 재현하는지 확인한다.

**검증 질문.**

| 질문 | 기대 결과 |
|---|---|
| 패치 영역의 등급 | `high` (부호 일치, 5개 op 재현, 대조 초과) |
| 비패치 영역의 등급 | 상당수 `unreliable` (부호 갈림) |
| `spearman_excl_top1` 재현 | 수동 분석값(−0.843 등)과 일치 |
| 부호 그룹 도출 | 경험적 그룹이 {constant_fill, gaussian_noise} / {mean_fill, blur, patch_shuffle}로 분리 |
| 선언·경험 정합성 | `aligned` 또는 `partial` |

**이 검증의 의의.** 모듈이 사람이 수동으로 발견한 구조를 자동으로 재현한다면, 도구가 실제 사용에서도 같은 통찰을 제공할 수 있음을 보인다. **논문에서 이 모듈의 유용성 근거로 직접 인용 가능하다.**

동시에, 아직 미검증 상태인 **부호 그룹 가설을 정식으로 확증하는 실험**이 된다(논문 메모 §5의 미실시 항목).

---

## 5. 잔여 결정 사항

| 항목 | 결정 시점 |
|---|---|
| 등급 판정 임계 구체값 | **결정됨 (2026-08-13)**: 제안값(z=2.0, cv=0.2) 유지, 조정하지 않음 — 근거와 남은 후속 조치는 `docs/RELIABILITY_THRESHOLD_CALIBRATION_v1.md` 참고 |
| 경험적 그룹 군집화 방법 | A4 구현 시 (단순 임계 vs 계층 군집) |
| `n_bootstrap` 기본값 | A5 구현 후 실행 시간 실측 |
| `area_match_tolerance` 기본값 | A1 구현 시 |
| control 참조 관계의 dump 기록 형식 | 코어 확인 필요 — 현재 `region_params_json`에 있는지 |
| jitter 조건의 코어 지원 여부 | 코어 확인 필요 — 미지원이면 A3(b)는 인터페이스만 |

**주의.** 마지막 두 항목은 코어 구현 상태에 의존한다. 착수 전 확인이 필요하다.

---

## 6. 설계 결정 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| 비교 구조 | AnchorKey / ConditionKey 분해 | 안정성과 대조군을 명확히 분리 |
| 조건 간 통합 | 단순 평균 금지, 값 보존 + 일치도 계산 | 반대 부호가 상쇄되어 감춰지는 것 방지 |
| 순위 상관 | `spearman_excl_top1` 필수 병기 | 강한 신호가 상관을 인위적으로 끌어올림 |
| strategy 그룹 | 선언적 속성 + 경험적 그룹 + 정합성 보고 | 부호는 연산자·모델 상호작용이라 하드코딩 불가 |
| 대조군 주지표 | `excess`, `z_vs_control` (ratio는 보조) | 비율은 분모가 작을 때 불안정 |
| 신뢰도 표현 | 플래그 집합 + 등급 + 사유 | 단일 점수는 판단 근거를 감춤 |
| 결측 처리 | `unavailable`을 `false`와 구분 | 미요청과 미충족은 다름 |
| 최하 등급 기준 | 값의 크기가 아니라 부호 불일치 | L3에서 확인된 실제 위험 |
| bootstrap 단위 | 샘플 복원추출 | 아이템 수준은 표본 부족 |
| 임계 기록 | manifest에 전부 저장 | 등급이 임계 의존적이므로 재현성 필수 |
| 첫 검증 | 기존 L3 실행 재분석 | 수동 발견 구조의 자동 재현 여부 |
