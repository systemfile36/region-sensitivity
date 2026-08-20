# 지표 엔진 설계 명세 및 검증 실험 계획 (v1)
## Spatial Sensitivity Audit Toolkit — Metrics Engine

---

## 0. 문서 범위

본 문서는 코어가 생성한 raw dump를 입력으로 하여 **진단 지표를 계산하고 최소 시각화를 제공하며, 도구 전체의 정확성을 검증**하는 단계를 기술한다.

### 포함 범위

| 구분 | 내용 |
|---|---|
| 지표 엔진 | dump 로드, clean–perturbed 결합, 오류 전환·연속 변화량 계산, 샘플·영역 축 집계 |
| 최소 시각화 | 디버깅 목적의 공간 히트맵 오버레이, 샘플 랭킹 확인 |
| 검증 실험 | 합성 shortcut 실험을 통한 end-to-end 정확성 판정 |

### 제외 범위 (후속)

대조군 비교 분석, 안정성(fill/seed/jitter) 분석, 클러스터링, 정식 HTML 리포트, 검출 태스크 지표.

### v1 전제

- **이미지 분류만** 대상. 코어의 v1 범위와 일치
- 지표는 1~3순위만 구현. 4순위 이상은 인터페이스만 확보
- 시각화는 **정식 리포트가 아니라 디버깅 도구**. 완성도보다 오류 발견 능력이 목적

### 핵심 원칙

**지표 엔진은 관측을 해석으로 변환하되, 결론을 내리지 않는다.** "이 샘플이 취약하다"는 순위와 수치를 제공하지만, 그것이 지름길 학습인지 마스크 아티팩트인지는 판정하지 않는다(§2.3 해석 수준 구분).

---

## 1. 전체 흐름

```
[raw dump]
  clean.parquet / perturbed/chunk_*.parquet / index.parquet / run_manifest.json
        │
        ▼
┌─ N0. DumpReader ────────────────────────────────────────┐
│   스키마 버전 검증 → clean·perturbed 로드 → 결합         │
│   status 필터링 → JoinedFrame                            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─ N1. OutputNormalizer ──────────────────────────────────┐
│   logits/probs 판별 → 확률·logit·rank·margin 파생        │
│   clean·perturbed 동일 처리                              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─ N2. MetricRegistry ────────────────────────────────────┐
│   플러그인 등록된 Metric들을 순회 계산                    │
│   ┌────────────────┬──────────────────────┐             │
│   │ 오류 전환(1순위)│ 연속 변화량(2순위)   │             │
│   └────────────────┴──────────────────────┘             │
│   부호 정규화(degradation 축) 적용                       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
                 [ItemMetrics]  item_id 단위 long-form
                       │
                       ▼
┌─ N3. Aggregator ────────────────────────────────────────┐
│   축별 집계 (3순위)                                       │
│   sample / region / class / (sample × region)            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
        [SampleMetrics] [RegionMetrics] [ClassMetrics] [SpatialProfile]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌─ N4. MetricsStore ──┐      ┌─ N5. DebugViz ────────────┐
│  parquet 저장       │      │  공간 히트맵 오버레이      │
│  metrics_manifest   │      │  샘플 랭킹 대조 뷰         │
└─────────────────────┘      └───────────────────────────┘
```

---

## 2. 모듈 명세

### N0. DumpReader

코어와 후단의 **유일한 계약 지점**. 이후 어떤 모듈도 parquet을 직접 읽지 않는다.

**책임.**
- `run_manifest.json` 로드, `schema_version` 검증
- `clean.parquet` 및 `perturbed/chunk_*.parquet` 로드
- `status` 기반 필터링 및 결측 보고
- sample_id 기준 clean–perturbed 결합

**인터페이스(개념).**
```
open(run_dir) -> DumpHandle
DumpHandle.manifest  -> RunManifest
DumpHandle.clean()   -> DataFrame
DumpHandle.items()   -> DataFrame        # perturbed 전체
DumpHandle.joined()  -> JoinedFrame      # clean 컬럼이 접미사로 병합
DumpHandle.summary() -> dict             # status별 개수, 결측 현황
```

**JoinedFrame의 컬럼(개념).**
```
item_id, sample_id, gt_label,
region_kind, region_params_json, intended_area_px, effective_area_px,
perturb_op, perturb_params_json, invert_mask, is_control, seed_used,
logits_perturbed, logits_clean,
status_perturbed, status_clean
```

**status 처리 규칙.**

| 조합 | 처리 |
|---|---|
| clean=ok, perturbed=ok, gt_label 있음 | 정상 계산 |
| clean=ok, perturbed≠ok | 지표 null, `excluded_reason=perturbed_status_not_ok` 기록 |
| gt_label 없음(라벨 없는 추론 전용 감사, 코어가 이미 `gt_label: int \| None`로 지원) | perturbed 상태와 무관하게 지표 null, `excluded_reason=gt_label_unknown` 기록 — 현재 등록된 모든 지표가 정답 클래스를 전제하므로 계산 자체가 불가능 (IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md 작업 중 발견·수정: 이전에는 `int(row.gt_label)`을 무조건 호출해 전체 실행이 크래시했다) |
| clean≠ok | 해당 샘플 전체 제외, 별도 집계에 보고 |
| clean 레코드 없음 | 오류로 처리. dump 무결성 문제 |

**설계 의도.** 결측을 조용히 버리지 않는다. 모든 제외는 사유와 함께 집계되어 최종 리포트에 노출된다. 분모가 달라지는 것을 사용자가 알 수 있어야 한다.

**스키마 버전 정책.** manifest의 `schema_version`이 지원 목록에 없으면 **거부**한다(경고 후 진행하지 않음). 잘못된 해석보다 명확한 실패가 낫다.

---

### N1. OutputNormalizer

모델 출력을 지표 계산에 필요한 파생값으로 변환한다. clean과 perturbed에 동일하게 적용된다.

**입력.** 로짓 또는 확률 벡터, `AdapterSpec.output_kind`

**파생값.**

| 값 | 설명 |
|---|---|
| `prob` | softmax 적용 (output_kind=logits인 경우) 또는 원값 |
| `logit` | 원값 또는 log 변환 (probs인 경우, 수치 안정성 주의) |
| `top1_index`, `top1_prob` | 최대값 클래스 |
| `gt_prob`, `gt_logit` | 정답 클래스 값 |
| `gt_rank` | 정답 클래스의 순위 (1이 최상) |
| `margin` | `gt_logit − max(other logits)`. 양수면 정답이 1위 |
| `entropy` | 예측 분포의 엔트로피 (확장 지표용) |

**설계 의도.**
- 지표 구현체가 로짓을 직접 다루지 않게 하여 중복 계산과 불일치를 방지한다
- `output_kind`가 probs인 경우 logit 복원은 불완전하므로, 해당 지표는 `available=false`로 표시하고 계산하지 않는다

---

### N2. MetricRegistry

지표를 **플러그인으로 등록**하고 일괄 계산한다. 확장성의 핵심 지점.

**Metric 인터페이스(개념).**
```
name: str
requires: list[str]          # 필요한 파생값 (예: ["gt_prob", "top1_index"])
higher_is_better: bool       # 부호 정규화 방향
kind: "binary" | "continuous"
available_when: callable     # output_kind 등 조건
compute(clean_derived, perturbed_derived) -> float | bool | None
```

**부호 정규화.** 모든 지표는 `degradation` 축으로 변환하여 저장한다.
- `higher_is_better=True`인 지표: `degradation = clean − perturbed`
- `higher_is_better=False`인 지표: `degradation = perturbed − clean`

원시 값(`value_clean`, `value_perturbed`)도 함께 저장하여 사후 재해석이 가능하게 한다.

#### 1순위 — 오류 전환 지표 (binary)

| name | 정의 |
|---|---|
| `flip_correct_to_wrong` | clean에서 정답, perturbed에서 오답 |
| `flip_wrong_to_correct` | clean에서 오답, perturbed에서 정답 |
| `pred_changed` | top-1 클래스가 달라짐 (정오 무관) |
| `topk_exit` | 정답이 clean에서 top-k 내, perturbed에서 이탈 |

**필수 보고 규칙.** clean에서 이미 오답인 샘플과 정답인 샘플은 **반드시 분리 집계**한다. 섞으면 flip rate의 해석이 불가능하다.

#### 2순위 — 연속 변화량 지표 (continuous)

| name | 정의 | higher_is_better |
|---|---|---|
| `gt_prob_drop` | 정답 클래스 확률 감소량 | True |
| `gt_logit_drop` | 정답 클래스 logit 감소량 | True |
| `margin_drop` | 마진 감소량 | True |
| `loss_increase` | cross-entropy 증가량 | False |
| `gt_rank_worsening` | 정답 순위 악화량 | False |

**설계 의도.** 이진 전환은 임계 근처에서 불안정하다. 연속값은 "틀리진 않았지만 흔들렸다"를 포착하며, §5.3의 샘플 간 비교에 필요하다.

#### 확장 지점 (인터페이스만 확보)

3순위 이후 추가될 지표: 예측 분포 거리(KL, JS), 엔트로피 변화, 대조군 상대 지표, 안정성 지표. 이들은 Metric 인터페이스를 그대로 따르며 registry에 등록만 하면 동작한다.

**출력.** `ItemMetrics` — long-form
```
item_id, sample_id, metric_name,
value_clean, value_perturbed, degradation,
available (bool), excluded_reason (nullable)
```

**long-form을 택한 이유.** 지표가 추가되어도 스키마가 변하지 않는다. wide-form이면 지표마다 컬럼이 늘어 스키마 버전 관리가 복잡해진다.

---

### N3. Aggregator (3순위)

`ItemMetrics`를 축별로 집계한다.

#### 축별 산출물

**SampleMetrics** — sample_id 단위
```
sample_id, gt_label, clean_correct,
n_items, n_valid,
{metric}_mean, {metric}_max, {metric}_std,
flip_rate,
vulnerability_score
```

**RegionMetrics** — region 단위 (region_kind + params로 식별)
```
region_key, region_kind, intended_area_px, effective_area_px,
n_samples, n_valid,
{metric}_mean, flip_rate
```

**ClassMetrics** — gt_label 단위
```
gt_label, n_samples, {metric}_mean, flip_rate
```

**SpatialProfile** — sample × region 단위. 시각화의 직접 입력.
```
sample_id, region_key, region_geometry_ref,
{metric}_degradation
```

#### vulnerability_score 정의

v1에서는 **단순하고 설명 가능한 정의**를 채택한다.

```
vulnerability_score(sample) = 해당 샘플의 유효 아이템에 대한
                              선택된 primary metric의 degradation 평균
```

- `primary_metric`은 설정으로 지정 (기본: `margin_drop`)
- 정규화·가중치·복합 스코어는 v1에서 도입하지 않는다

**근거.** 복합 스코어는 해석이 어렵고, 리뷰어가 "왜 이 가중치인가"를 물었을 때 답하기 곤란하다. 단순 평균은 방어 가능하며, 사용자가 필요하면 `ItemMetrics`에서 직접 재계산할 수 있다.

#### 집계 시 필수 규칙

- **clean 정오로 층화한다.** `clean_correct=True/False` 그룹을 분리 집계
- **면적을 함께 보고한다.** region 집계에는 항상 `intended_area_px`와 `effective_area_px`를 병기. 면적 통제 없는 영역 비교는 오해를 부른다
- **유효 개수를 병기한다.** `n_valid`가 없으면 분모 변화가 감춰진다
- **`is_control` 아이템은 기본 집계에서 분리한다.** 대조군 분석은 후속 모듈의 몫이지만, 지금부터 섞이지 않게 한다

---

### N4. MetricsStore

계산 결과를 저장한다. raw dump와 동일한 원칙(재계산 가능성, provenance)을 따른다.

**출력 구조.**
```
<run_dir>/metrics/
├── item_metrics.parquet
├── sample_metrics.parquet
├── region_metrics.parquet
├── class_metrics.parquet
├── spatial_profile.parquet
└── metrics_manifest.json
```

**metrics_manifest.json.**
```
metrics_schema_version
source_run_manifest_hash        # 어떤 dump에서 계산했는지
metric_config                   # primary_metric, topk 등
registered_metrics              # 계산에 사용된 지표 목록과 버전
computed_at
exclusion_summary               # status별 제외 개수
```

**설계 의도.** dump가 바뀌면 지표도 무효가 되어야 한다. `source_run_manifest_hash`로 이를 감지한다. 지표 정의가 바뀌면 `metrics_schema_version`이 올라간다.

---

### N5. DebugViz (최소 시각화)

**목적을 명확히 한다.** 이것은 사용자를 위한 리포트가 아니라 **개발자가 오류를 발견하기 위한 도구**이다. 미관보다 진단력이 우선이다.

#### 왜 필요한가

숫자 표만으로는 다음 오류를 발견할 수 없다.

- 마스크 좌표계가 뒤집힘 (상하/좌우)
- region 인덱스와 실제 위치의 매핑 오류
- `invert_mask`가 반대로 적용됨
- 전처리 crop이 마스크를 예상과 다르게 잘라냄
- 교란이 마스크 밖에도 적용됨

이들은 전부 **눈으로 보면 즉시 보이고, 표로는 거의 보이지 않는다.**

#### 제공 뷰

**V1. 마스크 검증 뷰**
- 원본 이미지 + 마스크 오버레이 + 실제 교란된 이미지를 나란히 표시
- 소수 샘플(예: 5개)에 대해 각 region 하나씩
- 확인 대상: 마스크 위치가 의도와 일치하는가, 교란이 마스크 안에만 적용되었는가

**V2. 공간 민감도 히트맵**
- `SpatialProfile`을 이미지 위에 히트맵으로 오버레이
- region_key → 기하 위치 복원이 필요 (아래 참조)
- 확인 대상: 민감 영역이 객체 위에 있는가, 배경에 몰려 있는가

**V3. 샘플 랭킹 대조 뷰**
- `vulnerability_score` 상위 N개와 하위 N개를 히트맵과 함께 나열
- 확인 대상: 상위와 하위가 눈으로 봐도 다른가. 구분이 안 되면 지표나 코어에 문제가 있을 가능성

#### region 기하 복원

시각화는 region_key로부터 **실제 픽셀 위치**를 알아야 한다. 두 경로가 있다.

- **절차적 region** (grid 등): `region_params_json`으로부터 RegionResolver를 재호출하여 마스크 재생성. 코어의 결정론 덕분에 동일 마스크가 나온다
- **explicit region**: `ref_hash`로 원본 마스크 파일 참조

**설계 의도.** 마스크 비트맵을 dump에 저장하지 않기로 한 결정(저장 효율)이 여기서 비용을 발생시키지만, 재생성이 결정론적이므로 문제가 없다. 다만 시각화 모듈이 코어의 RegionResolver에 의존하게 되므로, 이 의존을 명시적으로 인정한다.

#### 출력 형태

v1에서는 **PNG 파일 저장**으로 충분하다. 인터랙티브 뷰나 HTML 리포트는 후속.

```
<run_dir>/debug_viz/
├── mask_check/sample_*.png
├── heatmap/sample_*.png
└── ranking/top_*.png, bottom_*.png
```

---

## 3. 검증 실험 계획

### 3.1 검증의 층위

| 층위 | 대상 | 방법 |
|---|---|---|
| L1. 단위 정확성 | 각 지표 계산식 | 손계산 가능한 고정 입력 |
| L2. 파이프라인 무결성 | dump → 지표 → 집계 | 합성 dump 주입 |
| L3. End-to-end 타당성 | 코어+지표 전체 | **합성 shortcut 실험** |

L1·L2는 pytest로, L3는 별도 실험 스크립트로 수행한다.

---

### 3.2 L1 — 지표 단위 테스트

**방법.** 손으로 계산 가능한 작은 로짓 벡터를 고정 입력으로 두고 기대값과 비교.

**검증 항목.**
- 각 지표의 계산식 정확성
- 부호 정규화 방향 (degradation이 양수일 때 성능 저하인가)
- `output_kind=probs`일 때 logit 기반 지표가 `available=false`
- clean=오답인 샘플에서 `flip_correct_to_wrong`이 False
- 결측(status≠ok) 시 null 반환 및 `excluded_reason` 기록

---

### 3.3 L2 — 합성 dump 주입 테스트

**방법.** 코어를 실행하지 않고 **의도적으로 구성한 dump를 직접 만들어** 지표 엔진에 넣는다.

**시나리오 예시.**

| 시나리오 | 구성 | 기대 결과 |
|---|---|---|
| 완전 무감각 | 모든 perturbed 로짓 = clean 로짓 | 모든 degradation = 0, flip_rate = 0 |
| 단일 영역 취약 | region_3만 로짓 급변, 나머지 동일 | region_3만 높은 degradation |
| 특정 샘플 취약 | sample_A의 전 region 급변 | sample_A만 높은 vulnerability_score |
| 결측 혼재 | 일부 status=predict_failed | 해당 아이템 제외, n_valid 감소, 집계에 반영 |
| clean 실패 | 일부 clean status≠ok | 해당 샘플 전체 제외 및 보고 |

**설계 의도.** 코어와 지표 엔진의 오류를 분리한다. L2가 통과하면 이후 문제는 코어 쪽이다.

---

### 3.4 L3 — 합성 Shortcut 실험 (핵심)

**목적.** 정답을 아는 조건에서 도구 전체가 올바르게 작동하는지 판정한다. 기획서 §9.1에 해당하며, 논문의 가장 강한 증거가 된다.

#### 실험 설계

**데이터 구성.**
1. 소규모 공개 분류 데이터셋 선택 (클래스 수가 적고 학습이 빠른 것)
2. 각 이미지의 **고정된 위치**에 클래스와 강하게 연관된 합성 패치 삽입
   - 예: 클래스별로 다른 색·패턴의 작은 사각형을 좌상단에 배치
   - 삽입 위치는 격자 region 하나에 정확히 대응하도록 정렬
3. 세 종류의 데이터셋 준비
   - **A (오염)**: 패치가 클래스와 완전 상관
   - **B (무관)**: 패치가 클래스와 무상관 (무작위 배치)
   - **C (무패치)**: 원본

**모델 구성.**
- **M_shortcut**: A로 학습. 패치에 의존할 것으로 기대
- **M_normal**: C로 학습. 패치를 보지 않을 것으로 기대

**평가.** A 데이터셋(또는 A와 동일 분포)에 대해 두 모델 각각에 도구를 실행.

#### 검증 질문과 판정 기준

> **아래 임계는 실험 전에 확정하며, 결과를 본 뒤 변경하지 않는다.**

| 질문 | 판정 기준 (사전 등록) |
|---|---|
| Q1. 패치 영역을 식별하는가 | M_shortcut에서 패치 region의 평균 degradation이 **전체 region 중 상위 1위** |
| Q2. 대조와 구분되는가 | 패치 region의 degradation이 비패치 region 평균의 **k배 이상** (k는 사전 지정) |
| Q3. 정상 모델과 구분되는가 | M_normal에서는 패치 region이 상위 1위가 **아님** |
| Q4. 교란 방식에 강건한가 | Q1이 **최소 2개 이상의 fill strategy**에서 재현 |
| Q5. 실제 성능과 연관되는가 | M_shortcut을 C(무패치)에서 평가 시 성능 하락이 M_normal보다 큼 |

**Q3와 Q5가 특히 중요하다.** Q1·Q2만으로는 "면적이 큰 영역이 항상 높게 나오는" 자명한 결과일 수 있다. Q3는 도구가 모델 차이를 구분함을, Q5는 도구의 진단이 실제 일반화 실패와 연결됨을 보인다.

#### 실패 시의 처리

기준 미달 시 **기준을 조정하지 않고 그대로 보고**한다. 다음 순서로 원인을 좁힌다.

1. L2가 통과했는가 → 지표 엔진 문제 배제
2. DebugViz의 마스크 검증 뷰 확인 → 좌표·마스크 문제 배제
3. M_shortcut이 실제로 패치에 의존하는가 → 패치 제거 시 정확도 하락으로 확인
4. 위가 전부 정상이면 → 도구의 민감도 한계로 기록하고 논문에 한계로 기술

#### 산출물

- 실험 스크립트 (`experiments/synthetic_shortcut/`)
- 재현 가능한 설정 파일과 시드
- 결과 표와 히트맵
- 판정 기준 대비 결과 요약

---

### 3.5 부수 검증 — 마스크 방식 민감도 (축소판)

L3와 함께 가볍게 수행한다.

- 동일 실험을 fill strategy별로 반복 (constant / mean / blur / noise / patch_shuffle)
- region 순위의 상관을 비교
- 특정 방식에서만 나타나는 결과가 있는지 확인

**목적.** 마스크 아티팩트가 결과를 지배하는지 조기에 파악한다. 정식 안정성 분석은 후속 모듈이지만, 여기서 큰 문제가 발견되면 설계를 재고해야 한다.

---

## 4. 구현 순서

| 단계 | 내용 | 성공 조건 |
|---|---|---|
| 1 | DumpReader (N0) | 실제 dump 로드, status 요약 출력, 스키마 버전 거부 동작 |
| 2 | OutputNormalizer (N1) | 파생값 손계산 일치 |
| 3 | Metric 인터페이스 + 1순위 지표 | L1 단위 테스트 통과 |
| 4 | 2순위 지표 | L1 확장 통과, 부호 정규화 검증 |
| 5 | Aggregator (N3) | L2 합성 dump 시나리오 전부 통과 |
| 6 | MetricsStore (N4) | 저장·재로드 일치, manifest 무결성 |
| 7 | DebugViz V1 (마스크 검증) | 실제 dump에서 마스크 위치 육안 확인 |
| 8 | DebugViz V2·V3 (히트맵·랭킹) | 히트맵이 SpatialProfile과 일치 |
| 9 | **L3 합성 shortcut 실험** | 사전 등록 기준 대비 결과 확보 |

**7단계를 5·6단계 직후에 두는 이유.** 지표가 나오자마자 눈으로 확인해야 코어 오류를 조기에 발견한다. L3 실험 전에 마스크가 올바른지 확인되어 있어야 실패 원인을 좁힐 수 있다.

---

## 5. 잔여 결정 사항

| 항목 | 결정 시점 |
|---|---|
| `topk_exit`의 k 기본값 | 3단계 |
| Q2의 배수 k, Q5의 임계 | L3 실험 착수 전 (사전 등록) |
| 합성 패치의 크기·위치·개수 | L3 설계 시 |
| L3에 사용할 데이터셋·모델 | L3 설계 시 |
| 확률 입력에서 logit 복원 허용 여부 | 2단계 |
| SpatialProfile의 저장 형식(long vs wide) | 5단계 |

---

## 6. 설계 결정 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| dump 접근 | DumpReader 경유만 허용 | 스키마 변경 격리 |
| 스키마 버전 불일치 | 거부 | 잘못된 해석보다 명확한 실패 |
| 지표 저장 형식 | long-form | 지표 추가 시 스키마 불변 |
| 부호 규약 | degradation 축 정규화 + 원시값 병기 | 리포트 일관성, 사후 재해석 가능 |
| 지표 확장 | Metric 플러그인 registry | 3순위 이후 확장 대비 |
| vulnerability_score | primary metric의 단순 평균 | 설명 가능성 우선, 복합 스코어 회피 |
| clean 정오 | 항상 층화 집계 | flip rate 해석 가능성 |
| 대조군 아이템 | 기본 집계에서 분리 | 후속 분석과의 경계 유지 |
| region 기하 복원 | RegionResolver 재호출 | dump 크기 절약, 결정론이 보장 |
| 시각화 목적 | 디버깅 우선 | 조기 오류 발견이 미관보다 중요 |
| L3 판정 기준 | 사전 등록, 사후 변경 금지 | 확증 편향 차단 |
