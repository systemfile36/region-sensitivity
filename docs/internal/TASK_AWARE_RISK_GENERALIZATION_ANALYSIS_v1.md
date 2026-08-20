# Task-Aware Occlusion Risk 연구의 일반화 관점에서 본 SSAT 보강 분석 (v1)

> 이 문서는 코드 변경 없이 현재 코드베이스(`ssat/core`, `ssat/metrics`, `ssat/analysis`,
> `ssat/report`)를 검토하고, "Task-aware quality diagnosis for action recognition"
> 연구(이하 "원 연구")의 핵심 발견을 이 프로젝트가 어디까지 일반화했고, 어디서부터
> 다시 채워 넣어야 하는지를 분석한 결과다. 사용자가 지적한 대로 grid region은 현재
> 리포트(§4 참고)로 충분히 다뤄지지만, skeleton 기반 의미적(semantic) region — 특히
> 라벨(행동 클래스)별 취약 부위 집계 — 은 core 계층 구현이 report/analysis 계층보다
> 앞서 나가 있어 새로 채워야 할 격차가 명확히 존재한다.

## 0. 결론 요약

- 원 연구의 핵심 주장은 "**샘플마다 취약한 부위가 다르고, occlusion 종류(어떤 부위가
  가려졌는지)만으로는 위험도를 예측할 수 없다**"(corruption-type-alone AUC 0.652는
  정보 상한선일 뿐이고, video-specific vulnerability를 결합해야 0.774까지 오른다)는
  것이다. SSAT는 이미 이 주장의 "진단(diagnosis)" 절반 — 어떤 샘플의 어떤 region이
  실제로 위험한지 exhaustive하게 감사하는 것 — 을 잘 수행하도록 설계되어 있다.
- **grid region은 이번 세션에 재구성한 리포트(A+C 조합)로 원 연구의 "dataset-wide
  spatial pattern" 질문에 이미 잘 답한다.** grid의 `region_key`가 샘플 전체에서
  동일한 물리적 위치를 가리키기 때문에, dominant-region-share/spatial-entropy 같은
  dataset 레벨 집계가 그대로 성립한다.
- **skeleton 기반 semantic region(`skeleton_parts`)은 core 계층(`ssat/core/region/
  skeleton_*.py`, `ssat/core/config/schema.py`)에 이미 구현되어 있다** — 이는
  `docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md`가 "확장 가능"이라고 분석했던
  것보다 더 진행된 상태다. 하지만 **`ssat/metrics`/`ssat/analysis`/`ssat/report`
  세 계층은 여전히 "region_key가 데이터셋 전체에서 재사용되는 고정 위치"라는
  grid 전용 암묵적 전제 위에 서 있다.** skeleton_parts의 `region_instance_id`는
  샘플마다 다르므로(`region_id/sample_id` 형식), 이 전제가 깨지고, 그 결과:
  - `ssat/report/types.py`의 `RegionRow`에는 sample-invariant한 semantic
    식별자(`region_id`)가 아예 없다 — `region_key`만 있다.
  - 이번 세션에 추가한 `SpatialConcentration`(dominant-region-share/spatial-entropy)은
    skeleton_parts에서 수학적으로 퇴화한다(§2.3).
  - `ssat/analysis`의 region-level 신뢰도(A5 `IntervalRow`)도 같은 이유로
    skeleton_parts에서 통계적 검정력을 잃는다(§2.4).
  - **라벨(행동 클래스, `gt_label`)별 취약 부위 집계는 어느 계층에도 존재하지
    않는다** — `ClassMetrics`는 `(gt_label, metric_name)` 축만, `RegionMetrics`/
    `SpatialProfile`은 `(region_key, metric_name)` 축만 갖고 둘이 교차되지 않는다
    (`ssat/metrics/aggregate.py`).
- 따라서 사용자가 요청한 "최소한 각 라벨별로 취약한 부위를 집계"하는 기능은
  **완전히 새로운 교차축(class × semantic region)이며, 기존 어떤 계층에도 부분적으로도
  구현되어 있지 않다.** 이를 추가하려면 (a) semantic `region_id` 축을 report 계층까지
  끌어올리고, (b) `gt_label × region_id` 교차 집계를 새로 계산하는 두 가지가 필요하다.
  구체적 위치와 트레이드오프는 §3에서 다룬다.

---

## 1. 원 연구의 핵심 주장과 이 프로젝트의 목적

원 연구(Abstract 요약)의 구조는 다음 세 요소로 나뉜다.

1. **위험(risk)의 정의**: occlusion의 존재 자체가 아니라, teacher recognizer의
   hard correctness를 실제로 뒤집는지(binary flip) 여부.
2. **정보 상한선(information ceiling) 실험**: "어떤 부위가 가려졌는가"라는
   corruption-type 정보만으로는 AUC 0.652가 한계 — 이는 **모델의 한계가 아니라
   문제 자체의 정보 상한선**임을 별도로 증명했다.
3. **일반화 모델**: 비디오만 보고 (a) 어떤 부위가 가려졌는지, (b) 그 비디오에서
   어떤 부위들이 취약한지를 동시에 예측하고, 파라미터 없는 결정 규칙으로 결합해
   AUC 0.774를 달성 — 즉 **"어떤 부위가 가려졌는가"와 "그 비디오가 어떤 부위에
   취약한가"는 서로 다른 정보이고, 후자가 핵심 기여**라는 것이 논문의 증거다.

사용자가 이 연구에서 일반화하고자 하는 것은 3번이 아니라 **그 전 단계** — "샘플마다
취약한 부위가 다르다"는 사실을 실제로 관측/감사하는 도구다. 즉 SSAT는 원 연구의
online predictor(비디오 하나만 보고 실시간으로 위험을 추정하는 모델)가 아니라,
**exhaustive하게 모든 (샘플, 후보 부위) 조합을 실제로 가려보고 측정하는 offline
audit 도구**다. 이 위치 설정은 §5에서 다시 정리한다.

---

## 2. 현재 상태: skeleton_parts가 core에는 있지만 상위 계층이 못 따라간다

### 2.1 `region_key`의 구조: grid vs skeleton_parts

`region_key = f"{region_id}::{region_instance_id}"`
(`ssat/metrics/aggregate.py:162`)이며, `region_instance_id`는 region kind마다
다르게 만들어진다.

| Region kind | `region_instance_id` 생성 규칙 | 샘플 간 재사용 여부 |
|---|---|---|
| `grid` (`ssat/core/plan/region_expanders.py` `GridRegionExpander.expand`) | `f"{region_id}/r{row}/c{col}"` — row/col만으로 결정 | **재사용됨**: 모든 샘플에서 `grid/r0/c0`은 같은 물리적 위치를 가리킴 |
| `explicit` (`ExplicitRegionExpander.expand`) | `region_id` 그대로 | 재사용됨(설정 자체가 데이터셋 전역 마스크 1장) |
| `skeleton_parts` (`ssat/core/region/skeleton_provider.py` `SkeletonRegionProvider.expand`) | `f"{region_id}/{sample.sample_id}"` — **샘플 ID가 포함됨** | **재사용되지 않음**: `occlude_left_arm/video_0001`과 `occlude_left_arm/video_0002`는 서로 다른 `region_key` |

즉 skeleton_parts는 "왼팔"이라는 의미적 부위 하나를, 사람이 프레임마다 이동하므로
샘플마다 다른 bbox로 추적해야 하기 때문에 의도적으로 `region_instance_id`에
`sample_id`를 넣었다(§3.2 참고 — `docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md`
§3.2 항목 3이 이미 예견한 설계). **이 설계 자체는 올바르다** — 문제는 상위 계층이
`region_key`를 "고정된 하나의 위치"로 취급하는 지점들이다.

### 2.2 report 계층에 semantic `region_id`가 없다

`RegionRow`(`ssat/report/types.py:242` 부근)는 `region_key: str`,
`region_kind: str` 필드만 갖고 있고, `region_id`(가족 수준 식별자, 예:
`"occlude_left_arm"`)를 별도로 보관하지 않는다. `ssat/report/assembler.py`의
`_build_region_summary`도 `SpatialProfile.region_key`만 그룹핑 키로 쓴다
(`ssat/report/assembler.py:582` 부근, `sorted(region_rows, key=lambda row:
row.region_key)`).

결과적으로 skeleton_parts로 감사를 실행하면, region 표(및 이번 세션에 추가한
composition bar/히트그리드)는 "occlude_left_arm/video_0001", "occlude_left_arm/
video_0002", ... 를 **서로 무관한 수백~수천 개의 개별 region**으로 나열하게 된다.
"왼팔을 가렸을 때 전반적으로 얼마나 위험한가"라는, 사용자가 실제로 알고 싶어할
질문에는 현재 리포트가 답을 줄 수 없다 — 집계할 공유 키가 애초에 존재하지 않기
때문이다.

### 2.3 `SpatialConcentration`(이번 세션 추가분)이 skeleton_parts에서 퇴화하는 이유

이번 세션에 `ssat/report/assembler.py`에 추가한
`_build_spatial_concentration`은 "샘플별 top region"을 `Counter`로 세어
`dominant_region_share`/`spatial_entropy`를 계산한다. grid에서는 이 값이
의미 있다 — 여러 샘플이 실제로 같은 `region_key`(예: `grid/r0/c0`)를 공유할 수
있기 때문이다. 그러나 skeleton_parts에서는 **모든 `region_key`가 정확히 1개
샘플에만 존재**하므로:

- `Counter(top_region_by_sample.values()).most_common(1)`의 최댓값은 항상 1이 되고,
  `dominant_region_share = 1 / n_scored_samples` — 샘플 수가 늘어날수록 자동으로
  0에 수렴하는, **실제 집중도와 무관한 인공적인 값**이 된다.
- 정규화 엔트로피 `H = -Σp log(p) / log(len(region_keys))`도 `len(region_keys) ==
  n_scored_samples`에 근접하고 모든 `p = 1/n_scored_samples`인 균등분포이므로
  **항상 1에 가깝게 나온다** — "얼마나 분산되어 있는가"가 아니라 "region_key가
  얼마나 잘게 쪼개져 있는가"만 측정하는 지표로 변질된다.

즉 이 두 지표는 **"region_key가 데이터셋 전체에서 몇 번이나 재사용되는가"에
암묵적으로 의존**하고 있고, 이는 grid에서만 성립하는 전제다. skeleton_parts에
그대로 적용하면 계산은 되지만(에러는 안 남) 숫자가 아무 의미도 갖지 않는다 —
"unavailable ≠ 0"이라는 이 프로젝트의 기존 원칙(§6.2 C1)에 비춰 보면, 이 경우는
"계산 가능하지만 무의미"라는 세 번째 범주여서 더 위험하다(사용자가 숫자를 보고
잘못된 결론을 내릴 수 있음).

### 2.4 region-level 신뢰도/CI도 같은 문제를 공유한다

`ssat/analysis/reliability.py`의 A5 `IntervalRow`는 `(region_key, metric)` 단위로
계산된다(같은 파일 24-27행 주석: "``IntervalRow``(A5)는 ``(region_key, metric)``으로만
키가 잡혀 있어 ``AnchorKey``보다 성긴 grain — 같은 ``region_key``를 공유하는
모든 것이 그 region의 CI 판정을 공유한다"). 이 설계는 "여러 샘플이 같은
`region_key`를 공유해 신뢰구간을 풀링(pool)할 수 있다"는 grid 전제 위에서만
통계적으로 의미가 있다. skeleton_parts에서는 `region_key`당 샘플이 정확히 1개뿐이므로,
seed 반복(같은 anchor를 여러 seed로 반복 실행하는 것)으로 얻는 CI는 여전히
유효하지만, **여러 샘플을 풀링해서 얻는 region 수준 CI/threshold 보정은 사실상
표본 크기 1로 축소**된다 — `docs/RELIABILITY_THRESHOLD_CALIBRATION_v1.md`가
전제한 "많은 anchor를 풀링해 임계값을 보정한다"는 절차 자체가 skeleton_parts에는
그대로 적용되지 않는다.

---

## 3. 원 연구의 premise를 일반화하기 위해 추가로 필요한 것

### 3.1 semantic `region_id` 축을 report 계층까지 끌어올리기

`region_id`(가족 수준 식별자, config의 `regions[].region_id`)는 이미
`ssat/core/plan/region_expanders.py`/`ssat/metrics/aggregate.py:162`에 존재하고
`region_key`의 구성 요소이지만, `region_key`로 합쳐진 뒤 report 계층(`RegionRow`,
`SpatialConcentration`)까지는 전달되지 않는다. skeleton_parts처럼 "같은 semantic
region_id, 샘플마다 다른 concrete instance"인 kind를 다루려면, **`region_key`
(anchor 단위 신뢰도 표시에는 계속 필요) 와 별개로 `region_id`(dataset 레벨 집계
축)를 report 데이터 모델에 명시적으로 추가**해야 한다. grid에서는 이 둘이 사실상
같은 역할(각 셀이 곧 하나의 region_id 인스턴스)이라 회귀가 없고, skeleton_parts에서만
비로소 의미가 갈린다.

이 축이 report 계층에 들어오면:
- `_build_spatial_concentration`을 `region_key` 대신 `region_id`로 그룹핑하는
  경로를 (kind에 따라) 선택할 수 있다 — grid는 지금처럼 셀 단위, skeleton_parts는
  부위 단위로 자연스럽게 집계된다.
- region 표/composition bar도 "왼팔(전체 샘플 풀링)"처럼 의미 있는 행을 가질 수
  있다.

### 3.2 라벨(행동 클래스)별 취약 부위 집계 — 새로운 교차축

사용자가 명시적으로 요청한 항목이다. 현재 존재하는 두 축을 다시 보면:

- `ClassMetrics`(`ssat/metrics/types.py:216`): `(gt_label, metric_name)` — region
  정보 없음.
- `RegionMetrics`/`SpatialProfile`(`ssat/metrics/aggregate.py`): `(region_key,
  metric_name)` / `(sample_id, region_key, metric_name)` — `gt_label`은
  `SpatialProfile`이 `sample_id`를 갖고 있어 join으로 복원은 가능하지만, **사전에
  집계된 `(gt_label, region_id)` 교차표는 어디에도 없다.**

즉 "행동 클래스 A(예: '앉기')에서는 왼팔보다 다리를 가리는 쪽이 훨씬 위험하다"
같은, 원 연구의 핵심 질문에 정확히 대응하는 표는 **파이프라인 어디에도 계산되어
있지 않다.** 새로 만들어야 한다. 필요한 최소 형태는 `(gt_label, region_id,
metric_name) -> {n_samples, mean_degradation, flip_rate}` 이며, 선택적으로
`dominant_region_share`/`spatial_entropy`를 클래스별로도 분리해 "클래스마다
집중되는 부위가 다른가"까지 답할 수 있다.

### 3.3 이 새 집계를 어느 계층에 둘 것인가 — 설계 경계상의 열린 질문

이 프로젝트는 지금까지 "R0(`ssat/report/assembler.py`)는 조립만 하고 새 통계를
만들지 않는다"는 경계(Gap#6, 이번 세션 §의 근거이기도 했음)를 지켜왔다. `(gt_label
× region_id)` 교차 집계는 이 경계에 비춰보면 R0의 영역이 아니라 **N3
(`ssat/metrics/aggregate.py`)에 `ClassMetrics`/`RegionMetrics`와 나란히 새 출력
타입(예: `ClassRegionMetrics`)을 추가**하는 쪽이 기존 설계 원칙과 더 잘 맞는다 —
`RegionMetrics`가 이미 "item-grain을 sample-grain으로 축약 후 평균"하는 정확히
같은 패턴을 쓰고 있어(`ssat/metrics/aggregate.py` 모듈독스트링), 세 번째 교차축을
같은 방식으로 추가하는 것은 새 알고리즘이 아니라 기존 패턴의 반복이다. 다만 이는
스키마 동결(schema freeze, 같은 독스트링이 "frozen (stage 0) schemas"라고 명시)
정책과 부딪힐 수 있어, **N3에 새 타입을 추가하는 결정 자체는 이 문서가 내리지 않고
사용자/설계자의 판단으로 남긴다.** 대안으로, `SpatialProfile`(이미 `sample_id` +
`region_key` + `degradation`을 갖고 있음)을 `ssat/analysis` 또는 `ssat/report`
단계에서 `gt_label`과 join해 계산하는 방법도 있으나, 이 경우 "새 통계를 어디서
계산했는가"가 흩어져 Gap#6이 지켜온 경계가 흐려질 수 있다는 점을 함께 고려해야
한다.

### 3.4 리포트 레이어의 표현 — label × part 표/히트맵

3.1~3.3이 데이터 모델/집계 문제라면, 이를 사용자에게 보여주는 문제도 별도로
남는다. 이번 세션에 재구성한 A+C 조합 메인 리포트(§4 히트그리드 등)는 명시적으로
grid 형태 `region_key`만 파싱하도록 만들어졌다(`_grid_layout`의 `/r(\d+)/c(\d+)$`
정규식, `ssat/report/html_renderer.py`) — grid가 아니면 그레이스풀하게 표로만
폴백한다. `gt_label × region_id` 교차표가 §3.2대로 새로 생기면, 이를 위한
**별도의 신규 섹션**(예: "클래스별 취약 부위" 표 또는 히트맵 — 행: 클래스,
열: semantic region_id)이 필요하다. 이는 이번 세션에 완료한 A+C+B 레이아웃
재구성과는 독립적인, 후속 확장 지점으로 분리해 다루는 것이 적절하다(이번 재구성이
이미 8단 구조 + 보조 리포트로 상당히 커졌기 때문).

### 3.5 (부가) 이진 위험 라벨과의 정합성

원 연구는 위험을 "teacher의 hard correctness를 뒤집는지"라는 **이진(binary)**
정의로 잡는다. SSAT는 이미 `flip_rate`(`ssat/metrics/types.py`의 `ItemMetrics.
flip_rate` 등, item→sample→region→class까지 롤업됨)라는, **개념적으로 정확히
같은 것**을 갖고 있다 — clean 예측이 맞았던 샘플에서 perturbed 예측이 틀리는지를
이진으로 집계한 값이다. 다만 현재 report 계층은 `vulnerability_score`(연속값,
기본 `margin_drop`)를 주로 전면에 내세우고 `flip_rate`는 스코어카드의 보조 카드
정도로만 노출된다. 만약 이 프로젝트의 산출물을 원 연구류의 predictor를 학습시키는
**라벨 소스**로 쓰고자 한다면(§5), `flip_rate`(이진, teacher 관점의 hard failure)를
`vulnerability_score`(연속, degradation 크기)와 **동등하게 1급 시민으로 노출**하고,
`(sample_id, region_id) -> binary_risk_label` 형태로 export하는 경로(CSV/JSON)를
추가하는 것이 원 연구의 라벨 정의와 가장 잘 맞물린다. 이는 새 계산이 아니라 이미
있는 `flip_rate`의 노출 범위를 넓히는 문제이므로, §3.1~3.3보다 훨씬 작은 작업이다.

---

## 4. 원 연구 개념 ↔ SSAT 개념 대응표

| 원 연구 개념 | SSAT의 현재 대응물 | 상태 |
|---|---|---|
| Teacher의 hard correctness flip (이진 위험 라벨) | `ItemMetrics.flip_rate` → `SampleMetrics`/`RegionMetrics`/`ClassMetrics.flip_rate` | 이미 존재, 계층 전체에 롤업됨 (§3.5) |
| "어떤 부위가 가려졌는가" (corruption type) | 감사자가 config로 지정하는 `regions[].region_id`/`kind` — SSAT는 이 정보를 **예측할 필요가 없다**(실행 시점에 이미 알고 있음) | 개념적으로 다른 위치: SSAT는 진단 도구이지 실시간 predictor가 아님(§5) |
| Corruption-type-alone AUC ceiling (0.652) | grid 한정: dataset 레벨 `dominant_region_share`/`spatial_entropy`(이번 세션 추가) | grid에서만 유효, skeleton_parts에서 퇴화(§2.3) |
| "그 비디오에서 어떤 부위가 취약한가" (video-specific vulnerability) | `SampleCard.top_regions`(`ssat/report/assembler.py`의 `_top_regions_for_sample`) — 샘플별로 가장 취약한 region들을 이미 순위화해 갖고 있음 | **이미 잘 대응됨** — 원 연구의 핵심 기여와 개념적으로 가장 가까운 SSAT 자산 |
| 부위별 취약도의 클래스(라벨)별 요약 | 없음 | **완전히 부재 — 이 문서의 핵심 갭(§3.2)** |
| Occlusion을 트리거로 한 selective restoration | 없음(SSAT는 감사/진단 도구, 복원 트리거는 범위 밖) | 의도적으로 범위 밖 — SSAT의 산출물(§5)이 그런 predictor의 입력 자료가 될 수 있을 뿐 |

---

## 5. SSAT를 원 연구 파이프라인에서 어떻게 위치시킬지

원 연구의 predictor는 **비디오 하나만 보고 실시간으로** "어떤 부위가 가려졌는지"와
"어떤 부위가 취약한지"를 추론한다. SSAT는 반대로 **모든 후보 (샘플, 부위) 조합을
실제로 가려보고 결과를 측정하는 exhaustive offline 도구**다. 이 둘은 경쟁 관계가
아니라 파이프라인의 서로 다른 단계다:

1. **SSAT(진단)** — 데이터셋 전체에 대해 "실제로 어떤 부위를 가렸을 때 어떤
   샘플/클래스가 위험해지는가"를 exhaustive하게 측정해 `(sample_id, region_id,
   gt_label) -> {vulnerability_score, flip_rate, reliability_grade}` 형태의
   신뢰할 수 있는 ground-truth급 자료를 만든다.
2. **원 연구류 predictor(학습)** — 1번의 산출물을 지도학습 라벨로 사용해, 비디오만
   보고 실시간으로 같은 정보를 추론하는 모델을 학습한다.

이 관점에서 보면 §3의 갭들은 단순한 "리포트 개선"이 아니라 **SSAT가 1단계
파이프라인의 라벨 소스 역할을 제대로 하기 위한 최소 요건**이다 — 특히 §3.2(클래스별
취약 부위 집계)는, 클래스마다 다른 취약 부위 패턴을 predictor가 학습하려면 그
패턴이 먼저 SSAT의 산출물 안에 명시적으로 드러나 있어야 한다는 점에서 더 이상
"있으면 좋은 기능"이 아니라 **원 연구를 일반화한다는 목적 자체에 필수적인 항목**이다.

---

## 6. 제안 우선순위 (구현 순서 제안, 코드는 이 문서에서 작성하지 않음)

1. **`region_id`를 report 데이터 모델(`RegionRow`, `SpatialConcentration`)까지
   끌어올린다(§3.1).** 가장 작고, 다른 모든 후속 작업의 전제 조건이다. grid에는
   회귀가 없다(§2.1 표 참고 — grid는 `region_id`와 `region_key`가 사실상 동형).
2. **`flip_rate`를 report 1급 시민으로 노출 + `(sample_id, region_id) -> binary
   risk label` export 경로 추가(§3.5).** 새 계산이 필요 없어 2번째로 작다.
3. **`(gt_label, region_id)` 교차 집계를 N3 또는 그 대안 위치에 추가한다(§3.2,
   §3.3).** 어느 계층에 둘지부터 먼저 결정이 필요한, 이 문서에서 가장 설계 논의가
   필요한 항목.
4. **region-level CI/threshold 보정(§2.4)을 skeleton_parts처럼 `region_key`당
   샘플이 1개뿐인 경우에 어떻게 재정의할지 별도 검토.** 통계적으로 가장 까다로운
   항목이라 마지막으로 미룰 것을 권장 — 3번이 끝나 실제 skeleton_parts 감사
   데이터가 쌓인 뒤에 실증적으로 접근하는 편이 안전하다.
5. **리포트에 label × part 표/히트맵 섹션을 추가한다(§3.4).** 1~3이 끝나야 렌더링할
   데이터가 존재하므로 마지막.

---

## 7. 미해결 질문 / 리스크

1. **§3.3의 계층 배치 결정이 이 문서의 가장 큰 열린 질문이다.** N3에 새 타입을
   추가하는 것은 "frozen schema" 정책과 충돌 가능성이 있고, 반대로 R0/분석 계층에서
   ad-hoc join으로 계산하면 "R0는 조립만 한다"는 기존 경계가 흐려진다. 코드
   작성 전에 먼저 결정해야 한다.
2. **skeleton_parts 감사를 실제로 대규모 실행한 실데이터가 아직 없다**(이 문서는
   코드 검토로만 작성됨). §2.3/§2.4의 퇴화 현상은 수식적으로는 확실하지만, 실제
   NTU-RGB+D 규모 실행에서 `flip_rate`/CI가 실무적으로 얼마나 불안정해지는지는
   실측이 필요하다.
3. **region_id를 집계 축으로 쓸 때, 서로 다른 body_part 이름 규칙(관절 인덱스
   기반 정의, `docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md` §4)이 데이터셋마다
   다를 수 있다** — "클래스별 취약 부위" 표가 여러 skeleton_source/body_part 정의를
   섞어 쓰는 실행 간에는 비교 가능하지 않다는 점을 리포트에 명시할 필요가 있다.
