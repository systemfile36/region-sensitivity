# 리포팅 계층 설계 명세 (v1)
## Spatial Sensitivity Audit Toolkit — Reporting Layer

---

## 0. 문서 범위

본 문서는 지표 엔진과 대조군·안정성 분석 모듈의 산출물을 사람이 소비할 수 있는 형태로 변환하는 **최종 단계**를 기술한다. 파이프라인에서 사용자가 실제로 "결과를 보는" 유일한 지점이며, 이게 없으면 도구가 아니라 라이브러리에 머문다.

### 포함 범위

JSON/CSV 요약 export, 정적 HTML 대시보드, 서버사이드 차트·히트맵 자산 생성, 분류 태스크 렌더링.

### 제외 범위 (v1.1 이후)

인터랙티브 대시보드(서버 기반 필터링·검색), 클러스터링 뷰, 동적 WebUI. 다만 이 문서의 설계는 **그 확장을 전제로** 한다 — 방향은 §4.6에서 다룬다.

### v1 전제

- 입력은 `MetricsStore`(지표 엔진)와 `AnalysisStore`(대조군·안정성 분석)의 parquet + manifest
- 분류 태스크만 실제 렌더링. 검출은 스키마 수준에서 자리만 확보(§5)
- **완전 오프라인 산출물.** 외부 CDN·네트워크 호출 없음. 생성된 폴더만으로 어디서든 열람 가능해야 함
- 프레임워크 비의존. 무거운 JS 프레임워크나 빌드 파이프라인을 두지 않음

### 핵심 원칙

**리포트는 계산하지 않고 조립한다.** 모든 수치는 이미 MetricsStore·AnalysisStore에 있다. 리포팅 계층은 이를 선택·정렬·시각화할 뿐, 새로운 통계를 도출하지 않는다. 새 계산이 필요해 보이면 그것은 지표 엔진이나 분석 모듈의 몫이다.

**JSON 모델이 먼저이고 HTML은 그 뷰 중 하나다.** 향후 WebUI가 동일한 JSON을 소비하게 될 것이므로, 데이터 조립과 HTML 렌더링을 처음부터 분리한다.

---

## 1. 전체 흐름

```
[MetricsStore]              [AnalysisStore]           [DumpReader.manifest]
  sample/region/class          reliability grades         run 설정, 소요시간,
  metrics, spatial_profile     control/stability          실패율, 모델 식별자
        │                          │                          │
        └──────────────┬───────────┴──────────────────────────┘
                       ▼
┌─ R0. ReportDataAssembler ────────────────────────────────────┐
│   전 소스를 취합 → 정규화 → 스키마 버전 검증                  │
│   TaskPresentationAdapter(R5)에 태스크별 변환 위임             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
                 [ReportModel]   ← 단일 진실 공급원 (JSON 직렬화 가능)
                       │
        ┌──────────────┼──────────────┬───────────────────┐
        ▼              ▼              ▼                    ▼
┌─ R1. Exporter ┐ ┌─ R2. Chart ──┐ ┌─ R3. AssetLinker ┐   │
│  JSON / CSV   │ │  Renderer    │ │                  │   │
│  그대로 저장  │ │  (서버사이드 │ │  DebugViz 산출물 │   │
│               │ │   SVG 생성)  │ │  선별·재배치     │   │
└───────┬───────┘ └──────┬───────┘ └────────┬─────────┘   │
        │                │                   │              │
        └────────────────┴───────────────────┴──────────────┘
                                   ▼
                    ┌─ R4. HTMLRenderer ──────────────┐
                    │   템플릿 + ReportModel + 자산    │
                    │   → 정적 HTML 조립               │
                    └──────────────┬───────────────────┘
                                   ▼
                    <run_dir>/report/  (자기완결 폴더)
```

---

## 2. 모듈 명세

### R0. ReportDataAssembler

**책임.**
- `MetricsStore`, `AnalysisStore`, `run_manifest.json`을 읽어 하나의 `ReportModel`로 정규화
- 각 소스의 `schema_version` 호환성 확인 (기존 Reader들과 동일 원칙 — 불일치 시 거부)
- `TaskPresentationAdapter`(R5)를 호출해 태스크 종속 필드를 공통 스키마로 변환
- top-K 선별 (전체 샘플이 아니라 취약 상위/하위 K개만 리포트 모델에 포함 — 나머지는 CSV/parquet으로 충분)

**설계 의도.** 이 모듈이 유일하게 "무엇을 보여줄지" 결정한다. R1~R4는 전부 `ReportModel`을 받아 형식만 바꾸는 순수 변환기다. 계산 로직과 표현 로직을 여기서 분리한다.

**K 선정 기준.** `top_k`, `bottom_k` 설정(기본 20). 전체를 담으면 HTML이 무거워지고, 대부분의 사용자는 극단값에 관심이 있다. 전량은 항상 CSV/parquet으로 접근 가능하다.

---

### R1. Exporter (JSON/CSV)

**책임.** `ReportModel`을 그대로 JSON으로 직렬화하고, 표 형태의 하위 구조를 CSV로 평탄화한다.

**산출.**
```
report/data/
├── report_model.json       # ReportModel 전체
├── sample_rankings.csv     # 취약도 내림차순 전체 (top-K 아님, 전량)
├── region_summary.csv      # region_key, area, degradation, reliability_grade 등
└── flagged_items.csv       # reliability_grade=unreliable 항목만 별도 추출
```

**설계 의도.** `report_model.json`이 **향후 WebUI의 API 응답과 동일한 형태**가 되도록 설계한다. 지금은 파일로 저장하지만, 나중에 이 객체를 그대로 HTTP 응답으로 바꿔치기할 수 있어야 한다. `flagged_items.csv`를 별도로 두는 이유는, 신뢰도 낮은 결과를 사람이 검토할 때 CSV 하나만 열면 되게 하기 위함이다.

---

### R2. ChartRenderer

**책임.** 취약도 분포 히스토그램, region 순위 비교 등 집계 시각화를 **서버사이드에서 SVG로** 생성한다.

**왜 서버사이드인가.** 클라이언트 JS 차트 라이브러리(Chart.js 등)를 쓰면 CDN 의존이 생기거나 자산을 번들해야 한다. SVG를 미리 그려 넣으면 HTML은 그 결과물만 담고, 오프라인·의존성 원칙(§0)을 자연히 만족한다. 부수 효과로 **재현성 테스트가 쉬워진다** — 같은 입력이면 같은 SVG가 나와야 한다(난수 요소가 있는 시각화는 seed 고정).

**v1 차트 목록.**
- 취약도 점수 히스토그램 (전체 샘플 분포)
- region별 평균 저하도 막대 (신뢰도 등급별 색 구분)
- (가용 시) fill strategy 간 순위 상관 히트맵 — 분석 모듈 산출물 그대로 시각화

**설계 의도 — §3 요구사항 직결.** 표준 벤치마크 도구는 스칼라 하나(mCE 등)로 요약한다. 이 도구의 차별점은 분포와 이질성이므로, **평균 하나가 아니라 분포 전체를 첫 화면에서 보여주는 것**이 정체성이다. 히스토그램이 그 역할을 한다.

---

### R3. AssetLinker

**책임.** 지표 엔진 단계에서 만든 `DebugViz` 산출물(마스크 검증 뷰, 공간 히트맵, 샘플 랭킹) 중 리포트에 필요한 것만 선별해 `report/assets/`로 복사·재배치한다.

**설계 의도.** 히트맵을 다시 그리지 않는다. DebugViz가 이미 만든 것을 재사용한다. 다만 DebugViz는 디버깅 목적으로 전체 또는 임의 샘플을 대상으로 했고, 리포트는 **R0가 선정한 top-K 샘플에 한정**해서 링크한다.

**의존 방향에 대한 주의 (구현자 확인 필요).** 코어 설계 명세에서 DebugViz가 마스크 기하 복원을 위해 코어의 `RegionResolver`를 재호출하는 구조를 잠정 채택하며 의존 방향 이슈를 남겨둔 바 있다. 리포팅 계층에서 동일한 필요(신규 히트맵 생성 시)가 다시 발생하면, **코어에서 마스크 재생성 유틸리티를 분리해 `metrics`와 `report` 양쪽이 그것만 참조**하도록 정리하는 것을 권한다. 이번 모듈에서 이 부채를 굳이 새로 얹지 않도록, 원칙적으로 R3는 **이미 생성된 자산의 재배치만** 하고 신규 렌더링은 하지 않는다.

---

### R4. HTMLRenderer

**책임.** `ReportModel` + R2 차트 + R3 자산을 받아 정적 HTML로 조립한다.

**기술적 원칙.**
- 템플릿 엔진(예: Jinja2)으로 데이터와 마크업을 분리. 로직을 템플릿에 넣지 않는다
- **JS 없이도 모든 콘텐츠가 보여야 한다.** 표 정렬, 탭 전환 같은 것은 점진적 향상(progressive enhancement)으로 추가하되, 기본 정렬 순서(취약도 내림차순)는 서버에서 이미 확정해 렌더링한다
- CSS는 반응형(그리드/플렉스 기반), 인쇄용 미디어 쿼리 포함 — 논문 부록이나 발표 자료로 캡처하기 좋게
- 폰트·아이콘·CSS 전부 로컬 번들. 외부 요청 0건

**출력.**
```
<run_dir>/report/
├── report.html
├── data/                    (R1 산출물)
├── assets/
│   ├── css/style.css
│   ├── js/enhance.js        (선택적 상호작용, 없어도 콘텐츠 접근 가능)
│   ├── img/heatmaps/*.png   (R3)
│   └── img/charts/*.svg     (R2)
└── report_manifest.json
```

**report_manifest.json.**
```
report_schema_version
source_manifest_hashes: {run, metrics, analysis}
top_k, bottom_k
generated_at
```

---

### R5. TaskPresentationAdapter

**책임.** 태스크별 개념(분류의 정확도 vs 검출의 mAP/recall)을 **공통 표현 구조**로 변환한다. 확장의 핵심 지점.

**인터페이스(개념).**
```
summarize_performance(metrics) -> list[MetricCard]
sample_extra_fields(sample_metrics) -> dict   # 태스크 고유 부가 정보
applicable_charts() -> list[str]
```

**MetricCard — 태스크 무관 공통 단위.**
```
key, label, value, unit, higher_is_better, note
```

분류 어댑터는 `[accuracy, mean_margin_drop, flip_rate]` 같은 카드를 만들고, 검출 어댑터(§5)는 `[mAP, recall, mean_miss_rate]`를 만든다. `ReportModel`이나 R4는 이 차이를 전혀 모른다 — 항상 `MetricCard` 리스트만 다룬다.

---

## 3. ReportModel 스키마 (개념)

```
ReportModel:
  meta:
    run_id, generated_at, tool_version,
    schema_versions: {dump, metrics, analysis, report}
    task_kind: "classification" | "detection"

  run_summary:
    dataset_name, n_samples, n_regions_per_sample, n_conditions,
    duration_seconds, failure_rate,
    model_id, preprocessing_desc

  scorecard: list[MetricCard]        # R5가 생성

  vulnerability_distribution:
    histogram_asset_ref,
    summary_stats: {mean, median, p90, p99}

  sample_rankings:
    most_vulnerable: list[SampleCard]   # top-K
    most_robust:     list[SampleCard]   # bottom-K

  region_summary:
    rows: list[RegionRow]
    reliability_distribution: {high, moderate, low, unreliable}

  reliability_spotlight:
    flagged_examples: list[FlaggedItem]

  provenance:
    embedded manifest 요약 + 원본 파일 경로 링크
```

**SampleCard.**
```
sample_id, gt_label, clean_correct,
vulnerability_score, reliability_grade,
heatmap_asset_ref, thumbnail_asset_ref,
top_regions: list[{region_key, degradation, reliability_grade}],
task_extra: dict          # R5.sample_extra_fields()의 출력. 분류는 비어있을 수 있음
```

**RegionRow.**
```
region_key, region_kind,
intended_area_px, effective_area_px,
mean_degradation, flip_rate, n_valid,
reliability_grade
```

**FlaggedItem.**
```
anchor_key, reason_summary, reliability_reasons: list[str]
```

**설계 의도.** `task_extra`와 `applicable_charts()`가 검출 확장의 유일한 진입점이다. 스키마 나머지는 완전히 태스크 무관이다.

---

## 4. 시각적 구성과 전달 전략

### 4.1 문제의식 — 무엇과 차별화해야 하는가

기존 로버스트니스 벤치마크의 전형적 출력은 **표 하나에 스칼라 점수**다("mCE = 45"). 이 도구가 표준 벤치마크 표만 크게 그려서 보여주면, 리포트 자체가 도구의 차별점을 스스로 지우는 꼴이 된다.

**전달 전략의 핵심.** 요약 스코어카드는 두되, **그 바로 아래에서 즉시 "이 평균 뒤에 숨은 변이"를 보여준다.** 순서 자체가 메시지다.

### 4.2 페이지 구성 (위에서 아래로)

**(1) 헤더 — run 요약 바.** 데이터셋, 모델, 소요 시간, 실패율. 한눈에 훑는 메타정보.

**(2) 스코어카드.** `MetricCard` 나열. 여기까지는 일반 벤치마크 리포트와 비슷하다. 의도적으로 그렇다 — 익숙한 진입점을 준 뒤 바로 다음에서 반전을 준다.

**(3) 취약도 분포.** 히스토그램 + 텍스트 콜아웃. 예: "평균 저하도는 X이지만, 개별 샘플은 Y~Z 범위에 걸쳐 있다 — 아래에서 극단값을 확인하라." 이 문장이 (2)와 (4)를 잇는 서사적 연결고리다.

**(4) "모델이 어디를 보는가" — 취약 샘플 갤러리.** 가장 시각적인 섹션. top-K 취약 샘플 각각에 대해 원본 썸네일 + 공간 히트맵 오버레이 + 신뢰도 배지를 나란히 배치. 표가 아니라 이미지 그리드다. **이 섹션이 도구의 정체성을 가장 직접적으로 전달한다** — 숫자표로는 불가능한, "어디가 문제인지 보는" 경험.

**(5) Region 요약 표.** 정렬 가능한 표(정렬 없이도 기본 정렬로 이미 유용). area와 신뢰도 등급을 항상 병기 — §7.5·§7.4에서 확립한 원칙 그대로.

**(6) 신뢰도 스포트라이트.** `unreliable`로 판정된 항목을 별도로 모아 사유와 함께 나열. "이 결과는 믿지 말라"를 명시적으로 보여주는 섹션. 이게 있어야 (4)의 히트맵을 볼 때 사용자가 신뢰도를 함께 읽는다.

**(7) Provenance (접이식).** 전체 설정·해시·임계값. 평소엔 접혀 있고 필요할 때 펼침. 재현성 요구를 충족하되 주된 서사를 방해하지 않는다.

### 4.3 신뢰도 노출 방식

숫자 옆에 **색상 배지(칩)**를 붙인다: `HIGH`(녹) / `MODERATE`(청) / `LOW`(회) / `UNRELIABLE`(적, 취소선 느낌의 시각 처리). 배지에는 `title` 속성으로 사유를 담아 마우스오버 시 노출(JS 불필요, 순수 HTML `title`). 별도 툴팁 라이브러리를 두지 않는다.

**원칙.** 신뢰도 등급이 없는 곳에 숫자만 단독으로 노출되는 일이 없어야 한다. 모든 degradation·순위 관련 수치는 배지와 함께 다닌다.

### 4.4 취약 샘플 갤러리의 카드 레이아웃

```
┌────────────────────────────┐
│ [원본 썸네일] [히트맵 오버레이] │
│ sample_id            [배지]  │
│ vulnerability_score: 0.83    │
│ 최다 취약 region: r0/c1 (HIGH)│
└────────────────────────────┘
```

카드형 그리드, 반응형(화면 너비에 따라 열 수 조정).

### 4.5 텍스트 콜아웃의 역할

차트나 표 옆에 한두 문장의 해설을 배치하는 것을 표준 패턴으로 둔다. "평균이 낮다고 안전한 게 아니다", "이 등급의 결과는 fill strategy에 따라 방향이 바뀔 수 있다" 같은 문장을 **템플릿에 고정 문구로 넣거나, ReportModel의 `note` 필드로 데이터 기반 생성**한다. 이게 리포트를 "숫자 나열"에서 "진단서"로 바꾸는 장치다.

### 4.6 향후 WebUI로의 확장 경로

지금의 R0(ReportDataAssembler)가 만드는 `ReportModel`이 그대로 향후 WebUI의 API 응답 스키마가 되는 것을 목표로 한다.

- 지금: R0 → JSON 파일 저장 → R4가 그 파일을 읽어 정적 HTML 생성
- 향후: R0 → HTTP 응답으로 직접 반환 → 프론트엔드가 fetch해서 동적 렌더링

이 전환에서 **R0~R3는 전혀 바뀌지 않는다.** 바뀌는 것은 R4(정적 렌더러)가 동적 프론트엔드로 대체되는 것뿐이다. 따라서 v1에서 "정적 HTML을 굳이 잘 만들 필요가 있나"라는 의문이 들 수 있지만, **잘 만들어야 하는 이유는 바로 이 스키마 설계 검증에 있다.** 정적 HTML이 잘 동작한다는 것은 ReportModel 스키마가 실제로 충분하다는 증거이고, 그게 검증되어야 향후 API 설계 비용이 줄어든다.

클러스터링 뷰·필터링 UI 같은 v1.1 기능은 ReportModel에 새 최상위 필드를 추가하는 형태로 확장하며, 기존 필드 구조를 변경하지 않는 것을 원칙으로 한다(하위 호환).

---

## 5. 검출 태스크로의 확장 (설계만, 구현은 후속)

### 5.1 스키마 차원에서 이미 준비된 것

- `MetricCard`는 태스크 무관. 검출 어댑터가 `[mAP, mAR, mean_miss_rate]` 카드를 생성하면 R4는 수정 없이 그대로 렌더링
- `SampleCard.task_extra`가 검출 고유 정보(놓친 객체 수, confidence 변화 등)를 담는 자리
- `RegionRow`는 지표 엔진 설계에서 이미 검출 지표(§7.2, §7.4의 중첩 정책)를 태스크 무관 방식으로 다루도록 설계되어 있어 그대로 재사용 가능

### 5.2 새로 필요해질 것 (v1에서는 다루지 않음)

**객체 단위 서브 카드.** 샘플 하나에 여러 객체가 있으므로, `SampleCard` 아래 `objects: list[ObjectRow]`가 필요하다. `ObjectRow`는 `{object_id, gt_class, miss_in_perturbed, confidence_drop, iou_drop, mask_gt_overlap_ratio}` 정도.

**히트맵에 GT/예측 박스 오버레이.** 현재 DebugViz의 히트맵은 마스크 오버레이만 지원한다. 검출에서는 그 위에 GT 박스와 예측 박스(원본/교란 각각)를 함께 그려야 진단이 의미를 갖는다. 이건 R3가 아니라 **DebugViz(지표 엔진 쪽)의 확장 대상**이며, 리포팅 계층은 그 결과 이미지를 그대로 링크하면 된다.

**중첩 비율 기반 필터.** 분석 모듈 설계(§7.4)의 "포화" 그룹(마스크가 객체를 거의 다 덮은 경우)은 리포트에서 기본적으로 제외하거나 별도 탭으로 분리해야 오해를 막는다.

### 5.3 구현자를 위한 지침

검출 지원을 실제로 붙일 때, R5의 검출 어댑터만 추가하고 R0~R4는 손대지 않는 것이 목표다. 만약 그게 안 되면(즉 검출을 지원하려니 ReportModel 자체를 바꿔야 한다면) 그건 현재 스키마 설계가 충분히 태스크 무관하지 않다는 신호이므로, v1 스키마를 되짚어야 한다.

---

## 6. 검증 계획

### 6.1 층위

| 층위 | 대상 | 방법 |
|---|---|---|
| C1 | ReportModel 조립 정확성 | 합성 metrics/analysis 데이터로 단위 테스트 |
| C2 | HTML의 JS 비의존 | JS 비활성 상태에서 콘텐츠 존재 확인 (raw HTML 파싱, headless 실행 불필요) |
| C3 | 실제 run 재현 | 완료된 L3 실행 결과로 리포트 생성, 육안 검증 |

### 6.2 C1 — 단위 테스트

- top-K/bottom-K 선정이 vulnerability_score 기준으로 정확
- schema_version 불일치 시 명확한 실패
- 신뢰도 등급별 분포 합계가 전체 표본 수와 일치
- `flagged_items.csv`가 `reliability_grade=unreliable`인 항목과 정확히 일치
- 결측(예: 대조군 없음)일 때 해당 섹션이 "데이터 없음"으로 명시되고 조용히 생략되지 않음

### 6.3 C2 — 오프라인·비JS 검증

- 생성된 `report.html`에서 모든 `<script>` 태그를 제거한 사본을 만들어도 핵심 콘텐츠(스코어카드, 표, 이미지)가 그대로 보이는지 확인
- 외부 URL 참조(http/https 시작 링크)가 자산 파일에 존재하지 않는지 정적 검사
- `report/` 폴더를 다른 경로로 옮긴 뒤 상대경로가 깨지지 않는지 확인

### 6.4 C3 — 실제 데이터 재현 (권장 첫 검증)

**이미 완료된 L3(crop-free) 실행 결과에 본 모듈을 적용**하여 다음을 육안 확인한다.

| 확인 항목 | 기대 |
|---|---|
| 취약 샘플 갤러리 최상단 | 패치를 포함한 샘플/영역이 두드러지게 노출 |
| 해당 영역의 신뢰도 배지 | HIGH |
| 비패치 영역 다수의 배지 | LOW 또는 UNRELIABLE (§3.5 후속 분석 결과와 일치해야 함) |
| 취약도 분포 히스토그램 | 패치 관련 극단값과 나머지의 뚜렷한 분리가 시각적으로 보임 |
| 신뢰도 스포트라이트 | crop 버그 조사 당시 발견했던 부호 불일치 사례가 사유와 함께 나타남 |

**이 검증의 의의.** 사람이 스크립트로 수동 분석해 발견했던 것(§3.5 follow-up)을, 리포트가 자동으로 한 화면에 보여줄 수 있는지 확인하는 것이다. 통과하면 리포트가 "장식"이 아니라 실제 진단 도구로 기능함이 입증된다.

---

## 7. 잔여 결정 사항

| 항목 | 결정 시점 |
|---|---|
| `top_k`/`bottom_k` 기본값 | R0 구현 시, 실제 HTML 파일 크기를 보고 조정 |
| 차트 SVG 생성 라이브러리 (matplotlib vs 수기 SVG) | R2 구현 시 |
| 대조군·안정성 분석 결과가 없는 run(대조군 미요청)에서의 리포트 축소 방식 | R0 구현 시 — 섹션을 숨길지, "해당 없음"으로 표시할지 |
| Provenance 섹션의 접이식 구현(JS 유무) | R4 구현 시 — `<details>` HTML 요소로 JS 없이 구현 가능한지 우선 검토 |
| 인쇄용 CSS의 페이지 분할 규칙 | R4 구현 시 |
| 단일 파일 배포(자산 base64 임베드) 옵션 제공 여부 | v1.1 검토 — 지금은 폴더 배포 기본 |

---

## 8. 설계 결정 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| 계산 vs 조립 | 리포팅 계층은 새 통계를 계산하지 않음 | 책임 분리, 지표 엔진·분석 모듈과의 경계 유지 |
| 데이터-표현 분리 | ReportModel(JSON) → HTMLRenderer는 그 소비자 중 하나 | 향후 WebUI 전환 시 R0~R3 무변경 목표 |
| 차트 생성 위치 | 서버사이드 SVG, 클라이언트 차트 라이브러리 미사용 | 오프라인·의존성 원칙, 재현성 |
| 외부 자산 | CDN 등 외부 요청 전면 금지 | 완전 오프라인 산출물 |
| JS 의존 | 콘텐츠는 JS 없이 전부 접근 가능, JS는 점진적 향상만 | 접근성, 감사·인쇄 환경 대응 |
| 서사 구조 | 스코어카드 → 분포 → 취약 샘플 갤러리 → region 표 → 신뢰도 스포트라이트 | 평균 뒤 변이를 드러내는 것이 도구의 정체성 |
| 신뢰도 노출 | 모든 수치에 등급 배지 동반, 단독 숫자 노출 금지 | 오해 방지, L3에서 확인된 실제 위험 대응 |
| 샘플 노출 범위 | top-K/bottom-K만 HTML에, 전량은 CSV/parquet | 파일 크기, 실사용 패턴 |
| 태스크 확장 | TaskPresentationAdapter + MetricCard로 태스크 무관화 | 검출 추가 시 R0~R4 무변경 목표 |
| 자산 재사용 | DebugViz 산출물 재배치, 신규 렌더링 지양 | 중복 구현 방지, 의존 방향 문제 확산 차단 |
| 배포 형태 | 자기완결 폴더 (base64 단일파일은 후속) | 자산 크기와 단순성의 절충 |
