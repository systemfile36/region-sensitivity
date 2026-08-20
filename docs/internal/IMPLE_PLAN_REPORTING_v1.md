# 구현 계획서 (v1 리포팅 계층)
## Spatial Sensitivity Audit Toolkit — Reporting Layer

> 본 계획서는 [REPORT_LAYER_DESIGN_v1.md](REPORT_LAYER_DESIGN_v1.md)의 설계를 현재 저장소 구현 상태와 대조하여 작성한 실행 계획이며, 방향과 틀을 정하기 위한 것이다. 세부 사항은 구현 과정에서 조정한다.
> 전제: 코어([IMPL_PLAN_CORE_v1.md](IMPL_PLAN_CORE_v1.md)), 지표 엔진([IMPLE_PLAN_METRIC_DESIGN_v1.md](IMPLE_PLAN_METRIC_DESIGN_v1.md)), 대조군·안정성 분석([IMPLE_PLAN_CONTROL_STABILITY_v1.md](IMPLE_PLAN_CONTROL_STABILITY_v1.md))이 이미 구현되어 있다. 본 문서는 그 세 결과물(`ssat/core/*`, `ssat/metrics/*`, `ssat/analysis/*`)을 재사용하는 최종 해석 단계를 다룬다.
> 패키지 위치: 리포팅 계층은 `ssat/report/`라는 **신규 최상위 패키지**로 구현한다(코어·지표 엔진·분석 모듈과 형제 관계). 근거는 §3.2 참조.

---

## 1. 현재 구현 상태 대비 격차

설계서(R0~R5)와 현재 저장소를 대조한 결과, **본 계층은 아직 한 줄도 구현되어 있지 않다.** `ssat/`에는 `core/`, `application/`, `metrics/`, `analysis/`, `utils/`만 존재하며 `report/`는 없다.

다만 이번에는 선행 계층 **세 개**(코어, 지표 엔진, 대조군·안정성 분석)가 모두 완성되어 있고, `ssat analyze` CLI까지 이미 연결되어 있다(`ssat/cli.py`, `ssat/application/application.py:461`). 그래서 이 계획의 성격은 지표 엔진·분석 모듈 착수 시점보다 한 단계 더 "조립"에 가깝다 — 새 통계는 거의 없고, 이미 계산되어 흩어져 있는 세 저장소(MetricsStore, AnalysisStore, run_manifest)를 어떻게 정확히 이어붙일지가 관건이다. 코드를 직접 대조한 결과, 설계서가 암묵적으로 전제한 부분에서 **설계서에 없던 격차 여섯 개**가 발견되었다. 이 여섯 가지가 이 계획서 전체의 구조를 결정한다.

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| 1 | AnalysisStore의 실제 파일 목록이 설계서 §1과 다르다 | **격차 — R0가 재구성해야 함** | 설계서 §1은 `[AnalysisStore] reliability grades, control/stability`라고만 뭉뚱그려 적었지만, 실제 `ssat/analysis/store.py`가 쓰는 파일은 `control_comparison.parquet`, `seed_stability.parquet`, `strategy_stability.parquet`, `rank_correlation.parquet`, `strategy_profile.parquet`, `intervals.parquet`, `reliability.parquet` 일곱 개다(파일-당-row타입 원칙, `ssat/analysis/schema.py` 모듈독스트링). **A1의 `AnchorTable`(모든 AnchorKey의 목록)은 어디에도 영속화되지 않는다** — 인메모리 산출물로만 존재하고 `save_analysis`의 인자 목록(`ssat/analysis/store.py:111-128`)에 없다. 따라서 R0가 "이 dump에 어떤 AnchorKey들이 있었는가"를 알려면 `reliability.parquet`(모든 (AnchorKey, metric_name) 쌍에 대해 예외 없이 한 행씩 존재하는 유일한 저장 산출물, `ssat/application/application.py:500-507`의 `compute_reliability` 호출 순서 참조)를 앵커 전체 목록의 사실상 원천으로 삼아야 한다. |
| 2 | RegionRow의 `region_kind`/면적은 AnalysisStore가 아니라 MetricsStore에서 온다 | 격차 — 소스 재배정 | 설계서 §3의 `RegionRow`는 `region_kind, intended_area_px, effective_area_px`를 갖지만, AnalysisStore 어느 parquet에도 이 필드들이 없다(A1 `AnchorRow`가 갖고 있었으나 격차#1처럼 비영속). 이 값들은 이미 `ssat.metrics.store`의 `region_metrics.parquet`(`RegionMetrics`, `ssat/metrics/types.py:164-213`)에 region_key 단위로 존재한다 — **N3 집계가 이미 dataset 전체에 걸쳐 region_key로 평균한 형태이므로 (§1 CONTROL_STABILITY 격차#3이 지적한 "조건축 소실"과 같은 성격) 이 필드들을 가져오는 데는 문제가 없다**(면적/kind는 조건에 따라 바뀌지 않는 값이므로). 즉 RegionRow는 MetricsStore(`region_metrics.parquet`)와 AnalysisStore(`reliability.parquet` 집계, 격차#3)를 region_key로 조인해서 만든다. |
| 3 | RegionRow는 region_key 단위(데이터셋 전체)인데 ReliabilityRow는 AnchorKey 단위(sample × region × invert_mask)다 — **집계 정책이 설계서에 없다** | **격차 — 이번 계획서에서 정책 확정 필요** | `region_metrics.parquet`은 이미 region_key당 한 행(여러 샘플에 걸쳐 평균, `ssat/metrics/aggregate.py:361-388`의 "Sample-grain: each ... is one datum" 집계)인데, `reliability.parquet`은 같은 region_key라도 샘플마다(그리고 invert_mask 극성마다) 별도 행이다. 설계서 §3의 `RegionRow.reliability_grade`는 필드가 하나뿐이라 "여러 샘플의 등급을 하나로 어떻게 합칠지"가 정해져 있지 않다. **결정(§5 단계 2, §8에 기록): 최악값(가장 우려스러운 등급) 정책을 쓴다** — `UNRELIABLE > LOW > MODERATE > HIGH` 순으로, 그 region_key를 공유하는 어떤 anchor라도 UNRELIABLE이면 RegionRow 전체가 UNRELIABLE로 표시된다. 이는 설계서 §4.3의 "오해 방지" 원칙("숫자 옆에 등급 배지 필수")과 일치하는 보수적 선택이다. 부작용(과도한 비관)을 완화하기 위해 **`RegionRow`에 설계서에 없는 `reliability_distribution: dict[str, int]` 필드를 추가**한다 — 이 region_key를 공유하는 anchor들의 등급 분포 전체를 함께 보여주기 위함이며, 설계서 §4.6이 허용한 "기존 필드를 바꾸지 않는 추가적 확장"의 범위 안에 있다. |
| 4 | `spatial_profile.parquet`에는 `invert_mask`도 `perturb_op`도 없다 — SampleCard.top_regions의 크기(magnitude) 출처가 불분명하다 | **격차 — 근거를 갖고 확정** | `ssat.metrics.aggregate._aggregate_spatial_profile`(`ssat/metrics/aggregate.py:337-358`)은 `(sample_id, region_key, metric_name)`로만 묶어 평균한다 — CONTROL_STABILITY 계획서 §1 항목3이 `region_metrics.parquet`에서 지적한 것과 똑같은 "조건축 소실"이 `spatial_profile.parquet`에도 있다. 반면 `reliability.parquet`은 AnchorKey(= invert_mask 포함) 단위지만 등급·플래그만 갖고 크기(magnitude) 수치가 없다. **결정: SampleCard.top_regions의 `degradation` 값은 `spatial_profile.degradation`을 그대로 쓴다.** 근거는 세 가지다 — (a) `ssat.metrics.viz.heatmap`이 히트맵의 픽셀 강도(intensity)를 만드는 데 쓰는 값이 정확히 이 `SpatialProfile.degradation`이다(`ssat/metrics/viz/heatmap.py` `HeatmapView`/`resolve_heatmap_view` 참조) — R3가 이 값으로 그린 히트맵 그림과 R0가 텍스트로 적는 수치가 어긋나면 카드 안에서 그림과 숫자가 모순되므로, **그림과 텍스트가 같은 소스를 참조하는 것이 무엇보다 중요**하다. (b) 같은 region_key가 한 샘플 안에서 invert_mask 두 극성으로 모두 쓰이는 경우는 실제 설정에서 드물다(patch-grid 등 대부분의 region 설계가 anchor당 단일 극성). (c) 조건 평균이라는 사실 자체를 리포트 UI 문구("여러 perturbation 방식의 평균 저하도")로 명시해 오해를 막는다(§4.5 텍스트 콜아웃 원칙과 일치). region_key당 여러 invert_mask 값이 실제로 존재하는 드문 경우엔, 그 region_key에 대해 reliability.parquet에서 최대 크기(가장 우려스러운) anchor의 등급을 짝지어 보여준다. |
| 5 | R1의 `sample_rankings.csv`("top-K 아님, 전량")는 `ReportModel`(top-K만 담음)만으로는 만들 수 없다 | **격차 — R0의 출력 계약을 확장해야 함** | 설계서 §2 원칙("JSON 모델이 먼저")과 §R1("전량은 항상 CSV로 접근 가능") 사이에 이 계획서가 메워야 할 틈이 있다: `report_model.json`은 top-K/bottom-K로 잘려 있지만 `sample_rankings.csv`는 전체 샘플이어야 한다. **결정: R0(`ReportDataAssembler.assemble(...)`)는 `ReportModel` 하나가 아니라 `(ReportModel, full_sample_rankings)` 튜플(또는 이를 감싼 `AssembledReport` 컨테이너)을 반환한다.** `full_sample_rankings`는 `report_model.json`에 직렬화되지 않는 부속 데이터로, R1(CSV 전량 export)과 R2(히스토그램 — 전체 분포가 있어야 유의미함, top-K만으로는 분포를 그릴 수 없음)가 소비한다. R4(HTMLRenderer)는 이 부속 데이터를 절대 건드리지 않는다 — `ReportModel`과 이미 렌더링된 R2/R3 자산만 받는다(설계서 §2 "계산과 표현의 분리" 원칙을 그대로 지키되, "표현"에 필요한 입력의 범위를 이 계획서가 명시적으로 좁힌 것). |
| 6 | `ssat.metrics.viz`(DebugViz)는 지금까지 `ssat/metrics/viz/` 바깥에서 한 번도 import된 적이 없다 | 격차 아님(순수 발견), 리스크로 기록 | `grep -rl "metrics\.viz"` 결과 `ssat/metrics/viz/` 내부 파일들 외에는 아무 곳에도 없다 — CLI에도, `application/`에도 연결되어 있지 않다. `heatmap.py`의 `__all__`이 `resolve_heatmap_view`/`render_heatmap_panel`/`select_spatial_profile_rows`를 이미 공개 API로 노출하고 있으므로 **재사용 자체는 설계상 문제 없다**. 다만 R3(AssetLinker)가 이 모듈의 **첫 패키지 간(cross-package) 소비자**가 된다는 뜻이므로, 지금까지 스크립트에서 1회성으로만 호출되던 이 함수들을 top-K/bottom-K 루프 안에서 반복 호출할 때 matplotlib figure가 제대로 닫히는지(메모리 누수) 등을 이번 단계에서 처음 검증하게 된다(§9 리스크 표). |

**부수 확인.** `dataset_name`이라는 필드는 코어·지표 엔진·분석 모듈 스키마 어디에도 없다(`grep -rn dataset_name ssat/` 결과 0건 — 설계서 §3에만 등장). `run_summary.dataset_name`은 `ResolvedConfig.source_provenance.manifest`의 부모 디렉터리 이름(또는 `config_source`의 stem, `source_provenance`가 없으면 `"unknown"`)에서 파생해야 한다. 또한 설계서 §1 흐름도의 `[DumpReader.manifest]` 박스는 실제로는 `ssat.core.dump`를 직접 열어서가 아니라 **`ssat.metrics.dump_reader.DumpHandle(dump_root).manifest`를 통해서만** 접근한다 — 이미 AnalysisReader/DebugViz가 확립한 "dump 접근은 `metrics.dump_reader`가 유일한 관문" 원칙(`IMPLE_PLAN_CONTROL_STABILITY_v1.md` §3.1)을 리포팅 계층도 그대로 잇는다. 즉 R0는 `dump_dir`도 인자로 받는다(설계서 다이어그램에 이미 암시되어 있었으나 "인자"로 명시된 적은 없었다).

**결론.** 이 계획서는 설계서 §0~§8 전체를 다루는 **신규 구현 계획**이며, 코어·지표 엔진·분석 모듈이라는 세 계층 위에 네 번째, 사용자 대면 계층을 쌓는 작업이다. 핵심 난점은 새 계산이 아니라 — 설계서 스스로 "계산하지 않고 조립한다"고 못박았다 — **서로 다른 grain(anchor 단위 vs region 단위 vs sample 단위)으로 저장된 세 저장소를 어떤 정책으로 하나의 사람이 읽는 문서로 접을 것인가**이다.

---

## 2. 기술 스택과 의존성 방침

### 2.1 신규 의존성: Jinja2 하나만 추가한다

| 용도 | 방법 | 근거 |
|---|---|---|
| HTML 템플릿(R4) | **Jinja2를 `requirements.txt`에 신규 추가** | 대조군·안정성 분석 계획(§2.1)은 "신규 의존성 없음"을 원칙으로 세웠지만, 그건 scipy/sklearn처럼 이미 pandas/numpy로 충분한 통계 계산을 피하기 위한 것이었다. 여기서는 사정이 다르다 — 설계서 §R4 스스로가 "템플릿 엔진(예: Jinja2)으로 데이터와 마크업을 분리"라고 명시했고, 표준 라이브러리의 `string.Template`은 조건문·반복문·템플릿 상속이 없어 §4.2의 7단 페이지 구성(스코어카드/히스토그램/갤러리/표/스포트라이트/접이식 provenance)을 유지보수 가능하게 만들기 어렵다. Jinja2는 순수 파이썬·오프라인·네트워크 호출 없음(§0 오프라인 원칙과 충돌 없음)이며 개발 컨테이너에 이미 (전이적으로) 설치되어 있음을 확인했다(`python3 -c "import jinja2"` 성공, v3.1.2). |
| SVG 차트(R2) | **matplotlib의 SVG 캔버스(`FigureCanvasAgg` 대신 `matplotlib.backends.backend_svg.FigureCanvasSVG`, 또는 `fig.savefig(..., format="svg")`)를 재사용** | matplotlib은 이미 `ssat/metrics/viz/*`가 쓰는 기존 의존성이다(신규 의존성 없음). 설계서 §7이 "matplotlib vs 수기 SVG"를 미정으로 남겼는데, 이미 PNG 히트맵에 matplotlib을 쓰고 있으므로 같은 라이브러리로 SVG를 내보내는 쪽이 새 코드·새 의존성을 만들지 않는다. 결정적으로 확정한다. |
| 썸네일 리사이즈(R3) | Pillow(이미 의존성, `requirements.txt`의 `pillow`) | 신규 의존성 없음. `ssat.core.source.ImageFolderSource`가 이미 Pillow 기반으로 원본을 읽는다. |

`.devcontainer/Dockerfile`, `scripts/install_deps.sh`는 `requirements.txt`를 그대로 `pip install -r`하는 구조이므로 별도 수정이 필요 없다 — 컨테이너 재빌드(또는 `pip install jinja2`) 한 번만 필요하다.

### 2.2 CLI 표면은 이번 계획서에서 바로 추가한다 — 지표 엔진·분석 모듈과 다른 결정

지표 엔진(§2.2)과 분석 모듈(§2.2)은 둘 다 "먼저 라이브러리 API로 완성하고, 반복 호출 패턴이 관찰되면 CLI를 추가"하는 순서를 따랐다. 리포팅 계층은 **그 선례를 따르지 않고 이번 계획서 안에서 `ssat report` CLI를 바로 추가한다.** 근거: 설계서 §0 자체가 리포팅 계층을 "파이프라인에서 사용자가 실제로 결과를 보는 유일한 지점"이라고 규정한다 — `metrics`/`analyze`는 다른 내부 계산의 입력을 만드는 중간 산출물이라 스크립트에서 먼저 반복 호출되며 패턴이 드러날 시간이 필요했지만, 리포트는 애초에 스크립트가 아니라 **사람이 터미널에서 실행해 결과물을 받는** 최종 명령이므로 관찰 기간을 둘 이유가 없다. `ssat run` → `ssat metrics` → `ssat analyze` → `ssat report`로 이어지는 명령 체인이 자연스럽게 완성된다(§5 단계 7).

---

## 3. 디렉터리 구조

```
ssat/
├── core/                              # 기존 (변경 없음)
├── metrics/                           # 기존 (변경 없음, viz/heatmap이 처음으로 외부에서 import됨 — §1 격차#6)
├── analysis/                          # 기존 (변경 없음)
├── application/                       # AuditApplication에 generate_report() 추가
│   ├── application.py                 # + generate_report()
│   └── types.py                       # + ReportRequest / ReportResult / ApplicationErrorCode.REPORT
├── report/                            # ← v1 리포팅 계층 구현 범위 (신규)
│   ├── __init__.py
│   ├── types.py                       # ReportModel 스키마 (§3): MetricCard, SampleCard,
│   │                                  #   RegionRow, FlaggedItem, ReportModel 등
│   ├── errors.py                      # ReportError, ReportSchemaError, ReportDataError
│   ├── adapters.py                    # R5  TaskPresentationAdapter (분류 어댑터만; 검출은 stub)
│   ├── assembler.py                   # R0  ReportDataAssembler
│   ├── exporter.py                    # R1  JSON/CSV export
│   ├── charts.py                      # R2  ChartRenderer (matplotlib SVG)
│   ├── assets.py                      # R3  AssetLinker (metrics.viz.heatmap 재사용)
│   ├── html_renderer.py               # R4  HTMLRenderer (Jinja2 Environment는 Python 문자열
│   │                                  #   상수로 정의 — 저장소 관례상 ssat/ 아래 .py 아닌
│   │                                  #   파일을 두지 않으므로, §5 단계 6 참조)
│   └── static.py                      # R4 부속: CSS/JS 원문을 문자열 상수로 보관
├── presentation.py                    # + format_report()
├── cli.py                             # + `ssat report` 명령
└── ...
tests/
├── unit/
│   ├── test_report_types.py
│   ├── test_report_adapters.py
│   ├── test_report_assembler.py
│   ├── test_report_exporter.py
│   ├── test_report_charts.py
│   ├── test_report_assets.py
│   └── test_report_html_renderer.py
├── integration/
│   └── test_report_synthetic_dump.py    # C1: 코어 미실행, 합성 dump+지표+분석 직접 주입
├── fixtures/
│   └── synthetic_dump_builder.py        # 기존 파일 확장: compute_and_save_analysis() 추가(§5 단계 0)
└── ...
experiments/
└── synthetic_shortcut/
    └── generate_report.py                # C3 (단계 8에서 산출, pytest 밖)
```

### 3.1 구조 설계 의도

**`report/`가 R0~R5에 1:1 대응한다.** 지표 엔진의 `metrics/`가 N0~N5에, 분석 모듈의 `analysis/`가 A0~A7에 대응한 것과 동일한 원칙이다.

**`assembler.py`(R0)가 이 계층 안에서 유일하게 세 저장소(MetricsStore·AnalysisStore·run_manifest)를 동시에 여는 지점이다.** §1 격차#1~#5에서 확정한 조인·집계 정책이 전부 여기 모인다. `exporter.py`/`charts.py`/`assets.py`/`html_renderer.py`는 R0가 만든 `(ReportModel, full_sample_rankings)`만 받는 순수 변환기다 — 이들은 `ssat.metrics.store.load_metrics`나 `ssat.analysis.store.load_analysis`를 직접 호출하지 않는다(단, `assets.py`는 R3의 특수성으로 `ssat.metrics.dump_reader.DumpHandle`과 `ssat.metrics.viz.heatmap`을 예외적으로 더 연다 — 아래 참조).

**`assets.py`(R3)는 원칙과 실제 구현 사이의 긴장을 명시적으로 해소한 결과물이다.** 설계서 §R3는 "이미 생성된 자산의 재배치만 하고 신규 렌더링은 하지 않는다"고 못박았지만, §1 격차#6에서 확인했듯 **DebugViz 산출물이 사전에 어디에도 영속화되어 있지 않다** — `ssat metrics`/`ssat analyze`는 히트맵 PNG를 만들지 않는다. 따라서 R3는 "이미 있는 파일을 복사"하는 대신, **`ssat.metrics.viz.heatmap`의 이미 존재하는 렌더 함수(`select_spatial_profile_rows`, `resolve_heatmap_view`, `render_heatmap_panel`)를 R0가 정한 top-K/bottom-K `sample_id` 집합에 한정해서 호출**한다. 이것은 설계 원칙 위반이 아니다 — 새 시각화 로직·새 통계를 만드는 것이 아니라, **"무엇을 그릴지"는 이미 R0가 다 정했고, R3는 그 결정된 목록에 대해 기존 렌더 코드를 실행할 뿐**이기 때문이다(§0 핵심 원칙 "계산하지 않고 조립한다"의 "계산"은 통계적 계산을 말하는 것이지 렌더링을 말하는 것이 아니라는 점을 이 계획서에서 명시적으로 해석한다).

**`html_renderer.py`/`static.py`가 템플릿·CSS·JS를 별도 파일이 아니라 파이썬 문자열 상수로 갖는다.** 저장소 전체(`ssat/`)에 지금까지 `.py`가 아닌 소스 파일이 하나도 없다(`find ssat -type f ! -name "*.py"` 결과 `__pycache__` 외 0건) — 별도 `.html.jinja`/`.css`/`.js` 파일을 두려면 `pyproject.toml`에 `package-data`/`MANIFEST.in` 설정을 새로 추가해야 한다. 이 계획서는 그 패키징 변경을 피하고, `jinja2.Template`(또는 `DictLoader`)에 파이썬 문자열 상수를 직접 넘기는 방식으로 기존 관례(순수 `.py` 패키지)를 유지한다. **생성된 리포트 폴더(`<run_dir>/report/`) 자체는 물론 진짜 `.html`/`.css`/`.js` 파일이다** — 이건 패키지 소스가 아니라 실행 결과물이므로 문제되지 않는다.

### 3.2 패키지 위치 근거 — `core/`·`metrics/`·`analysis/` 밖에 둔다

분석 모듈 계획의 §3.2가 세운 원칙을 그대로 확장한다.

- 리포팅 계층은 **MetricsStore와 AnalysisStore 양쪽 모두를, 그리고 간접적으로 run_manifest까지** 소비한다(§1). 어느 한 계층의 하위 개념이 아니라 파이프라인의 네 번째 형제 계층이다.
- REPORT_LAYER_DESIGN_v1.md §0은 이 계층을 "지표 엔진과 대조군·안정성 분석 모듈의 산출물을 사람이 소비할 수 있는 형태로 변환하는 **최종 단계**"로 정의한다 — 계산이 전부 끝난 뒤의 표현 계층이라는 별도 책임이다.
- 코어의 §2.2 의존 방향 규칙(단방향, 후단으로의 역참조 금지)이 여기도 그대로 적용된다: `core/*`, `metrics/*`, `analysis/*` → `ssat.report` 역참조는 전면 금지.

### 3.3 의존 방향 규칙

```
report.types              → (없음)
report.errors              → (없음)
report.adapters              → report.types
report.assembler               → report.types, report.adapters,
                                  ssat.metrics.store, ssat.metrics.dump_reader,
                                  ssat.analysis.store, ssat.utils
report.exporter                  → report.types, ssat.utils
report.charts                      → report.types  (matplotlib)
report.assets                        → report.types,
                                        ssat.metrics.dump_reader, ssat.metrics.viz.heatmap,
                                        ssat.utils
report.html_renderer                   → report.types, report.static  (jinja2)
report.static                            → (없음)
```

`ssat.report`는 `ssat.core.dump`나 `ssat.analysis.reader`(AnalysisReader — dump와 지표를 조인하는 A0의 내부 관문)를 직접 import하지 않는다. dump 접근은 `ssat.metrics.dump_reader.DumpHandle`을 통해서만 이루어진다(§1 부수 확인) — 이미 저장된 지표/분석 산출물을 다시 계산하는 것이 아니라 run_manifest의 메타데이터(모델 ID, 소요 시간 등)와 R3의 이미지 렌더링에만 필요하기 때문이다. `report.assets`가 `ssat.metrics.viz.heatmap`을 import하는 것은 §3.1에서 정당화한 명시적 예외다. `core/*`, `metrics/*`, `analysis/*` → `ssat.report` 역참조는 전면 금지하며, import-linter로 강제하는 방식은 기존과 동일하게 따른다.

---

## 4. 개발 환경

기존 Dev Container / Docker Compose 워크스페이스 이미지를 사용하되, §2.1에서 확정한 Jinja2를 `requirements.txt`에 추가하고 컨테이너에 반영(재빌드 또는 컨테이너 내 `pip install -r requirements.txt` 재실행)한다. 테스트는 컨테이너 안에서 `pytest`로 실행한다. C3(§5 단계 8)는 `experiments/synthetic_shortcut/results_crop_free/`(gitignore 대상)에 의존하므로 `.github/workflows/ci.yml`의 `pytest -q` 잡에는 합류하지 않는다 — 분석 모듈의 B3와 같은 위치다.

---

## 5. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행하며, 조건 미충족 시 다음 단계로 넘어가지 않는다. 검증 층위는 설계서 §6이 이미 정의한 **C1(단위)/C2(오프라인·비JS)/C3(실제 데이터)** 명칭을 그대로 쓴다(대조군·안정성 계획서가 B1/B2/B3라는 새 이름을 지어야 했던 것과 달리, 이번엔 설계서 자신이 이미 이름을 정해 두었다).

---

### 단계 0. 스캐폴딩 + 계약 타입 + 테스트 픽스처 확장

> **가장 먼저 하는 이유:** 이후 모든 모듈이 `ReportModel`을 참조하며, §1에서 확정한 집계 정책(격차#3, #4)이 여기서 타입 수준으로 고정된다. 또한 이 계층의 테스트는 dump+metrics+**analysis**가 모두 갖춰진 합성 픽스처가 필요한데, 기존 `synthetic_dump_builder.py`는 metrics까지만 지원한다.

**작업.**
- `report/types.py`:
  - `MetricCard`(`key, label, value, unit, higher_is_better, note`) — 설계서 §R5 그대로.
  - `SampleCard`(`sample_id, gt_label, clean_correct, vulnerability_score, reliability_grade, heatmap_asset_ref, thumbnail_asset_ref, top_regions, task_extra`) — `top_regions`는 `TopRegionEntry(region_key, degradation, reliability_grade)`의 리스트.
  - `RegionRow`(`region_key, region_kind, intended_area_px, effective_area_px, mean_degradation, flip_rate, n_valid, reliability_grade, reliability_distribution`) — 마지막 필드가 §1 격차#3에서 확정한 확장.
  - `FlaggedItem`(`anchor_key_repr, reason_summary, reliability_reasons`) — `anchor_key_repr`는 `AnchorKey`를 그대로 담지 않고 `f"{sample_id}::{region_key}::{invert_mask}"` 같은 문자열로 평탄화한다(`ReportModel`이 `ssat.analysis.types`에 의존하지 않도록 하기 위함 — 아래 참조).
  - `RunSummary`(`dataset_name, n_samples, n_regions_per_sample, n_conditions, duration_seconds, failure_rate, model_id, preprocessing_desc`).
  - `VulnerabilityDistribution`(`histogram_asset_ref, summary_stats: {mean, median, p90, p99}`).
  - `ReportModel`(설계서 §3의 최상위 스키마 그대로: `meta, run_summary, scorecard, vulnerability_distribution, sample_rankings, region_summary, reliability_spotlight, provenance`).
  - **`report.types`는 `ssat.analysis.types`나 `ssat.metrics.types`를 import하지 않는다** — `AnchorKey`/`ReliabilityGrade` 같은 타입은 문자열/얕은 값으로 평탄화해 옮겨 담는다. 이는 §3.3의 의존 방향 규칙을 지키는 동시에, `report_model.json`이 향후 WebUI API 응답으로 그대로 쓰일 때 파이썬 dataclass가 아니라 JSON-호환 원시값만 있으면 되기 때문이다(설계서 §4.6).
  - `ReportGrade`는 `ssat.analysis.types.ReliabilityGrade`의 값(`"high"/"moderate"/"low"/"unreliable"`)과 문자열로 동일하게 맞추되(사람이 읽을 때 혼란이 없도록), 별도 `Enum`으로 재정의한다(타입 의존을 만들지 않기 위해).
- `report/errors.py`: `ReportError`(base), `ReportSchemaError`, `ReportDataError`(예: 요청한 metric_name이 어느 저장소에도 없을 때).
- `report/__init__.py`에 공개 API 확정.
- `requirements.txt`에 `jinja2` 추가.
- `tests/fixtures/synthetic_dump_builder.py`에 **`compute_and_save_analysis(dump_root, metrics_dir, analysis_dir, ...)` 추가** — 기존 `compute_and_save_metrics`가 N2~N4를 이어붙인 것과 같은 방식으로, `AnalysisReader → ComparisonIndexer → A2~A6 → save_analysis`를 이어붙인다. `ssat/application/application.py:461-549`의 `AuditApplication.analyze()`가 이미 이 순서를 정확히 구현해 두었으므로, 그 로직을 그대로 재사용/위임하는 얇은 헬퍼로 만든다(새 로직을 만들지 않는다).

**테스트.**
- `python -c "import ssat.report"` 성공.
- `ReportModel`/`SampleCard`/`RegionRow`가 `dataclasses.asdict` 또는 자체 `to_dict()`로 JSON 직렬화 가능함을 확인(순환 참조·비직렬화 가능 필드가 없는지).
- `report.types`가 `ssat.analysis`나 `ssat.metrics`를 import하지 않는지 정적 검사(예: `ast`로 import 목록 확인, 또는 import-linter 규칙 추가).
- `compute_and_save_analysis` 확장 헬퍼로 만든 `analysis_dir`가 `ssat.analysis.store.load_analysis`로 다시 읽혔을 때 `AuditApplication.analyze()`를 직접 호출한 것과 동일한 결과를 냄을 확인(회귀 방지).

**성공 조건.**
- 패키지 import 성공, 직렬화 가능성 확인.
- `report.types`의 의존 방향 위반 없음.
- 확장된 테스트 픽스처가 실제 `analyze()` 결과와 일치.

---

### 단계 1. R5. TaskPresentationAdapter (분류 어댑터만)

**작업.** `report/adapters.py`.
- `TaskPresentationAdapter` 프로토콜(설계서 §R5 그대로): `summarize_performance(...) -> list[MetricCard]`, `sample_extra_fields(...) -> dict`, `applicable_charts() -> list[str]`.
- `ClassificationAdapter` 구현:
  - `summarize_performance`: `sample_metrics`(primary_metric으로 필터된) 로부터 다음 카드를 만든다 — `accuracy`(`mean(clean_correct)`, §1 격차#6에서 확정한 "단순 산술 축약은 조립의 일부" 원칙에 따라 여기서 계산해도 됨), `mean_{primary_metric}`(`region_metrics`/`sample_metrics.metric_mean`의 평균), `flip_rate`(primary metric이 binary일 때만; `AggregationResult`가 이미 계산해 둔 `flip_rate` 필드를 그대로 옮긴다 — 새 계산 아님).
  - `sample_extra_fields`: 분류는 빈 dict를 반환(설계서 §R5 "분류는 비어있을 수 있음").
  - `applicable_charts`: `["vulnerability_histogram", "region_bar", "fill_strategy_correlation_heatmap"]` (마지막은 `available_analyses.fill_strategy_stability`가 True일 때만).
- 검출 어댑터는 이번 계획서 범위 밖(설계서 §5 그대로) — `DetectionAdapter`는 클래스 골격과 `NotImplementedError`만 남겨 향후 R0~R4를 건드리지 않고 이 파일 하나만 확장하면 된다는 것을 코드로 보여준다.

**테스트.**
- 고정된 `sample_metrics`/`region_metrics` 합성 입력으로 `accuracy`/`mean_margin_drop`/`flip_rate` 카드 값이 손계산과 일치.
- primary_metric이 continuous(binary 아님)일 때 `flip_rate` 카드가 생성되지 않음(또는 `note`로 "해당 없음" 명시 — §1의 "결측은 조용히 생략하지 않는다" 원칙).
- `applicable_charts`가 `fill_strategy_stability=False`일 때 상관 히트맵을 제외함.

**성공 조건.**
- 스코어카드 값 전부 손계산 일치.
- 검출 어댑터가 인스턴스화 시점에는 실패하지 않고 호출 시점에만 `NotImplementedError`를 던짐(향후 확장 지점이 깨끗하게 분리되어 있음을 보증).

---

### 단계 2. R0. ReportDataAssembler (핵심 조인 단계)

> **이 계획서의 병목.** §1의 격차#1~#5가 전부 여기서 코드로 고정된다. 잘못되면 그 위의 R1~R4 전부가 잘못된 조립 위에서 렌더링된다 — 분석 모듈 계획의 "A0가 병목"과 정확히 같은 위치.

**작업.** `report/assembler.py`의 `ReportDataAssembler`.
- 생성자: `ReportDataAssembler(dump_dir, metrics_dir, analysis_dir=None, *, adapter, top_k=20, bottom_k=20)`. `analysis_dir=None`은 대조군·안정성 분석을 아예 돌리지 않은 run을 위한 것(§1 부수 확인, 설계서 §6.2 C1 "결측 시 섹션이 '데이터 없음'으로 명시").
- `assemble(primary_metric) -> AssembledReport`(`ReportModel` + `full_sample_rankings` 튜플, §1 격차#5):
  1. `MetricsStore`(`item_metrics` 불필요, `AggregationResult` + `MetricsManifest`만) + `DumpHandle(dump_dir).manifest`를 읽어 `RunSummary`를 조립(§1 부수 확인의 파생 규칙: `dataset_name`은 `source_provenance`에서, 없으면 `"unknown"`).
  2. `analysis_dir`가 주어졌다면 `AnalysisStore`의 7개 parquet + `AnalysisManifest`를 읽고, `verify_source_metrics`로 metrics-analysis 정합성을 확인(이미 있는 함수 재사용). 주어지지 않았다면 `scorecard`의 대조군 관련 카드와 `reliability_spotlight`를 "해당 없음"으로 채운다.
  3. `adapter.summarize_performance(...)`로 `scorecard` 생성.
  4. `full_sample_rankings`: `sample_metrics`(primary_metric으로 필터)를 `vulnerability_score` 내림차순 정렬 — **전체 샘플**(§1 격차#5).
  5. `sample_rankings.most_vulnerable`/`most_robust`: `full_sample_rankings`의 상위/하위 `top_k`/`bottom_k`에 대해 `SampleCard`를 만든다. `top_regions`는 §1 격차#4에서 확정한 대로 `spatial_profile.degradation` + `reliability.parquet`(있으면)에서 짝짓는다.
  6. `region_summary.rows`: `region_metrics.parquet`(있는 모든 region_key)를 `reliability.parquet`(있으면, region_key로 group)와 조인해 §1 격차#3의 최악값 정책으로 `RegionRow`를 만든다. `analysis_dir`가 없으면 `reliability_grade`는 전부 "해당 없음"(등급 없음을 나타내는 전용 값, `FALSE`로 오염시키지 않음 — 분석 모듈의 `unavailable≠false` 원칙을 리포팅 계층까지 이어간다).
  7. `reliability_spotlight.flagged_examples`: `reliability.parquet`에서 `reliability_grade == UNRELIABLE`인 행 전부.
  8. `vulnerability_distribution.summary_stats`: `full_sample_rankings`의 `vulnerability_score` 배열에서 mean/median/p90/p99 (numpy, §1 격차#5에서 확정한 "단순 기술통계는 조립"의 경계 안).
  9. `provenance`: `metrics_manifest`/`analysis_manifest`(있으면)의 해시·임계값 요약 + 원본 파일 절대경로.

**테스트.**
- `tests/integration/test_report_synthetic_dump.py`: 단계 0에서 확장한 `compute_and_save_analysis`로 dump+metrics+analysis 삼종 세트를 합성 구성해:
  - top-K/bottom-K가 `vulnerability_score` 기준으로 정확히 선정됨(대조군·안정성 계획의 C1 재사용 패턴).
  - `full_sample_rankings`의 길이가 `n_samples`와 정확히 일치(잘리지 않음), `ReportModel.sample_rankings`는 `top_k+bottom_k` 이하로 잘림.
  - `analysis_dir=None`으로 호출했을 때 `reliability_spotlight`가 빈 리스트가 아니라 **"해당 없음" 마커**로 채워지고, `scorecard`의 대조군 카드가 조용히 사라지지 않고 `note="분석 미실행"` 같은 형태로 남음(설계서 §6.2 C1의 "결측 시 명시" 요구를 그대로 단위 테스트로 고정).
  - RegionRow 최악값 정책: 같은 region_key를 공유하는 두 샘플의 등급이 `HIGH`/`UNRELIABLE`로 다를 때 `RegionRow.reliability_grade == UNRELIABLE`이고 `reliability_distribution == {"high": 1, "unreliable": 1, ...}`.
  - SampleCard.top_regions의 `degradation`이 `spatial_profile.parquet`의 값과 정확히 일치(손으로 넣은 합성 데이터와 대조).
  - `region_metrics`와 `reliability` 양쪽에 존재하지 않는 region_key(예: control 전용 region)가 `region_summary.rows`에 나타나지 않음(§1 부수 확인 — control은 N3 집계에서 이미 제외).

**성공 조건.**
- 위 6개 케이스 전부 통과.
- `analysis_dir=None` 경로가 예외 없이 완결된 `ReportModel`을 만들어냄(대조군·안정성 분석은 v1의 선택 기능이라는 원칙이 리포팅 계층까지 깨지지 않고 이어짐).

---

### 단계 3. R1. Exporter (JSON/CSV)

**작업.** `report/exporter.py`.
- `export(assembled: AssembledReport, output_dir: Path) -> ExportedPaths`:
  - `report_model.json`: `ReportModel`을 그대로 직렬화(`json.dumps(..., sort_keys=True)` — 재현성 있는 바이트열, 다른 저장소의 manifest들과 같은 관례).
  - `sample_rankings.csv`: `full_sample_rankings` 전량(§1 격차#5).
  - `region_summary.csv`: `region_summary.rows` 그대로 평탄화(`reliability_distribution` dict는 `high_count,moderate_count,low_count,unreliable_count` 컬럼으로 펼침).
  - `flagged_items.csv`: `reliability_spotlight.flagged_examples`와 정확히 일치(설계서 §6.2 C1 요구).

**테스트.**
- `report_model.json`을 다시 읽어 `ReportModel`로 역직렬화했을 때 원본과 필드 단위로 일치.
- `sample_rankings.csv`의 행 수가 `n_samples`와 일치(top-K로 잘리지 않았는지 회귀 테스트로 고정 — 설계서 §R1 "전량은 항상 CSV로 접근 가능"의 핵심 검증).
- `flagged_items.csv`가 `reliability_grade=unreliable`인 항목과 정확히 일치(설계서 §6.2 C1 그대로).
- 결정론: 같은 `AssembledReport`로 두 번 export했을 때 파일 바이트가 완전히 동일(정렬 키 고정 확인).

**성공 조건.**
- 왕복(export → reload) 일치, 전량 CSV 검증, 결정론 확인 전부 통과.

---

### 단계 4. R2. ChartRenderer (SVG)

**작업.** `report/charts.py`.
- `render_vulnerability_histogram(full_sample_rankings, *, seed=0) -> str`(SVG 문자열): `vulnerability_score` 배열을 matplotlib으로 히스토그램, SVG 캔버스로 저장. 난수 요소 없음(단순 binning)이므로 seed는 향후 지터링 등을 대비한 예비 인자.
- `render_region_bar(region_summary_rows) -> str`: region_key별 `mean_degradation` 막대, `reliability_grade`(§1 격차#3의 최악값)로 색 구분(§4.3 배지 색상 — HIGH 녹/MODERATE 청/LOW 회/UNRELIABLE 적을 막대 색에도 동일하게 적용해 배지-차트 간 색 일관성 유지).
- `render_fill_strategy_correlation(rank_correlation_rows) -> str | None`: `rank_correlation.parquet` 원본(요약 아님, `op_a, op_b, spearman`)이 있을 때만(`applicable_charts()`에 포함된 경우만) op×op 상관 히트맵.
- 모든 함수는 **매 호출마다 `plt.close(fig)`**(또는 `Figure` 객체를 직접 만들어 `pyplot` 전역 상태를 아예 쓰지 않음 — `ssat/metrics/viz/heatmap.py`가 이미 `FigureCanvasAgg`를 직접 쓰고 `pyplot`을 안 쓰는 패턴을 그대로 따른다) — §1 격차#6 리스크(matplotlib figure 누수)를 방지.

**테스트.**
- 같은 입력으로 두 번 렌더링했을 때 SVG 바이트가 동일(재현성, 설계서 §R2 "같은 입력이면 같은 SVG가 나와야 한다").
- 생성된 SVG 안에 `http://`/`https://` 문자열이 없음(외부 참조 없음, 폰트 임베드 없이 벡터 패스만 사용).
- `rank_correlation_rows`가 빈 시퀀스일 때 `render_fill_strategy_correlation`이 `None`을 반환(예외 없이).
- 100개 샘플 규모로 반복 호출했을 때(top_k+bottom_k 루프를 흉내) 열린 figure 개수가 0으로 수렴(`matplotlib.pyplot.get_fignums()` 또는 객체 참조 카운트로 확인 — §1 격차#6 리스크의 회귀 테스트).

**성공 조건.**
- SVG 재현성·오프라인성 확인, figure 누수 없음 확인.

---

### 단계 5. R3. AssetLinker

**작업.** `report/assets.py`.
- `link_assets(assembled, dump_dir, output_dir, *, primary_metric) -> AssetManifest`:
  1. `DumpHandle(dump_dir).manifest.resolved_config.source_provenance`가 `None`이면 **갤러리 섹션 전체를 "원본 이미지 없음"으로 표시하고 조기 반환**(§1 격차#6 인접 위험 — `ssat.metrics.viz._shared.open_image_source`가 `source_provenance` 없이는 애초에 실패하므로, 실패를 전파시키지 않고 여기서 먼저 감지).
  2. `sample_ids = {top-K ∪ bottom-K}`(`ReportModel.sample_rankings`에서). `ssat.metrics.viz.heatmap.select_spatial_profile_rows(spatial_profile, metric_name=primary_metric, sample_ids=sample_ids)` + `resolve_heatmap_view` + `render_heatmap_panel`을 그대로 호출해 `report/assets/img/heatmaps/sample_<id>.png`로 저장(§3.1에서 정당화한 "기존 렌더 함수 재사용" 원칙).
  3. 썸네일: 같은 `HeatmapView.original` 배열을 Pillow로 축소해 `report/assets/img/thumbnails/sample_<id>.png`로 저장(새 픽셀 계산이 아니라 리사이즈뿐).
  4. `DebugVizError`(예: `random_area_match` region이라 재현 불가 — `ssat/metrics/viz/heatmap.py`의 알려진 제약)가 개별 샘플에서 나면, 그 샘플의 `heatmap_asset_ref`만 `None`으로 두고 나머지는 계속 진행(전체 리포트 생성이 한 샘플 때문에 실패하지 않도록).

**테스트.**
- top-K/bottom-K 합집합 크기만큼 정확히 PNG가 생성됨.
- `source_provenance`가 없는 합성 dump에서 예외 없이 "자산 없음" 상태로 완결됨.
- 개별 샘플이 `DebugVizError`를 내는 상황(예: `random_area_match` 합성 region)을 흉내내 그 샘플만 `None`이고 나머지는 정상 생성됨.
- `report/`를 다른 경로로 옮긴 뒤에도 `ReportModel`의 `heatmap_asset_ref`가 **상대경로**라서 깨지지 않음(설계서 §6.3 C2 요구 — 폴더 이동 시 상대경로 보존, 이 단계에서 절대경로를 쓰지 않도록 고정).

**성공 조건.**
- 자산 생성·결측 축소·상대경로 보존 전부 통과.

---

### 단계 6. R4. HTMLRenderer + C1/C2 검증

**작업.** `report/html_renderer.py`, `report/static.py`.
- Jinja2 `Environment`(`autoescape=True` — 사용자 제공 문자열(가설적으로 파일 경로 등)이 섞여 들어가므로 XSS/마크업 깨짐 방지, 오프라인 리포트라도 원칙적으로 켠다)에 §3.1에서 정당화한 문자열 상수 템플릿을 등록.
- 페이지 구성은 설계서 §4.2의 7단을 그대로 따른다: 헤더 → 스코어카드 → 취약도 분포(히스토그램+콜아웃) → 취약 샘플 갤러리(카드 그리드, §4.4 레이아웃) → Region 요약 표 → 신뢰도 스포트라이트 → Provenance(`<details>` 태그로 JS 없이 접이식 — 설계서 §7 미정 항목을 여기서 확정).
- 신뢰도 배지: `title` 속성에 `reliability_reasons`를 넣어 마우스오버로 사유 노출(설계서 §4.3, 별도 툴팁 JS 없음).
- `assets/css/style.css`, `assets/js/enhance.js`(표 정렬 등 점진적 향상)를 `report/static.py`의 상수에서 그대로 파일로 씀.
- `report_manifest.json`: `report_schema_version, source_manifest_hashes{run,metrics,analysis}, top_k, bottom_k, generated_at`.

**테스트 — C1(단위).**
- 템플릿 렌더링이 `ReportModel`의 모든 최상위 섹션에 대해 예외 없이 완결.
- `analysis_dir=None`으로 조립된 `ReportModel`을 넣었을 때 신뢰도 관련 섹션이 "데이터 없음" 문구로 채워지고 섹션 자체가 사라지지 않음(§6.2 C1과 §5 단계 2 테스트의 HTML 레벨 재확인).

**테스트 — C2(오프라인·비JS, 설계서 §6.3 그대로).**
- 생성된 `report.html`에서 모든 `<script>` 태그를 제거한 사본을 만들어도 스코어카드·표·이미지(`<img>` 태그, alt 텍스트 포함)가 raw HTML 파싱으로 확인 가능.
- `report/` 폴더 전체 파일(HTML/CSS/JS/SVG)에서 `http://`/`https://`로 시작하는 속성 값이 없는지 정적 검사.
- `report/` 폴더를 `tmp_path`의 다른 하위 경로로 통째로 복사한 뒤 브라우저 없이 각 자산 상대경로가 실존 파일을 가리키는지 확인(파일시스템 존재 확인만으로 충분, headless 브라우저 불필요 — 설계서 §6.3이 명시).

**성공 조건.**
- C1 전 섹션 렌더링 완결, 결측 시 명시.
- C2 세 검증(비JS 콘텐츠 존재, 외부 참조 0건, 폴더 이동 후 상대경로 보존) 전부 통과.

---

### 단계 7. Application/CLI 통합

**작업.**
- `ssat/application/types.py`: `ReportRequest`(`dump, metrics_dir=None, analysis_dir=None, report_dir=None, primary_metric=DEFAULT_PRIMARY_METRIC, top_k=20, bottom_k=20` — `ComputeMetricsRequest`/`AnalyzeRequest`와 동일한 "co-located 기본값" 관례, `report_dir` 기본값 `<dump>/report`), `ReportResult`(`dump, metrics_dir, analysis_dir, report_dir, n_samples, n_regions, grade_distribution, generated_at`). `ApplicationErrorCode.REPORT = "report_error"` 추가.
- `ssat/application/application.py`: `AuditApplication.generate_report(request) -> ReportResult` — `compute_metrics`/`analyze`와 동일한 구조(경로 정규화 → 하위 계층 호출 → 예외를 `ApplicationError`로 매핑 → 결과 객체 반환). `analysis_dir`가 존재하지 않는 경로면(디렉터리 자체가 없음) 조용히 `None`으로 취급해 R0의 "분석 없음" 경로로 넘긴다.
- `ssat/presentation.py`: `format_report(result)` — 기존 `format_metrics`/`format_analysis`와 같은 스타일.
- `ssat/cli.py`: `report` 명령 추가(`dump` 인자, `--metrics-dir`, `--analysis-dir`, `--report-dir`, `--primary-metric`, `--top-k`, `--bottom-k`, `--json`) — 기존 `metrics_command`/`analyze_command`와 동일한 형태.

**테스트.**
- `tests/integration/test_application_api.py`에 `generate_report` 케이스 추가: 합성 dump+metrics(+analysis 있음/없음 두 경로)로 `<dump>/report/report.html`이 실제로 생성됨.
- `tests/integration/test_cli.py`에 `ssat report` 케이스 추가(`--json` 옵션 포함).
- 존재하지 않는 `--analysis-dir`를 줬을 때 CLI가 실패하지 않고 "분석 없음" 리포트를 생성함(에러가 아니라 의도된 축소임을 확인).

**성공 조건.**
- CLI/Application 양쪽에서 end-to-end로 `report.html`이 생성되고, 분석 유무 두 경로 모두 통과.

---

### 단계 8. C3. 실제 데이터 검증 (핵심 검증)

설계서 §6.4를 그대로 계승하되, **실행 대상 dump 선정을 이번 계획서에서 구체화**한다.

**대상 선정 — 분석 모듈의 B3(§5 단계9)와 다른 이유.** 분석 모듈의 B3는 `shortcut_A_{constant_fill,mean_fill,blur,gaussian_noise,patch_shuffle}` **다섯 개의 별도 dump**를 스크립트 레벨에서 combine해야 했다(각 dump가 fill strategy 하나씩만 담고 있었기 때문에, `analyze_control_stability.py`가 "따로 저장된 `analysis/` 없이" 즉석에서 item_values를 합쳤다 — `experiments/synthetic_shortcut/analyze_control_stability.py` 모듈독스트링). 리포팅 계층은 **단일 dump+metrics+analysis 세 쌍**이 필요하므로(`generate_report`가 하나의 `analysis_dir`만 받음, §5 단계 7), 다섯 개를 합치는 그 스크립트를 그대로 재사용할 수 없다.

다행히 로컬에 이미 정확히 맞는 데이터가 있다: **`experiments/synthetic_shortcut/results_crop_free/dumps/shortcut_A_all_ops_thresholds_crop_free`**(`run_threshold_validation_full.py`가 생성, `RUN_ID = "shortcut_A_all_ops_thresholds_crop_free"`) — 5가지 fill strategy + 대조군(`"controls": [{"match_area_of": PATCH_REGION_ID, ...}]`) + 다중 seed가 **하나의 dump**에 전부 들어 있다(같은 스크립트 §51-121 확인). metrics는 이미 계산되어 있고(`results_crop_free/metrics/shortcut_A_all_ops_thresholds_crop_free/`), analysis만 아직 없다(`find ... -iname analysis` 결과 없음).

**절차.**
1. `experiments/synthetic_shortcut/generate_report.py` 작성: (a) 이미 있는 metrics로 `ssat analyze`(또는 `AuditApplication.analyze`)를 이 dump에 대해 1회 실행해 `analysis/`를 만들고, (b) `ssat report`(또는 `AuditApplication.generate_report`)를 실행해 `report/`를 만든다. 로컬에 해당 디렉터리가 없는 환경(신규 체크아웃, CI)에서는 실행 대상이 아님을 docstring과 이 문서에 명시하고, pytest collection에서 제외한다(분석 모듈 B3와 동일한 이유).
2. 생성된 `report.html`을 육안으로 연다(브라우저에서 파일 경로로 직접 열기 — 서버 불필요, §0 완전 오프라인 원칙의 실사용 확인이기도 하다).

**검증 질문(설계서 §6.4 표를 이 dump에 맞게 구체화, 대조군·안정성 계획서 §5 단계9가 이미 확보한 기대값을 그대로 인용).**

| 확인 항목 | 기대 | 근거 |
|---|---|---|
| 취약 샘플 갤러리 최상단 | 패치 region을 포함한 샘플이 두드러지게 노출 | `docs/L3_Synthetic-Shortcut Experiment Report.md`, Q1/Q2/Q4 PASS |
| 패치 region(0,0)의 RegionRow 배지 | `HIGH` | 동일 |
| 비패치 region 다수의 RegionRow 배지 | 상당수 `UNRELIABLE`(§1 격차#3의 최악값 정책이 여기서 실제로 드러나야 함 — fill strategy 간 부호가 갈리므로 sign_consistent=false인 샘플이 존재) | `IMPLE_PLAN_CONTROL_STABILITY_v1.md` §5 단계9 |
| 취약도 분포 히스토그램(R2) | 패치 관련 극단값과 나머지의 분리가 시각적으로 보임 | 동일 |
| 신뢰도 스포트라이트 | crop-free 버전에서도 fill strategy 간 부호 불일치 사례가 사유와 함께 나타남 | `docs/L3_Synthetic-Shortcut Experiment Report.md` "Sign-Group Premise Re-examination" |
| Region 요약 표의 `reliability_distribution` | 비패치 region의 경우 `unreliable`이 아닌 등급도 소수 섞여 있어(모든 fill strategy가 항상 반대 부호는 아니므로) 최악값 정책의 부작용 완화 필드(§1 격차#3)가 실제로 유의미한 정보를 담고 있음을 확인 | 이번 계획서 §1의 설계 결정 검증 |

**성공 조건.**
- `report.html`이 예외 없이 생성되고 위 6개 항목이 육안으로 확인됨.
- 특히 "Region 요약 표의 reliability_distribution" 항목은 §1 격차#3에서 이 계획서가 새로 도입한 필드이므로, 이게 실제로 빈 장식이 아니라 정보를 담고 있는지가 이 단계 고유의 성공 기준이다 — 만약 모든 region이 분포 없이 단일 등급으로만 나온다면 최악값 정책의 필요성 자체를 재검토해야 한다.
- 기준 미달 시 R0(§5 단계 2)의 조인·집계 정책까지 거슬러 올라가 재검증한다.

---

## 6. 단계 간 의존과 병렬화

```
0 ──> 1 ──> 2 ──┬──> 3 ──┐
                 ├──> 4 ──┤──> 6 ──> 7 ──> 8
                 └──> 5 ──┘
```

- **단계 3(R1 Exporter), 단계 4(R2 ChartRenderer), 단계 5(R3 AssetLinker)는 서로 독립적이다** — 셋 다 단계 2(R0)의 출력(`AssembledReport`)만 있으면 되므로 병행 가능(분석 모듈 계획의 A2/A3/A5 병렬화와 같은 구조).
- **단계 6(R4)은 3·4·5 전부가 끝나야 시작 가능하다** — HTML이 세 산출물(JSON 경로, SVG, 이미지 자산)을 전부 참조하기 때문이다.
- **단계 0~2가 병목이다.** 특히 단계 2(R0)에서 §1 격차#3·#4·#5의 조인·집계 정책을 잘못 구현하면, 그 위에 쌓이는 R1~R4 전부가 잘못된 조립 위에서 계산된다 — 분석 모듈 계획의 "A0가 병목" 결론과 정확히 같은 위치에 R0가 있다.

---

## 7. 테스트 전략

| 층위 | 범위 | 대응 단계 | 실행 방식 |
|---|---|---|---|
| C1. 단위·조립 정확성 | ReportModel 조립(top-K 선정, 집계 정책, 결측 축소), export 왕복, 스코어카드 계산 | 단계 1, 2, 3, 6 | pytest, `synthetic_dump_builder`(단계 0 확장분 포함)로 코어 미실행 합성 dump+metrics+analysis 직접 주입 |
| C2. 오프라인·비JS | HTML의 JS 비의존, 외부 참조 0건, 폴더 이동 후 상대경로 보존 | 단계 5, 6 | pytest, raw HTML/파일시스템 파싱(headless 브라우저 불필요) |
| C3. 실제 run 재현 | 전체 파이프라인의 실사용 검증 | 단계 8 | 별도 스크립트, CI 밖, 로컬 산출물(`results_crop_free/`) 필요 |

C1·C2는 지표 엔진·분석 모듈의 `unit`/`integration` 계층과 동일하게 기본 `pytest` collection에 포함되어 `.github/workflows/ci.yml`의 단일 `pytest -q` 잡에 자연히 합류한다. GPU가 필요한 테스트는 없다(pandas/numpy/matplotlib/jinja2/Pillow 연산과 파일 I/O만 수행).

---

## 8. 잔여 결정 사항의 처리 시점

| 항목 | 결정 시점 | 결정 내용(확정/제안) |
|---|---|---|
| 모듈 패키지 위치 | 이 계획서 작성 시 확정 | `ssat/report/` (§3.2) |
| 템플릿 엔진 | 이 계획서 작성 시 확정 | Jinja2 신규 의존성 추가(§2.1) — 유일하게 "신규 의존성 없음" 선례를 벗어나는 결정이며, 명시적으로 근거를 남김 |
| 차트 SVG 생성 라이브러리 | 이 계획서 작성 시 확정 | matplotlib SVG 캔버스 재사용(§2.1), 신규 의존성 없음 |
| AnalysisStore 파일 목록 재확인 | 이 계획서 작성 시 확정 | 설계서 §1의 뭉뚱그린 표기 대신 실제 7개 parquet + coverage_report.json + analysis_manifest.json을 R0가 직접 참조(§1 격차#1) |
| RegionRow의 다중-샘플 등급 집계 정책 | 이 계획서 작성 시 확정 | 최악값(worst-case) 정책 + `reliability_distribution` 서브필드 추가(§1 격차#3, 스키마 확장) |
| SampleCard.top_regions의 크기(magnitude) 소스 | 이 계획서 작성 시 확정 | `spatial_profile.degradation`(§1 격차#4) — R3 히트맵과 동일 소스를 써서 그림-텍스트 불일치 방지 |
| R1 전량 CSV와 R0 top-K ReportModel의 관계 | 이 계획서 작성 시 확정 | R0가 `(ReportModel, full_sample_rankings)`를 함께 반환(§1 격차#5) |
| Report v1의 metric_name 스코프 | 이 계획서 작성 시 확정 | `primary_metric` 하나만 다룬다 — 다중 metric 탭/섹션은 v1.1(설계서가 명시하지 않은 부분을 확정) |
| CLI 도입 시점 | 이 계획서 작성 시 확정 | 지표 엔진·분석 모듈과 달리 관찰 기간 없이 단계 7에서 바로 추가(§2.2, 근거 명시) |
| `top_k`/`bottom_k` 기본값 | 이 계획서 작성 시 확정 | 설계서 §R0 기본값(각 20) 그대로 채택, 실제 HTML 크기를 단계 8에서 보고 필요 시 조정 |
| 대조군·안정성 분석 결과가 없는 run에서의 리포트 축소 방식 | 이 계획서 작성 시 확정 | "해당 없음" 명시(섹션 숨김 아님) — 설계서 §6.2 C1 요구를 그대로 채택 |
| Provenance 섹션의 접이식 구현 | 이 계획서 작성 시 확정 | `<details>` HTML 요소, JS 불필요(설계서 §7 미정 항목 확정) |
| 인쇄용 CSS의 페이지 분할 규칙 | 단계 6 | 구현 중 실측 후 확정(설계서 §7 그대로 이월) |
| 단일 파일 배포(base64 임베드) 옵션 | v1.1 검토 | 설계서 §7 원안 유지 — 이번 계획서 범위 밖 |

---

## 9. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| R0가 세 저장소(MetricsStore·AnalysisStore·run_manifest)를 잇는 유일한 지점이라, 조인·집계 정책의 버그가 이후 R1~R4로 조용히 전파됨 | 단계 2의 성공 조건을 "top-K 정확, 전량 미절단, 결측 명시, 최악값 정책 정확, magnitude 소스 일치, control region 미노출"의 6개 명시적 케이스로 못박고, 단계 3~8은 이 결과만 소비하도록 강제(§3.1) |
| `spatial_profile`/`region_metrics`의 조건축 소실로 `SampleCard.top_regions`/`RegionRow.mean_degradation`이 여러 perturb_op의 평균이라는 근사치임 | R0가 만드는 문구(§4.5 텍스트 콜아웃)에 "여러 perturbation 방식의 평균"임을 명시해 오해를 막는다(§1 격차#4) — 새 조건별 저장소를 만들지 않고 문서화로 해소 |
| R3가 리포트 생성 시점에 PNG를 새로 렌더링해야 함(사전 생성 자산이 없음) — 설계 원칙("신규 렌더링 지양")과 문면상 충돌 가능 | §3.1에서 "무엇을 그릴지는 R0가 이미 정했고 R3는 기존 렌더 코드를 top-K/bottom-K에 한정해 실행할 뿐"이라는 해석을 명시적으로 남기고, top_k/bottom_k 값으로 렌더링 시간을 제어 가능함을 단계 8에서 실측 |
| `ssat.metrics.viz.heatmap`이 이번에 처음 패키지 밖에서(R3에서) 호출되어, 반복 호출 시 matplotlib figure 누수 등 지금까지 드러나지 않은 문제가 생길 수 있음 | 단계 4·5에 명시적인 "반복 호출 후 열린 figure 0개로 수렴" 회귀 테스트를 포함(§1 격차#6) |
| Jinja2 신규 의존성이 대조군·안정성 계획의 "신규 의존성 없음" 선례를 깬다는 점이 향후 리뷰에서 문제 제기될 수 있음 | §2.1에 근거(설계서 자신의 제안, 순수 파이썬·오프라인, `string.Template`로는 §4.2의 7단 구성을 유지보수하기 어려움)를 미리 명시해 둠 — 결정을 되돌릴 필요가 생기면 `report/html_renderer.py` 한 파일의 교체만으로 가능하도록 다른 모듈은 Jinja2를 몰라도 되게 격리(§3.3 의존 방향 규칙) |
| RegionRow 최악값 정책이 "region 대부분이 안전한데 샘플 하나가 UNRELIABLE이어도 전체가 UNRELIABLE로 보임"으로 과도하게 비관적일 수 있음 | 의도된 보수적 선택(설계 원칙 "오해 방지"와 일치)이며, `reliability_distribution` 서브필드로 실제 분포를 함께 노출해 완화(§1 격차#3). 단계 8에서 이 필드가 실제로 유의미한 분포를 담고 있는지 확인하는 것이 성공 조건의 일부 |
| `ssat.metrics.viz._shared.open_image_source`가 `ImageFolderSource`에 하드코딩되어 있어 비디오 소스(`video_folder`) run에서는 R3가 애초에 자산을 만들 수 없음 | 이번 계획 범위에서 고치지 않는다(DebugViz 자체의 기존 한계를 그대로 상속) — R3는 이 경우 "원본 이미지 없음"으로 갤러리를 축소하고, 비디오 소스 지원은 DebugViz 쪽 후속 과제로 남긴다(§5 단계 5) |
| 범위 확대 | 인터랙티브 대시보드, 클러스터링·slice discovery, 검출 태스크 실렌더링, 단일 파일(base64) 배포는 v1에서 손대지 않음(설계서 §0 제외 범위, §5 단계 1의 검출 어댑터 스텁도 골격만) |
