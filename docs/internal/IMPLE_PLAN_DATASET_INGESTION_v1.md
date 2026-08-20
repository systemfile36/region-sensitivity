# 구현 계획서 (v1 데이터셋 수집(ingestion) 확장)
## Spatial Sensitivity Audit Toolkit — Source Provider Registry, Skeleton BBox 생성 유틸, 공개 데이터셋 레시피

> 본 계획서는 세션 중 논의된 다음 문제 진단을 구현 계획으로 구체화한 것이다: 저장소의 `data/` 아래
> 실제 NTU-RGB+D 데이터가 있음에도 바로 end-to-end로 감사를 돌릴 수 없고, 원본 파일을 SSAT가 원하는
> 형태(특히 skeleton bbox 사전 계산 자산)로 바꾸는 오프라인 전처리를 매번 새로 자가 구현해야 한다는
> 불편함이 있다. 이는 NTU-RGB+D 한정 문제가 아니라 "사용자 자신의 모델·데이터셋을 감사한다"는 이
> 프로그램의 목적상 구조적으로 반복되는 문제다. 코드베이스를 근거로 확인한 결론은, 모델 어댑터
> 계층(`ssat/core/adapter/provider.py`)에는 이미 열린 `AdapterProviderRegistry` + `CallableAdapter`
> 탈출구가 있는 반면, 데이터 소스 계층에는 동일한 확장 지점이 없다는 것이다 — 그리고 이 비대칭은
> 이미 한 번 이 저장소 스스로 진단해 두고 아직 닫지 않은 것이다
> ([VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md](VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md) §5.3 리스크#2:
> "Application/CLI 계층의 DI 공백 — 지금은 `sample_region_provider`나 커스텀 `SampleSource`를
> YAML만으로 주입할 경로가 없다").
>
> 전제: 코어([IMPL_PLAN_CORE_v1.md](IMPL_PLAN_CORE_v1.md)), 비디오·skeleton 부위별 occlusion 확장
> ([VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md](VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md), 구현 완료),
> 의미적 영역 취약도 프로파일링
> ([IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md](IMPLE_PLAN_SEMANTIC_VULNERABILITY_v1.md), 구현 완료)가
> 모두 구현되어 있다.
> 패키지 위치: `ssat/core/source/*`(신규 provider 레지스트리), `ssat/core/region/*`(신규 skeleton
> bbox 생성 유틸), `ssat/application/*`(배선), `scripts/dataset_prep/*`(신규, 패키지 밖 — NTU-RGB+D
> 레시피). `ssat/core/adapter/*`(어댑터 레지스트리, frozen 선례로만 참조)와
> `ssat/core/region/skeleton_store.py`(런타임 로더, 계약 불변)는 건드리지 않는다.

---

## 0. 결론 요약

| 우선순위 | 무엇을 | 왜 | 이번 계획의 범위 |
|---|---|---|---|
| 1 | `ssat/core/source`에 어댑터와 대칭인 `SourceProviderRegistry` 추가 | NTU 한정 문제가 아닌 일반 인프라 격차. 오늘 데이터 소스는 YAML/Python 어느 경로로도 등록된 2종(`image_manifest`/`video_manifest`) 밖으로 확장할 방법이 없다 | 순수 리팩터(기존 2종을 provider로 이전) + 확장 지점 개방. 새 provider를 이번 계획에서 등록하지는 않는다 |
| 2 | 관절 좌표 → 부위별 bbox JSON 생성 유틸(`skeleton_bbox_builder.py`) | `ssat/core/region/skeleton_store.py`가 스스로 "an offline skeleton-to-bbox conversion (not implemented in this package)"라고 명시한 바로 그 공백. NTU-25든 COCO-17이든 관절 인덱스 테이블만 바꾸면 재사용되는 진짜 범용 로직이라 자가구현에 맡기기 아깝다 | `ssat` 패키지 안의 순수 함수 라이브러리로 제공. 특정 데이터셋 파일 포맷 파싱은 포함하지 않는다 |
| 3 | `scripts/dataset_prep/ntu_rgb_d.py` — NTU-RGB+D 레시피 스크립트 | 원본 파일 레이아웃 파싱은 데이터셋마다 근본적으로 다르므로 일반화 불가. 다만 NTU-RGB+D는 이 저장소가 이미 두 번(설계 분석 문서 + `docs/ntu_rgb_d.py`) 다룬 대표 사례라 검증된 예시 하나는 제공할 가치가 있다 | `docs/ntu_rgb_d.py`의 원본/skeleton 파싱 로직 + 우선순위2 유틸을 이어 붙여 SSAT가 실제로 소비하는 `video_manifest.json`/`skeleton_bbox.json`을 산출. **패키지 밖(`scripts/`), 안정 API 아님, 참고 구현으로 명시** |

우선순위 1과 (2, 3)은 서로 독립적이다 — 3번 스크립트의 산출물은 오늘 이미 동작하는 `source.kind:
video_manifest` + `skeleton_source.bbox_data` 파일 경로 계약을 그대로 쓰므로, 1번 레지스트리가 없어도
NTU-RGB+D 감사는 끝까지 돌아간다. 1번은 "이 스크립트를 매번 별도 실행하지 않고 YAML 설정 하나로
데이터셋을 통째로 등록하는" 미래 확장(예: `source.kind: ntu_rgb_d`)을 가능하게 하는 인프라이며, 이번
계획서는 그 인프라만 놓고 실제 데이터셋 provider 등록은 v1.1 이후로 미룬다(§8).

---

## 1. 현재 구현 상태 대비 격차

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| 1 | `source` 계층에 어댑터와 대칭인 확장 지점이 없다 | **격차 — 우선순위 1** | [`ssat/application/config.py:122-138`](../ssat/application/config.py)의 `_SOURCE_CONFIG_MODELS`는 `{"image_manifest": ..., "video_manifest": ...}` 고정 dict이고, `_load_source()`가 `isinstance` 분기로 직접 `ImageFolderSource`/`VideoFolderSource`를 생성한다. `register()` 같은 등록 지점이 없다. 반면 어댑터 쪽은 [`AdapterProviderRegistry.register()`](../ssat/core/adapter/provider.py#L254)가 실제로 열려 있고, `AuditApplication.__init__(adapter_registry: AdapterProviderRegistry \| None = None, ...)`(`ssat/application/application.py:169-177`)로 호출자가 자기 provider를 등록한 레지스트리를 주입할 수 있다. |
| 2 | Python API로도 매니페스트 파일을 우회할 수 없다 | 격차 아님(의도된 설계로 확정, §1 항목4 참고) | `load_application_config(config: str \| Path \| Mapping, ...)`가 dict를 받아도 `source.manifest`는 `pydantic`이 실제 파일 `Path`로 강제 검증한다(`_load_manifest_samples`). 모델 쪽 `CallableAdapter`처럼 이미 만들어진 객체를 그냥 넘기는 경로가 없는 것은 사실이지만, 이는 재현성 계약(§1 항목4)과 직결되므로 "격차"가 아니라 "지켜야 할 제약"으로 재분류한다. |
| 3 | Skeleton bbox 생성 로직이 SSAT 어디에도 없다 | **격차 — 우선순위 2** | [`ssat/core/region/skeleton_store.py`](../ssat/core/region/skeleton_store.py)의 모듈 docstring이 스스로 명시: *"The store holds the output of an offline skeleton-to-bbox conversion (**not implemented in this package**)... Runtime components only read this pre-computed data; no skeleton/joint parsing happens here."* `docs/CONFIG_REFERENCE.md:91-93`도 동일하게 "SSAT 범위 밖의 오프라인 전처리"라고 확정해 둔 지점이다. 즉 이 계획서가 새로 여는 것이 아니라, 기존에 명시적으로 "범위 밖"이라 선언된 것을 이번에 채우기로 결정하는 것이다. |
| 4 | 대표 공개 데이터셋(NTU-RGB+D)에 대한 검증된 end-to-end 예시가 없다 | **격차 — 우선순위 3** | `docs/ntu_rgb_d.py`는 이 저장소에 이미 있지만 (a) `docs/` 아래에 있어 실행 가능한 도구로 취급되지 않고, (b) SSAT가 실제로 읽는 `video_manifest.json`/`skeleton_bbox.json` 포맷이 아니라 참고 프로젝트 자신의 parquet 다중 파일 포맷을 산출하며, (c) `import src.file_io`처럼 참고 프로젝트 전용 내부 모듈에 의존해 이 저장소에서 바로 실행할 수 없다. |
| 5 | `SourceProvenance`가 항상 실제 파일 + SHA-256을 요구한다 | 격차 아님 — 의도적으로 유지할 제약 | [`ssat/core/config/schema.py:179-190`](../ssat/core/config/schema.py)의 `SourceProvenance.manifest: Path`/`manifest_hash: str`(64자 hex 필수)는 "이 실행이 정확히 어떤 데이터로 감사됐는지" 재현성을 보장하는 핵심 계약이다. 이번 계획은 이 계약을 완화하지 않는다 — in-memory 객체를 해시 없이 직접 주입하는 경로는 명시적으로 범위 밖에 둔다(§8). |

---

## 2. 기술 스택과 의존성 방침

**신규 하드 의존성 없음.** 우선순위 3의 NTU 스크립트가 쓰는 `pandas`/`numpy`는 이미
[`requirements.txt`](../requirements.txt)의 최상위 의존성이라(extras 아님) 무게가 늘지 않는다.
우선순위 1/2는 표준 라이브러리 + 이미 있는 `pydantic`/`numpy`만으로 충분하다.

**위치 원칙**: `ssat/core`의 여러 모듈은 "제네릭 계약, 특정 데이터셋 지식 없음"을 AST 테스트로까지
강제하는 관례가 있다(예: `ssat/report/types.py`, `ssat/report/html_renderer.py`의 의존성 경계
테스트). 우선순위 1(레지스트리 골격)과 2(관절→bbox 라스터화 유틸)는 데이터셋에 무관한 진짜 범용
로직이므로 `ssat/core` 안에 정식 라이브러리 코드로 둔다. 반면 우선순위 3(NTU-RGB+D 원본 파일 파싱)은
데이터셋 하나에 결속된 지식이라 `scripts/`(이 저장소가 이미 디버그·수동 확인 도구에 쓰는 자리,
예: `scripts/run_debug_viz.py`, `scripts/run_skeleton_mask_debug.py`) 아래 두고 `ssat` 패키지의
일부로 import되지 않게 한다 — "지원 데이터셋을 계속 늘리겠다"는 API 약속이 아니라 참고 구현임을
위치로도 드러낸다.

---

## 3. 데이터 모델과 계산 위치

### 3.1 우선순위 1 — `ssat/core/source/provider.py`(신규): `SourceProviderRegistry`

`ssat/core/adapter/provider.py`의 `AdapterProvider`/`AdapterProviderRegistry` 패턴을 그대로
미러링한다(§2 위치 원칙 — 이미 검증된 이 저장소의 확장 패턴을 재사용, 새 패턴을 발명하지 않음).

**작업.**
1. `SourceProviderConfig(BaseModel)` — `model_config = ConfigDict(extra="forbid", frozen=True)`,
   `kind: str`(어댑터 쪽 `ProviderConfig.provider`와 대응하되 필드명은 기존 YAML 계약을 유지하기
   위해 `kind`로 둔다 — `source.kind: ...`는 이미 사용자가 쓰고 있는 키이므로 바꾸지 않는다).
2. 기존 `ssat/application/config.py`의 `ImageManifestSourceConfig`/`VideoManifestSourceConfig`를
   그대로(필드 불변) `ssat/core/source/provider.py`로 옮기고 `SourceProviderConfig`를 상속시킨다.
   `_ManifestSample`/`_SampleManifest`/`_load_manifest_samples`도 함께 옮긴다 — 동작 변경 없는
   순수 이동.
3. `SourceProvider(ABC)`:
   ```python
   class SourceProvider(ABC):
       name: ClassVar[str]
       config_model: ClassVar[type[SourceProviderConfig]]

       @abstractmethod
       def build(
           self, config: SourceProviderConfig, *, base_dir: Path
       ) -> tuple[SampleSource, SourceProvenance]:
           """Build one sample source + its provenance record."""
   ```
   `SourceProvenance`는 그대로(§1 항목5) — 모든 provider는 반드시 실제 파일을 가리키는
   `manifest`/`manifest_hash`를 채워야 한다. 이번 계획은 이 반환 튜플의 두 번째 항을 없애지 않는다.
4. `ImageManifestProvider`/`VideoManifestProvider` — 기존 `_load_source()`의 두 분기를 그대로
   옮긴 것. `name = "image_manifest"` / `"video_manifest"`.
5. `SourceProviderRegistry` — `register()`/`parse()`/`build()`를 `AdapterProviderRegistry`와
   동일한 오류 처리 관례로 구현. `parse()`는 기존 `_parse_source_config`의 `kind` 생략 시
   `"image_manifest"` 기본값 동작을 그대로 보존한다(하위 호환 — 기존 YAML에 `source.kind`가 없는
   설정이 있다면 계속 동작해야 함).
6. `default_source_provider_registry()` — `ImageManifestProvider()`/`VideoManifestProvider()` 2개만
   등록. 이번 계획에서 새 provider는 등록하지 않는다(§0).

**테스트(`tests/unit/test_source_provider.py`, 신규).**
- `SourceProviderRegistry.register()`가 중복 이름/빈 이름/`SourceProvider`가 아닌 값을 거부.
- `parse()`가 `kind` 생략 시 `"image_manifest"`로 폴백(기존 동작 보존 확인), 알 수 없는 `kind`에
  등록된 이름 목록을 포함한 에러.
- `default_source_provider_registry().build(...)`가 기존 `ImageFolderSource`/`VideoFolderSource`와
  100% 동일한 `SampleSource`/`SourceProvenance`를 만드는지(기존 `_load_source` 단위 테스트가 있었다면
  그 케이스를 그대로 이식, 없었다면 `tests/fixtures/synthetic_classification`/`synthetic_video`
  매니페스트로 신규 작성).

**성공 조건.** 위 테스트 통과 + 이 신규 파일은 아직 아무 기존 코드에서도 import되지 않으므로
전체 회귀 스위트에 영향 없음(단계0의 정의).

### 3.2 `ssat/application/config.py` 리팩터 — 레지스트리로 교체

**작업.**
- `_SOURCE_CONFIG_MODELS`/`_parse_source_config`/`_load_source`/`_load_manifest_samples`/
  `_ManifestSample`/`_SampleManifest`/`ImageManifestSourceConfig`/`VideoManifestSourceConfig`를
  삭제하고 `ssat.core.source.provider`의 대응물을 import해서 쓴다.
- `load_application_config(config, adapter_registry, *, source_registry=None, base_dir=None)` —
  새 키워드 인자, `None`이면 `default_source_provider_registry()`로 폴백(어댑터 쪽
  `AuditApplication.__init__`이 `adapter_registry or default_adapter_provider_registry()`로 하는
  것과 동일한 관례). 내부적으로 `source_config = source_registry.parse(source_raw)`,
  `source, provenance = source_registry.build(source_config, base_dir=resolved_base)`로 교체.
- `LoadedApplicationConfig.sample_source`/`source_provenance` 필드는 불변(반환 타입 계약 유지).

**테스트.** 신규 테스트를 추가하지 않고 **기존 전체 회귀 스위트를 그대로 통과**시키는 것 자체가
이 단계의 테스트다 — `image_manifest`/`video_manifest`를 쓰는 기존 통합 테스트
(`tests/integration/test_application_api.py`, `test_cli.py`, `test_video_source_e2e.py` 등)가
전부 새 경로를 타면서도 결과가 바뀌지 않아야 한다.

**성공 조건.** 회귀 0건. `AuditApplication`이 아직 새 `source_registry` 인자를 받지 않으므로
(다음 단계) 실제 동작 경로는 여전히 기본 레지스트리 하나뿐이라 위험이 낮다.

### 3.3 `ssat/application/application.py` 배선 — `source_registry` 주입 지점

**작업.**
- `AuditApplication.__init__(self, adapter_registry: AdapterProviderRegistry | None = None, *,
  source_registry: SourceProviderRegistry | None = None, code_version: str = CODE_VERSION)` —
  `self._source_registry = source_registry or default_source_provider_registry()`.
- `_build_context()`의 `load_application_config(config, self._registry, base_dir=base_dir)` 호출에
  `source_registry=self._source_registry` 추가.

**테스트.** `tests/integration/test_application_api.py`에 커스텀 `SourceProvider`를 등록한
`AuditApplication(source_registry=...)`로 `ssat estimate`/`ssat run`을 실제로 돌리는 테스트 1개
추가(예: 테스트 전용 `EchoManifestProvider`가 `kind: test_echo`를 받아 이미 메모리에 있는 고정
`SampleMeta` 목록으로 `ImageFolderSource`를 만들되, provenance는 여전히 테스트 픽스처의 실제 JSON
파일을 가리키게 해 §1 항목5 제약을 지키는지 확인) — 이 테스트가 "레지스트리가 실제로 열려 있다"는
것의 유일한 실증적 증거가 된다.

**성공 조건.** 신규 테스트 통과 + 기존 전체 스위트 회귀 0건. `AuditApplication()`을 인자 없이
생성하는 기존 모든 호출부는 동작이 완전히 동일해야 한다(어댑터 쪽 `adapter_registry` 인자가
지금까지 그래왔던 것과 동일한 하위 호환 보장).

### 3.4 우선순위 2 — `ssat/core/region/skeleton_bbox_builder.py`(신규): 관절 → 부위별 bbox

`ssat/core/region/skeleton_store.py`가 요구하는 정확한 스키마(§1 항목3 인용)를 만드는 쪽. 이
모듈은 "어떤 관절 세트인지"(NTU-25, COCO-17 등)를 모른다 — `body_parts` 딕셔너리를 인자로 받을
뿐이라, 데이터셋에 무관하게 재사용된다는 점이 이 유틸을 `scripts/`가 아니라 `ssat/core`에 두는
근거다.

**작업.**
```python
FrameBBox: TypeAlias = tuple[float, float, float, float]  # skeleton_store.py와 동일 별칭 재사용

def joints_to_part_bboxes(
    joints_xy: NDArray[np.floating],   # (T, J, 2), 픽셀 좌표
    joint_valid: NDArray[np.bool_],    # (T, J)
    body_parts: Mapping[str, Sequence[int]],  # part_name -> joint index 목록
    *,
    frame_size: tuple[int, int],       # (width, height), clamp 기준
    margin_ratio: float = 0.15,
    min_valid_joints: int = 1,
) -> dict[str, list[FrameBBox | None]]:
    """부위별 관절의 프레임별 union bbox를 margin 적용 + clamp해서 계산한다.

    한 프레임에서 part의 관절 중 valid=True인 것이 min_valid_joints 미만이면
    그 프레임은 None(추적 실패로 표시) — skeleton_store.py의 null 프레임과
    같은 의미. bbox는 관절 min/max에 margin_ratio만큼 각 변을 확장한 뒤
    [0, frame_size]로 clamp한다.
    """

def write_skeleton_bbox_json(
    sample_bboxes: Mapping[str, Mapping[str, Sequence[FrameBBox | None]]],  # sample_id -> part -> frames
    frame_sizes: Mapping[str, tuple[int, int]],                             # sample_id -> (w, h)
    output_path: Path,
) -> str:
    """skeleton_store.py가 읽는 정확한 JSON을 쓰고, 쓴 파일을
    load_skeleton_bbox_store()로 즉시 왕복 검증한 뒤 SHA-256을 반환한다
    (write 시점에 스키마 오류를 바로 드러냄 — 조용한 손상 파일 방지).
    """
```

**테스트(`tests/unit/test_skeleton_bbox_builder.py`, 신규).**
- 합성 `joints_xy`/`joint_valid` 배열로 margin/clamp를 손계산 대조(예: 관절 2개, margin_ratio=0.1,
  프레임 경계에 걸치는 케이스로 clamp가 실제로 동작하는지).
- valid joint 수가 `min_valid_joints` 미만인 프레임이 `None`으로 나오는지.
- `write_skeleton_bbox_json()`이 쓴 파일을 `ssat.core.region.skeleton_store.load_skeleton_bbox_store()`
  로 다시 읽어 원본 배열과 일치하는지(왕복 검증 — 이 유틸의 유일한 소비자가 스키마를 어긴 파일을
  만들지 않는다는 것을 이 저장소의 실제 로더로 증명).
- `body_parts`에 다른 관절 세트(예: 5-joint 장난감 세트)를 넣어도 동작해 NTU-25 전용이 아님을 확인.

**성공 조건.** 위 테스트 통과. 이 모듈은 `ssat/core/region/skeleton_store.py`를 테스트에서만
import하고(왕복 검증용) 런타임 감사 경로(`resolver.py`, `mask_generators.py` 등) 어디에서도 이
모듈을 import하지 않는다 — 오프라인 생성 전용이라는 스키마 경계를 정적 테스트로 강제
(`tests/unit/test_skeleton_bbox_builder.py`에 AST 기반 "런타임 모듈에서 import 안 됨" 테스트 추가:
`ssat/core/region/resolver.py`, `mask_generators.py`, `mask_base.py`, `skeleton_provider.py`가
`skeleton_bbox_builder`를 import하지 않는지 확인).

### 3.5 우선순위 3 — `scripts/dataset_prep/ntu_rgb_d.py`(신규): NTU-RGB+D 레시피

`docs/ntu_rgb_d.py`에서 재사용 가능한 부분과 아닌 부분을 명확히 가른다.

**`docs/ntu_rgb_d.py`에서 그대로/거의 그대로 재사용:**
- `normalize_video_key`, `parse_ntu_rgb_name`, `collect_ntu_rgb_data`, `split_ntu_rgb_data`(파일명
  → `sample_id`/`gt_label`/split, `video_manifest.json`의 재료).
- `BODY_PARTS`(NTU-25 관절 인덱스 테이블 — §3.4 유틸의 `body_parts` 인자로 그대로 전달).
- `parse_ntu_skeleton_file`, `choose_primary_body_index`, `skeleton_to_primary_arrays`(`.skeleton`
  파일 → `joints3d`/`joints2d_color`/`joint_valid` — §3.4 유틸의 `joints_xy`/`joint_valid` 재료).

**가져오지 않는 것(참고 프로젝트 자신의 파이프라인 전용, SSAT 산출물과 무관):**
- `_build_skeleton_info_chunk`/`load_skeleton_metadata_to_memory`/`filter_invalid_skeleton_path_from_df`
  등 parquet 다중 파일(`skeleton_summary.parquet` 등) 기반 저장 로직 전체 — SSAT는 §3.4의 단일 JSON만
  읽는다.
- `import src.file_io as file_io` 의존 — `ssat.utils.io`의 기존 헬퍼(`load_json`/`sha256_file` 등)로
  치환.

**새로 작성할 것.**
- CLI 진입점: `python scripts/dataset_prep/ntu_rgb_d.py --rgb-root <dir> --skeleton-root <dir>
  --split xsub --num-frames 16 --out <dir>`.
- 내부 흐름: (1) `collect_ntu_rgb_data`+`split_ntu_rgb_data`로 샘플 목록 확보 → `video_manifest.json`
  작성(`docs/CONFIG_REFERENCE.md`의 `{"samples": [{"sample_id", "path", "gt_label"}]}` 형식). (2)
  각 샘플의 `.skeleton` 파일을 파싱해 `joints2d_color`/`joint_valid` 확보 → §3.4의
  `joints_to_part_bboxes`로 부위별 bbox 계산 → `write_skeleton_bbox_json()`으로
  `skeleton_bbox.json` 작성. **`frame_size`는 SSAT가 실제로 디코딩하는 크기와 정확히 일치해야
  한다는 `docs/CONFIG_REFERENCE.md:85-87`의 경고를 그대로 지켜, decord가 디코딩하는 원본 해상도를
  스크립트가 직접 확인(첫 프레임 1개만 열어 shape 확인)해 채운다 — `.skeleton` 파일의
  `depthX/depthY`(depth 카메라 해상도)가 아니라 RGB 비디오 실제 해상도를 써야 함을 명시.** (3) 위
  둘을 참조하는 예시 `configs/examples/ntu_rgb_d_quickstart.yaml` 생성(§5).
- 파일 상단에 명시적 경고 주석: "이 스크립트는 NTU-RGB+D 전용 참고 구현이며 SSAT의 안정 API가
  아닙니다. 다른 데이터셋에는 이 파일을 복사해 자신의 원본 포맷 파싱 부분만 새로 작성하세요."

**테스트.** 이 스크립트 자체는 pytest 대상이 아니다(다른 `scripts/*.py`와 동일한 관례 —
`run_debug_viz.py`도 수동 확인용). 대신:
- `docs/ntu_rgb_d.py`에서 가져온 순수 함수(`parse_ntu_rgb_name`, `parse_ntu_skeleton_file` 등)는
  스크립트 안에 인라인하지 않고 스크립트가 import하는 형태로 유지해, 원한다면 나중에 단위 테스트를
  붙일 수 있는 여지를 남긴다(이번 계획에서 필수는 아님 — §8).
- §3.6(단계6, 실데이터 검증)이 이 스크립트의 실질적 검증이다.

**성공 조건.** §3.6에서 저장소의 `data/` 아래 실제 NTU-RGB+D 서브셋에 대해 스크립트를 돌려
`ssat estimate`가 에러 없이 끝나는 것.

---

## 4. Application/CLI 배선 변화 요약

| 항목 | 이전 | 이후 |
|---|---|---|
| `AuditApplication.__init__` | `(adapter_registry=None, *, code_version=...)` | `(adapter_registry=None, *, source_registry=None, code_version=...)` |
| `load_application_config` | `(config, registry, *, base_dir=None)` | `(config, adapter_registry, *, source_registry=None, base_dir=None)` |
| `source.kind` 허용값 | `image_manifest`, `video_manifest` 고정 | 동일(이번 계획에서 신규 등록 없음) — 단, 호출자가 `AuditApplication(source_registry=...)`로 자신의 provider를 더할 수 있음 |
| CLI(`ssat run`/`estimate`) | 변경 없음 | 변경 없음(CLI는 항상 기본 레지스트리만 사용 — 커스텀 provider 등록은 Python API 전용, 어댑터 쪽도 CLI에 provider 등록 옵션이 없는 것과 동일한 비대칭을 그대로 유지) |
| `scripts/dataset_prep/ntu_rgb_d.py` | 없음 | 신규. `AuditApplication`을 전혀 거치지 않는 독립 CLI 스크립트 — `video_manifest.json`/`skeleton_bbox.json`만 만들고 끝. 이후 `ssat estimate`/`ssat run`은 평소처럼 YAML로 실행 |

---

## 5. 문서화 범위

- `docs/CONFIG_REFERENCE.md`: "Source" 절 끝에 "커스텀 source provider 등록"(Python API 전용) 소절
  추가 — `AuditApplication(source_registry=...)` 예시 코드.
- `README.md`: "지원 데이터셋 레시피" 절 신규 — `scripts/dataset_prep/ntu_rgb_d.py` 사용법과 함께,
  "이 스크립트들은 참고 구현이며 안정된 API가 아닙니다"라는 경고를 명시.
- `configs/examples/ntu_rgb_d_quickstart.yaml`: `video_quickstart.yaml`/`skeleton_quickstart.yaml`과
  같은 위상의 새 예시(단, `data/`의 실제 NTU 파일을 가리키므로 CI/기본 테스트 대상은 아니고 §3.6
  수동 검증용).

---

## 6. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행한다.

### 단계 0. `SourceProviderRegistry` 골격 (§3.1)
**작업.** `ssat/core/source/provider.py` 신규 작성 — 아직 어디서도 import되지 않음.
**테스트.** §3.1의 4개 케이스.
**성공 조건.** 신규 테스트 통과, 기존 전체 스위트 100% 불변(신규 파일이라 영향 자체가 없음).

### 단계 1. `application/config.py` 리팩터 (§3.2)
**작업.** 기존 소스 로딩 로직을 전부 삭제하고 단계0의 레지스트리로 교체.
**테스트.** 신규 테스트 없음 — 기존 전체 스위트가 그대로 통과하는 것 자체가 성공 조건.
**성공 조건.** 회귀 0건. 이 단계가 우선순위1 트랙의 병목(가장 조심해야 하는 지점)이다.

### 단계 2. `AuditApplication` 배선 (§3.3)
**작업.** `source_registry` 생성자 인자 + `_build_context` 배선.
**테스트.** 커스텀 provider 등록 후 실제 `estimate`/`run`이 동작하는 통합 테스트 1개.
**성공 조건.** 신규 테스트 통과 + 회귀 0건.

### 단계 3. 관절 → 부위별 bbox 유틸 (§3.4)
**작업/테스트/성공 조건.** §3.4 그대로.

### 단계 4. NTU-RGB+D 레시피 스크립트 (§3.5)
**작업.** §3.5 그대로 — 단계3의 유틸을 라이브러리로 소비.
**테스트.** 단계3의 왕복 검증이 이 스크립트가 만드는 실제 산출물의 정합성을 이미 보장한다.
**성공 조건.** 스크립트가 저장소의 `data/` 서브셋 하나에 대해 예외 없이 `video_manifest.json`/
`skeleton_bbox.json`/예시 YAML을 만들어낸다.

### 단계 5. 문서화 (§5)
**작업.** `CONFIG_REFERENCE.md`/`README.md`/예시 YAML.
**성공 조건.** 문서 갱신 완료, 기존 문서 링크 깨짐 없음.

### 단계 6. 실데이터 검증
`data/` 아래 실제 NTU-RGB+D 서브셋(전체가 아니어도 됨 — 클래스 몇 개, 샘플 수십 개 규모로 충분)에
대해 단계4 스크립트를 실제로 실행하고, 그 산출물로 `ssat estimate configs/examples/
ntu_rgb_d_quickstart.yaml`이 끝까지 성공하는지 확인한다. 가능하면 `ssat run`까지 소규모로 돌려
`skeleton_parts` region이 실제로 사람을 따라가며 가려지는지 `scripts/run_skeleton_mask_debug.py`로
육안 확인한다(이 저장소의 기존 관례 — 디버그/검증 작업엔 pytest assertion과 별개로 수동 확인용
스크립트를 함께 쓴다).

**성공 조건.** `ssat estimate`가 에러 없이 끝나고, 합리적인 `pending_clean_samples`/
`pending_perturbed_items` 수치가 나온다. `ssat run`(소규모)이 최소 1개 이상의 유효한 perturbed item을
기록한다.

---

## 7. 단계 간 의존과 병렬화

```
0 ──> 1 ──> 2                (우선순위 1 트랙)

3 ──> 4 ──> 5 ──> 6           (우선순위 2·3 트랙)
```

두 트랙은 완전히 독립적이라 병행 가능하다(§0). 우선순위1 트랙의 병목은 단계1(기존 동작 100% 보존이
관건), 우선순위2·3 트랙의 병목은 단계4(원본 파일 파싱이 실제 NTU-RGB+D 데이터의 지저분함과 처음
맞닥뜨리는 지점)다.

---

## 8. 잔여 결정 사항의 처리 시점

| 항목 | 결정 시점 | 결정 내용 |
|---|---|---|
| in-memory `SampleSource`를 해시 없이 직접 주입하는 경로 | 이 계획서 작성 시 확정 | 반려. `SourceProvenance`의 재현성 계약(§1 항목5)을 깨뜨리므로 이번 계획 범위에서 제외. 필요성이 실제로 확인되면(예: 매 실행마다 원본이 바뀌는 스트리밍 데이터) 별도 계획서에서 "provenance를 caller가 직접 책임지고 채운다"는 `CallableAdapter`식 계약으로 논의 |
| NTU-RGB+D를 `SourceProvider`로 등록해 `source.kind: ntu_rgb_d` YAML 한 줄로 쓰게 할지 | v1.1로 이월 | 이번 계획은 우선순위1(레지스트리)과 우선순위2·3(NTU 스크립트)을 의도적으로 분리했다(§0). 스크립트가 실데이터로 한 번 검증된 뒤(단계6), 그걸 provider로 승격할지는 별도 결정 — 승격하려면 "합성 매니페스트/bbox 파일을 어디에 캐시하고 그 경로를 provenance로 삼을지"라는 새 설계 질문이 생기므로 지금 함께 풀지 않는다 |
| CLI(`ssat run`/`estimate`)에서도 커스텀 provider를 등록할 방법을 열지 | 범위 밖으로 확정 | 어댑터 쪽도 CLI에 provider 등록 옵션이 없다(Python API 전용) — 같은 비대칭을 유지해 일관성을 지킨다. CLI에서 커스텀 provider가 필요해지면 어댑터·소스 양쪽을 함께 다루는 별도 계획서가 필요 |
| NTU-RGB+D 외 다른 공개 데이터셋(Kinetics, COCO 등) 레시피 추가 | 수요 기반, 이번 계획 범위 밖 | 우선순위2 유틸이 관절 세트에 무관하게 설계돼 있어(§3.4) 다음 skeleton 데이터셋은 `body_parts` 테이블만 새로 만들면 되지만, 비-skeleton 데이터셋(예: 순수 이미지/영상 분류)은 애초에 `image_manifest`/`video_manifest`만으로 이미 충분해 레시피가 불필요할 수 있다 — 실제 요청이 생기면 그때 판단 |
| 관절→bbox 라스터화를 bbox 대신 polyline/convex hull로 정교화할지 | 범위 밖으로 확정(기존 결정 재확인) | `docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md` §3.4가 이미 "v1 범위에서는 권장하지 않음"으로 결론 낸 사항을 그대로 유지 |

---

## 9. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| §3.2 리팩터(고정 dict → 레지스트리)가 기존 `image_manifest`/`video_manifest` 동작을 미묘하게 바꿀 위험 | 단계1의 성공 조건을 "신규 테스트 0개, 기존 전체 스위트 100% 통과"로 명시(§6) — 새 코드가 기존 코드와 정확히 같은 동작을 하는지가 유일한 관심사이므로 회귀 스위트 자체가 완전한 안전망 |
| `frame_size`가 실제 decord 디코딩 해상도와 어긋나 skeleton bbox가 잘못된 위치를 가리는 위험(조용한 오류) | §3.5에서 스크립트가 `.skeleton` 파일의 depth 해상도가 아니라 실제 RGB 비디오 프레임 1개를 열어 확인한 해상도를 쓰도록 명시 + `load_skeleton_bbox_store`/`write_skeleton_bbox_json`의 왕복 검증이 스키마 오류는 잡아내지만 "해상도가 그럴듯하게 틀린" 경우는 못 잡으므로, 단계6에서 `run_skeleton_mask_debug.py`로 실제 마스크가 사람 위치에 맞는지 육안 확인을 성공 조건에 포함 |
| `docs/ntu_rgb_d.py`의 `choose_primary_body_index` 휴리스틱(2인 이상 동작에서 부정확할 수 있음, 원본 주석에도 명시)이 그대로 이식됨 | 이 계획서는 원본 참고 프로젝트의 알려진 한계를 그대로 인정하고 넘어간다 — NTU-RGB+D의 2인 상호작용 클래스(A050-A060)에서는 부정확할 수 있음을 스크립트 문서 주석에 명시, 개선은 범위 밖(§8과 동일한 "수요 기반" 원칙) |
| 우선순위3 스크립트가 안정 API처럼 오해되어 다른 데이터셋에도 그대로 재사용하려는 시도가 생길 위험 | §3.5에서 파일 상단 경고 주석 + §5 README 절에 "참고 구현, 안정 API 아님" 명시를 성공 조건의 일부로 포함 |
| 우선순위1(레지스트리)이 실제로 아무 커스텀 provider도 등록하지 않은 채 남아 "쓰이지 않는 확장 지점"이 될 위험 | §3.3 통합 테스트가 실제로 커스텀 provider를 등록해 `estimate`/`run`까지 돌리므로, 이 인프라가 껍데기만 남지 않고 최소 1개의 실증 사례로 뒷받침된다 |

---

## 10. 참고: 검토한 주요 파일 목록

`ssat/core/adapter/provider.py`(레지스트리 선례), `ssat/core/source/{base,image_folder,video_folder,types}.py`,
`ssat/application/config.py`, `ssat/application/application.py`, `ssat/application/types.py`,
`ssat/core/config/schema.py`(`SourceProvenance`, `SkeletonSourceConfig`),
`ssat/core/region/skeleton_store.py`, `docs/CONFIG_REFERENCE.md`,
`docs/VIDEO_SKELETON_EXTENSION_ANALYSIS_v1.md`, `docs/ntu_rgb_d.py`, `requirements.txt`.
