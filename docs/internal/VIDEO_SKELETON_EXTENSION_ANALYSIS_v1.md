# 비디오 입력 및 Skeleton 기반 부위별 Occlusion 확장 분석 (v1)

> 이 문서는 코드 변경 없이 현재 SSAT 코드베이스를 검토하고, (1) NTU-RGB+D류
> 비전 기반 Action Recognition 데이터셋(비디오) 입력 지원과 (2) skeleton
> 데이터 기반 부위별(body-part) occlusion을 config로 구성 가능하게 만드는
> 확장이 어느 수준의 작업인지 분석한 결과다. 비교 참고용으로
> `/home/limdongha/task-aware-quality-diagnosis-for-action-recognition`
> (이하 "참고 프로젝트")의 skeleton/body-part masking 구현을 함께 검토했다.

## 0. 결론 요약

- 두 요구사항 모두 **코어의 대대적인 재작성이 아니라 "확장"으로 구현 가능**하다.
  SSAT의 core 설계 문서(`docs/CORE_DESIGN_v1.md`)는 이미 비디오와
  skeleton 기반 부위 occlusion을 명시적인 향후 확장 지점으로 예약해두었고,
  실제 코드에도 그 자리가 비어 있는 채로 남아 있다.
- 요구사항 1(비디오 입력)은 **거의 순수 추가(additive) 작업**이다. 소스 계층에
  video decoder를 추가하고, 비디오를 받는 모델 어댑터를 하나 등록하면 된다.
  core(`ssat/core/*`)는 건드릴 필요가 없다.
- 요구사항 2(skeleton 기반 부위별 가림)는 대부분 추가 작업이지만, **마스크가
  현재 "샘플당 프레임 공통 `(H, W)` 1장"으로 고정되어 있다는 점**이 유일하게
  진짜 core 계약을 건드려야 하는 지점이다. 이 확장 역시 설계 문서에 "비디오
  확장 시 `(T, H, W)`를 허용"이라고 못 박혀 있어, 예정된 작업을 이제
  구현하는 것에 가깝다.
- 두 기능을 합쳐도 대략 **1,000~1,800 LOC** (신규 코드 + 기존 코드 수정 +
  테스트 + 문서) 수준으로 추정되며, 약 15~25개 파일에 걸쳐 있다. 자세한
  내역은 5장 표를 참고.

---

## 1. 현재 코드가 이미 비디오/부위 확장을 예비해 둔 지점

분석 중 가장 중요하게 확인한 사실은, 이 기능들이 "새로 설계"해야 하는 것이
아니라 **원 설계자가 이미 자리를 만들어 두고 v1에서는 구현만 보류한 상태**라는
점이다. 근거는 다음과 같다.

### 1.1 소스 계약이 항상 `(T, H, W, C)`

`ssat/core/source/types.py:44-58`의 `LoadedSample.array`는 이미지든 비디오든
관계없이 `(T, H, W, C)` uint8 배열이어야 한다는 계약을 강제한다. 이미지 소스인
`ImageFolderSource`(`ssat/core/source/image_folder.py:69`)는 단지 `T=1`로
감싸서 반환할 뿐이다. `docs/CORE_DESIGN_v1.md`는 이 결정을 다음과 같이 명시한다.

> "항상 (T, H, W, C)이다. 이미지도 T=1로 감싼다. **비디오 확장 시 코어를
> 건드리지 않기 위한 결정**이다."

즉 `Perturbator`, `RegionResolver`, 어댑터 전처리 파이프라인 등 이미 존재하는
모든 core 컴포넌트는 애초에 `T > 1`을 염두에 두고 설계되어 있다.

- `Perturbator.apply`(`ssat/core/perturb/perturbator.py:40-80`)는 `array`를
  `(T,H,W,C)`로 검증하고 마스크를 `mask.shape != array.shape[1:3]`로
  검증할 뿐, `T`에 대한 제약이 없다.
- `BlurOperator`, `GaussianNoiseOperator`, `PatchShuffleOperator`
  (`ssat/core/perturb/operators.py`)는 모두 프레임(`for frame in array`)
  또는 배치 축 전체(`array.shape`)를 그대로 다루도록 이미 구현되어 있어
  `T > 1`에서도 그대로 동작한다.
- 전처리 파이프라인의 `ChannelsFirst`(`ssat/core/adapter/preprocessing.py:86-98,
  258-259`)는 `(B, T, C, H, W)`로 변환한다 — 이는 곧 비디오 모델(3D CNN,
  TSM 등)이 기대하는 표준 입력 레이아웃이다. 오직 이미지 분류기를 위한
  `SqueezeTime` 연산(`T=1`을 강제로 제거)만 image-only 어댑터에서 사용될 뿐,
  이 연산을 config에서 빼면 비디오용 `(B,T,C,H,W)` 텐서가 그대로 나온다.
- `compute_dataset_stats`(`ssat/core/config/stats.py:106-108`)는 `array`의
  마지막 축(채널)만 고정하고 나머지 모든 축(`T,H,W` 포함)을 `reshape(-1, C)`로
  평탄화하므로 비디오 입력에도 수정 없이 동작한다(다만 5.3절의 성능 주의 참고).

### 1.2 Region 계층에 `skeleton_parts`가 이미 예약된 kind로 존재

`ssat/core/types.py:17-25`의 `RegionKind` enum에는 `GRID`, `EXPLICIT`,
`RANDOM_AREA_MATCH`와 나란히 **`BBOX_PARTITION`, `SKELETON_PARTS`,
`GT_BBOX`가 이미 정의**되어 있다. 그리고:

- `ssat/core/plan/expansion_base.py:22-38`에 **`SampleRegionProvider`
  Protocol**이 이미 정의되어 있다. `expand(sample, family) -> Sequence[RegionSpec]`
  하나만 구현하면 되는 인터페이스로, "sample metadata(annotation)를 기반으로
  concrete region을 생성"하도록 정확히 설계되어 있다.
- `ssat/core/plan/region_expanders.py:160-227`의
  `SampleDependentRegionExpander`가 `SKELETON_PARTS`/`BBOX_PARTITION`/`GT_BBOX`
  요청을 받아 `RegionExpansionContext.sample_region_provider`로 위임하도록
  이미 구현되어 있다. 현재는 provider가 주입되지 않으면 (v1 기본 상태)
  `"region kind 'skeleton_parts' is not implemented"` 오류를 던지는
  placeholder 상태다.
- `docs/CORE_DESIGN_v1.md:335-343`에 다음과 같이 정확히 이 사용자 요구사항과
  같은 설계가 문서화되어 있다.

  > "`skeleton_parts`, `gt_bbox`, `bbox_partition`은 향후 확장용 kind로
  > 예약한다. `SampleMeta`에는 annotation 전문을 넣지 않고, 별도
  > `SampleRegionProvider`가 sample metadata와 resolved family를 받아
  > deterministic RegionSpec 목록을 제공한다. skeleton 정보·부위 정의·가릴
  > 부위 목록이나 GT bbox 목록은 provider의 구체 구현이 담당하며 픽셀
  > 로딩에는 의존하지 않는다."

- 마스크 형태에 대해서도 `docs/CORE_DESIGN_v1.md` "마스크 형태" 절에
  명시적으로 이렇게 적혀 있다.

  > "`(H, W)` bool. 시간 축은 v1에서 전 프레임 공통으로 브로드캐스트한다.
  > **비디오 확장 시 `(T, H, W)`를 허용하되, 코어는 브로드캐스트 규칙만
  > 알면 된다.**"

  즉 지금 이 문서가 분석하려는 두 확장은, 설계 시점부터 명시적으로 예정되어
  있던 작업이다.

### 1.3 실행 계층에 이미 존재하는 의존성 주입(DI) 지점

`RegionResolver`, `Perturbator`, `PlanBuilder`(및 그 내부 `RegionExpander`)는
모두 생성자에서 커스텀 구현체를 주입할 수 있다.

- `PlanBuilder.__init__`(`ssat/core/plan/builder.py:43-54`)은
  `region_expander: RegionExpander | None`을 받는다.
- `RegionExpander.__init__`(`ssat/core/plan/region_expander.py:32-37`)은
  `sample_region_provider`를 받는다.
- `run_audit`(`ssat/core/runtime/execution.py:108-116`)과
  `run_worker`류 헬퍼(`ssat/core/runtime/pipeline.py:156-186`)는 모두
  `region_resolver`/`perturbator`를 주입받아 워커까지 전달한다.
- `AdapterProviderRegistry`(`ssat/core/adapter/provider.py:159-224`)는
  `register(provider)`로 새 모델 provider(예: 비디오 모델)를 등록할 수 있고,
  `AuditApplication(adapter_registry=...)`(`ssat/application/application.py:133-143`)
  로 애플리케이션 계층까지 그대로 주입 가능하다.

**단, 한 가지 공백은 있다.** `AuditApplication._build_context`
(`ssat/application/application.py:459-497`)와
`ConfigResolver.__init__`(`ssat/core/config/resolver.py:53-92`)은 이 DI
지점들을 **CLI/YAML 설정 레벨까지는 노출하지 않고 내부에서 기본값으로
고정 생성**한다. 즉 지금 `ssat run config.yaml`처럼 CLI로 실행하는
경로에서는 `sample_region_provider`나 커스텀 `SampleSource`를 주입할 방법이
없다 — Python API(`AuditApplication`, `ConfigResolver`, `PlanBuilder`를
직접 코드로 호출)를 쓰거나, Application 계층에 새 config 필드
(예: `source.kind: video_manifest`, `regions[].kind: skeleton_parts`에 대응하는
`skeleton_provider` 섹션)를 추가해 배선해야 한다. 이는 5장 규모 산정에
포함했다.

---

## 2. 요구사항 1: 비디오 입력 지원

### 2.1 필요한 변경

| 컴포넌트 | 현재 상태 | 필요한 작업 | 성격 |
|---|---|---|---|
| `SampleSource` 구현체 | `ImageFolderSource`만 존재 (`ssat/core/source/image_folder.py`) | `VideoFolderSource` 신규 추가: OpenCV(`cv2.VideoCapture`, 이미 의존성에 있음)로 비디오를 디코딩해 `(T,H,W,C)` uint8로 반환. 프레임 샘플링 정책(균등 샘플링 N프레임, 고정 stride, 전체 프레임 등)을 파라미터화 | 순수 추가 |
| Manifest 스키마 | `source.kind: image_manifest` 하나만 인식 (`ssat/application/config.py:26-42`) | `source.kind: video_manifest` 추가, manifest sample에 `path`(비디오 파일) + 선택적 `num_frames`/`fps`/`start_frame` 필드 | 순수 추가 (기존 image_manifest는 그대로 유지) |
| 모델 어댑터 | `torchvision`/`timm` provider만 등록 (`ssat/core/adapter/provider.py`) — 둘 다 이미지 분류기 대상 | 비디오 모델(예: TSM, SlowFast, X3D, PoseC3D 등) provider 신규 등록. `AdapterProvider` 인터페이스만 구현하면 됨 (`ssat/core/adapter/provider.py:84-93`) | 순수 추가 |
| 전처리 파이프라인 | `SqueezeTime`이 이미지 어댑터 config에서만 사용됨 | 비디오 어댑터 config에서는 `SqueezeTime`을 빼고 `ChannelsFirst`까지만 사용 → `(B,T,C,H,W)` 그대로 모델에 전달 | 설정만 다르게, 코드 불필요 |
| `compute_dataset_stats` | 모든 프레임을 로드해 평균 계산 (`ssat/core/config/stats.py`) | 그대로 동작은 하지만, 긴 비디오 전체를 다 읽으면 느림 → 선택적으로 프레임 서브샘플링 옵션 추가 권장(필수는 아님) | 성능 개선(선택) |
| CLI/Application 배선 | `source.kind`가 `image_manifest`로 하드코딩 (`ssat/application/config.py:74`) | `ImageManifestSourceConfig`처럼 `VideoManifestSourceConfig`를 만들고 `_load_source`에서 kind 분기 추가 | 소규모 수정 |

### 2.2 core를 건드릴 필요가 없는 이유

`RegionResolver`, `Perturbator`, `PlanBuilder`는 모두 `array.shape`에서
`T`를 읽어 그대로 사용하도록(하드코딩된 `T=1` 가정 없이) 이미 작성되어 있다.
1.1절에서 확인했듯 유일하게 `T=1`을 강제하는 지점은 이미지 전용 전처리
연산 `SqueezeTime`뿐이며, 이는 비디오 어댑터 config에서 그 연산을 사용하지
않으면 그만이다. 따라서 "grid" 또는 "explicit" region으로 비디오 전체에
동일한 공간 마스크를 적용하는 것(부위별 occlusion이 아닌 단순 공간 occlusion)은
**지금 당장 VideoFolderSource + 비디오 어댑터만 추가해도 동작**한다.

### 2.3 참고 프로젝트와 비교

참고 프로젝트의 `src/data/ntu_rgb_d.py`, `src/online_pipeline/`은 별도의
`.skeleton` 파서, 프레임 샘플링(`collate_functions.py`), TSM/SlowFast 모델
래퍼를 직접 구현한 대형 파이프라인이다. SSAT는 이미 "framework-independent
소스/어댑터 계약"을 원칙으로 설계돼 있으므로, 참고 프로젝트의 코드를
그대로 옮기기보다는 **그 파이프라인이 하던 일(비디오 디코딩, 모델 forward)을
`SampleSource`/`AdapterProvider` 계약에 맞게 얇게 감싸는 어댑터 계층만
작성**하면 된다. 즉 참고 프로젝트는 "무엇을 만들어야 하는지"에 대한
훌륭한 참조 구현이지만, SSAT에 이식할 때는 그 구현 전체가 아니라 데이터
흐름(비디오 → `(T,H,W,C)` 프레임, `video_key`/`sample_id` 매핑)만 재사용하면
충분하다.

---

## 3. 요구사항 2: Skeleton 기반 부위별 Occlusion

### 3.1 참고 프로젝트에서 확인한 설계 패턴

참고 프로젝트의 구현(`src/data/skeleton_metadata_registry.py`,
`src/online_pipeline/corruption/operators/bodypart_mask.py`,
`scripts/build_skeleton_metadata.py`)은 사용자가 설명한 요구사항과 정확히
같은 구조다.

- **`sample_id`(참고 프로젝트에서는 `video_key`) → 프레임별 skeleton/부위
  정보 딕셔너리**: `skeleton_info_by_key: dict[str, dict]`
  (`skeleton_metadata_registry.py:7`). 값(value)에는
  `frame_bodypart_bbox: {부위명: [T개의 (x1,y1,x2,y2)]}`,
  `frame_bodypart_valid`, `image_width/height`, `video_num_frames` 등이 들어있다.
- **부위 정의는 관절 인덱스 그룹의 딕셔너리**: `BODY_PARTS: dict[str, list[int]]`
  (`ntu_rgb_d.py:34-47`, 예: `"left_arm": [4,5,6,7,21,22]`, NTU-RGB+D
  25-joint 기준).
- **부위별 프레임 단위 bbox는 사전에(오프라인) 계산**해 parquet으로 저장
  (`scripts/build_skeleton_metadata.py`가 `.skeleton` 파일을 읽어
  `frame_bodypart_bbox.parquet`/`segment_bodypart_bbox.parquet` 생성).
  런타임에는 원본 skeleton을 다시 파싱하지 않고 이 사전 계산 결과만 사용한다.
- **런타임 occlusion 연산자**(`BodyPartMaskCorruptor.apply`)는 `video_key`로
  해당 샘플의 정보를 찾고, 프레임 인덱스별로 bbox를 조회해 그 영역만
  `mask_color`로 채운다. `bbox_source: "frame" | "segment"`, `bbox_scale`,
  `skip_invalid` 같은 파라미터로 유연성을 준다(`spec.py:230-274`).

이 패턴은 그대로 SSAT의 `SampleRegionProvider` + `RegionMaskGenerator`
계약에 대응시킬 수 있다:

| 참고 프로젝트 | SSAT 대응 개념 |
|---|---|
| `skeleton_info_by_key` 딕셔너리 | 새 `SkeletonRegionProvider`(SSAT의 `SampleRegionProvider` 구현체)가 내부에 보관 |
| `BODY_PARTS` 상수 | config 또는 provider 초기화 인자로 넘기는 부위 정의(“일정한 포맷”) |
| `BodyPartMaskCorruptionSpec` | `RegionSpec.kind = RegionKind.SKELETON_PARTS`, `params`에 부위명/프레임별 bbox 등 |
| `BodyPartMaskCorruptor.apply` | 새 `SkeletonPartsMaskGenerator`(`RegionMaskGenerator` 구현체) |

### 3.2 필요한 신규 컴포넌트

1. **Skeleton 데이터 로더/저장소** — `sample_id -> 프레임별 부위 정보`
   딕셔너리를 만들고(참고 프로젝트의 `build_skeleton_metadata.py`와 유사한
   전처리 스크립트, 또는 그 결과 parquet/json을 그대로 로드), 메모리에 상주시켜
   region expansion·mask 생성 시 조회할 수 있게 한다.
2. **부위 정의 포맷** — “일정한 포맷”이라는 요구사항에 맞춰
   `{part_name: [joint_index, ...]}` 같은 JSON/YAML 스키마를 새로 정의하고
   `AuditConfig`에 최상위 필드(예: `skeleton.body_parts`, `skeleton.source`)로
   추가한다.
3. **`SkeletonRegionProvider`** (`SampleRegionProvider` 구현) — config에
   나열된 부위 목록을 받아 `sample_id`별로 `RegionSpec` 인스턴스를 생성한다.
   `RegionSpec.params`에는 (a) 프레임별 bbox 전체를 직접 담거나, (b) 외부
   skeleton 데이터 파일에 대한 참조(`ref`/`ref_hash`와 유사한 방식)만 담고
   런타임에 공유 저장소에서 조회하는 두 가지 방식이 있다. 결정론적
   해시(`ssat/core/plan/hashing.py`)가 `RegionSpec.params`를 통째로
   직렬화해 item ID를 만들기 때문에, 프레임 수가 많은 클립에서는 (a)가
   해시 payload와 dump manifest 크기를 불필요하게 키운다 — **(b)를
   권장**하며, `ExplicitMaskGenerator`가 이미 `ref_hash`로 외부 파일을
   참조·검증하는 것과 같은 패턴을 재사용할 수 있다(`mask_generators.py:159-237`).
4. **`SkeletonPartsMaskGenerator`** (`RegionMaskGenerator` 구현) — 조회한
   프레임별 bbox(또는 관절 좌표)를 실제 `(T,H,W)` boolean mask로 래스터화한다.

### 3.3 유일하게 core 계약을 넓혀야 하는 지점: 마스크의 시간 축

지금 마스크는 **샘플 1개당 1장의 `(H,W)`**로 고정되어 있고, 이 1장이 모든
프레임에 동일하게 broadcast된다(`Perturbator._validate_inputs`,
`ssat/core/perturb/perturbator.py:107-110`: `mask.shape != array.shape[1:3]`
검사). Grid/explicit처럼 "이미지 전체에 걸쳐 고정된 위치"를 가리는 경우는
문제없지만, **사람이 움직이는 비디오에서 특정 신체 부위를 계속 따라가며
가리는 것은 프레임마다 다른 마스크가 필요**하다. 이는 1.2절에서 인용한
설계 문서가 정확히 예견한 지점이다 — "비디오 확장 시 `(T,H,W)`를
허용하되, 코어는 브로드캐스트 규칙만 알면 된다."

이 계약 확장이 실제로 건드리는 지점:

- `RegionMaskGenerator.get_mask`(`ssat/core/region/mask_base.py:103-124`)
  반환 타입을 `(H,W)` 또는 `(T,H,W)`로 확장.
- `RegionResolver.resolve`/`_get_mask`(`ssat/core/region/resolver.py:65-146`)
  — 반환된 마스크가 `(H,W)`면 기존처럼 처리, `(T,H,W)`면 `T`가 원본
  샘플의 `T`와 일치하는지 검증하는 분기 추가.
- `Perturbator._validate_inputs`/`apply`
  (`ssat/core/perturb/perturbator.py:40-114`) — 마스크 shape 검사를
  `(H,W)` 또는 `(T,H,W)` 허용으로 완화.
- 모든 `PerturbationOperator.apply`(`ssat/core/perturb/operators.py`) —
  현재 `result[:, mask, :] = fill` 같은 인덱싱은 `mask`가 `(H,W)`일 때만
  유효한 numpy 브로드캐스팅이다. `(T,H,W)` 마스크에서는 프레임별로
  인덱싱하는 경로를 분기 추가해야 한다(4개 연산자: constant_fill, mean_fill,
  blur, gaussian_noise, patch_shuffle — 공용 헬퍼 `_apply_fill`/`composite`에
  일부 공통화되어 있어 파급은 제한적).
- `ChunkProcessor`(`ssat/core/runtime/processors.py:150-162`) — 여러 아이템의
  마스크를 `np.stack(masks)`로 쌓는 부분이 `(H,W)`를 가정하므로 `(T,H,W)`
  케이스를 다루도록(혹은 스택하지 않고 리스트로 유지하도록) 조정.
  `RegionMeta.intended_area_px/ratio` 계산(`resolver.py:99-106`)도
  "전체 프레임 합" 또는 "프레임 평균" 중 의미를 정의해야 한다.
- `DeclarativePreprocessor.transform_mask`/`transform_mask_geometry`
  (`ssat/core/adapter/preprocessing.py:131-135, 267-279`) — 현재
  `mask[None,None,:,:,None]`로 강제로 `(1,1,H,W,1)`을 만들어 리사이즈/크롭한다.
  `(T,H,W)` 입력을 프레임별로 같은 기하 연산에 통과시키도록 일반화 필요.
- 관련 단위 테스트 일체 (`tests/unit/test_region_mask_factory.py`,
  `test_region_resolver.py`, `test_perturbator.py`,
  `test_perturb_operator_factory.py`, `test_runtime_processors.py` 등).

**중요:** 이 확장은 기존 동작을 깨뜨리지 않는 **상위 호환(strict superset)**
으로 설계할 수 있다 — 기존 `(H,W)` 반환/검증 경로는 "모든 프레임에 동일하게
적용"이라는 특수 케이스로 그대로 유지하고, `(T,H,W)`는 새로 추가되는
분기이기 때문이다. 따라서 grid/explicit/random_area_match 등 기존 region
kind와 기존 config는 전혀 영향을 받지 않는다.

### 3.4 부위별 마스크 래스터화 방식 선택

참고 프로젝트는 bbox(사각형) 기반으로 부위를 가린다(`bodypart_mask.py:420-441`).
관절 좌표에 여백(`margin_ratio`)을 주고 union bbox를 계산하는 방식이다.
SSAT에 이식할 때도 동일하게 **bbox 래스터화가 가장 구현이 단순하고
참고 프로젝트와 결과를 비교하기도 쉬워 권장된다.** (관절을 잇는 스켈레톤
선분에 두께를 주는 polyline 마스크나 convex hull 방식은 더 정교하지만
난이도와 LOC이 늘어나므로 v1 범위에서는 권장하지 않음.)

---

## 4. 부위 정의를 "일정한 포맷"으로 관리하는 방법

사용자가 요청한 "일정한 포맷으로 Skeleton 데이터 및 부위 정의를 추가하면
config yaml에 따라 해당 부위를 가릴 수 있게" 하는 요구는 두 개의 분리된
데이터로 나누는 것이 SSAT의 기존 설계 원칙(설정 vs. 외부 참조 자원의 분리,
`docs/CONFIG_REFERENCE.md`가 보여주는 explicit mask의 `ref`/`ref_hash` 패턴)과
가장 잘 맞는다.

1. **부위 정의(스키마)**: `{part_name: [joint_index, ...]}` 형태의 작은
   JSON/YAML. 데이터셋(NTU-RGB+D 25-joint 등)마다 하나씩 존재하며, config
   YAML에서 region family 파라미터로 참조한다. 참고 프로젝트의 `BODY_PARTS`
   상수를 그대로 데이터 파일화하면 된다.
2. **프레임별 skeleton/부위 bbox 데이터**: `sample_id -> {part_name: [T개의
   bbox]}` 형태의 사전 계산된 자산(parquet 또는 json). `ExplicitMaskGenerator`가
   외부 mask 파일을 `ref`+`ref_hash`로 참조·검증하는 것과 동일한 원칙으로,
   이 자산도 경로 + content hash로 config에 등록해 재현성을 보장한다.

이렇게 분리하면 config YAML은 대략 다음과 같은 새 섹션 하나만 추가하면 된다
(예시이며 실제 스키마는 구현 시 확정):

```yaml
regions:
  - region_id: occlude_left_arm
    kind: skeleton_parts
    params:
      body_part: left_arm
      bbox_scale: 1.15

skeleton_source:
  body_part_defs: ../data/ntu_body_parts.json   # {part_name: [joint_idx,...]}
  frame_bbox_data: ../data/skeleton_frame_bbox.parquet  # sample_id -> part -> [T,4]
```

---

## 5. 종합 규모 분석

### 5.1 난이도 판단: 재작성이 아니라 확장

- **재작성이 필요한 부분: 없음.** 소스/어댑터/region/perturb 4개 계층
  모두 이미 `Protocol`/`ABC` 기반의 first-match dispatch 구조로 되어 있고
  (`SampleSource`, `AdapterProvider`, `RegionMaskGenerator`,
  `RegionFamilyExpander`, `PerturbationOperator` 모두 `supports()` +
  구현체 등록 패턴), 신규 kind를 추가하는 것은 기존 관례를 그대로 따르는
  작업이다.
- **유일한 "핵심 계약 변경": 마스크의 시간 축을 `(H,W)`에서 `(H,W)|(T,H,W)`로
  넓히는 것**(3.3절). 이것도 상위 호환이며, 이미 설계 문서에 명시된
  예정된 작업이다.
- 따라서 전체 작업 성격은 **"확장(extend)"**으로 분류하는 것이 정확하다.
  다만 3.3절 항목은 여러 파일에 걸쳐 있어 신중한 리뷰가 필요한 유일한
  "중간 위험" 구간이다.

### 5.2 LOC 추정表

아래는 신규 파일은 "신규", 기존 파일 수정은 "수정"으로 표시한 대략적인
추정치다(테스트/문서 포함, 참고용 근사치).

| 영역 | 파일(예시) | 신규/수정 | 추정 LOC |
|---|---|---|---|
| **요구사항 1: 비디오 입력** | | | **약 350~600** |
| `VideoFolderSource` + 관련 타입 | `ssat/core/source/video_folder.py` 등 | 신규 | 120~200 |
| `video_manifest` source config | `ssat/application/config.py` | 수정 | 30~60 |
| 비디오 모델 `AdapterProvider` 1종(예: TSM 래퍼) | `ssat/core/adapter/*_adapter.py` | 신규 | 100~200 |
| 단위/통합 테스트 | `tests/unit/test_video_folder_source.py` 등 | 신규 | 100~150 |
| 문서(`CONFIG_REFERENCE.md` 등) | `docs/*.md` | 수정 | ~50 |
| **요구사항 2: skeleton 부위별 occlusion** | | | **약 650~1200** |
| Skeleton 데이터 로더/저장소 | `ssat/core/region/skeleton_provider.py` 등 | 신규 | 150~250 |
| 부위 정의 스키마 + config 필드 | `ssat/core/config/schema.py` 등 | 수정 | 60~120 |
| `SkeletonPartsMaskGenerator` | `ssat/core/region/mask_generators.py`/신규 파일 | 신규 | 100~180 |
| **마스크 시간 축 확장**(3.3절 전 항목) | `mask_base.py`, `resolver.py`, `perturbator.py`, `operators.py`, `processors.py`, `preprocessing.py` | 수정 | 200~350 |
| Application/CLI 배선(YAML→provider) | `ssat/application/application.py`, `config.py` | 수정 | 80~150 |
| 단위/통합 테스트 | `tests/unit/test_region_mask_factory.py` 등 다수 확장 + 신규 | 신규+수정 | 150~250 |
| 전처리 스크립트(오프라인 skeleton→bbox 변환) | `scripts/build_skeleton_bbox_cache.py` | 신규 | 80~150 |
| 문서 | `docs/*.md` | 수정 | ~50 |
| **합계** | | | **약 1,000~1,800 LOC** |

(메모리에 저장된 프로젝트 관례에 따라, 실제 구현 시에는 `scripts/` 아래
수동 확인용 CLI 스크립트도 함께 추가하는 것이 이 리포지토리의 기존 패턴과
일치한다 — `scripts/run_debug_viz.py` 등 기존 예시 참고.)

### 5.3 리스크/주의사항

1. **마스크 시간 축 확장(3.3절)이 유일한 "중간 위험" 작업**이다. 여러
   계층(region/perturb/runtime/adapter)에 걸쳐 있고, `RegionMeta`의
   면적 통계처럼 의미를 새로 정의해야 하는 값이 있어 리뷰가 필요하다.
   다만 상위 호환으로 설계 가능하므로 기존 회귀 테스트가 안전망 역할을 한다.
2. **Application/CLI 계층의 DI 공백**(1.3절 마지막 문단) — 지금은
   `sample_region_provider`나 커스텀 `SampleSource`를 YAML만으로 주입할
   경로가 없다. 이 부분은 `AuditApplication`/`ConfigResolver`에 새 config
   섹션과 배선 코드를 추가해야 하며, 5.2절 표의 "Application/CLI 배선"
   항목이 이를 포함한다.
3. **결정론/해시 영향**(3.2절 3번) — `RegionSpec.params`가 곧 item ID
   해시의 입력이므로(`ssat/core/plan/hashing.py`), 프레임별 bbox를 params에
   직접 담으면 비디오 클립마다 다른 해시 payload 크기가 되어 대규모 실행 시
   비용이 커질 수 있다. 외부 참조(`ref`+`ref_hash`) 방식을 쓰는 것이 안전하다.
4. **성능**: `compute_dataset_stats`(2.1절)가 전체 비디오 프레임을 다 읽는
   구조라 대형 비디오셋에서 느릴 수 있다. 필수 변경은 아니지만 실사용
   전에 서브샘플링 옵션을 넣는 것을 권장한다.
5. **비디오 디코딩 라이브러리**: `opencv-python-headless`가 이미
   `requirements.txt`에 있어 `cv2.VideoCapture`로 바로 시작할 수 있다.
   다만 대량 비디오 처리 시 `decord`/`PyAV` 같은 전용 라이브러리가 더
   빠를 수 있어, 실제 데이터 규모에 따라 재검토를 권장한다(신규 의존성
   추가 여부는 별도 결정 필요).

### 5.4 권장 구현 순서

1. **1단계 (낮은 위험, 즉시 가치)** — `VideoFolderSource` + 비디오
   `AdapterProvider` 1종만 추가. 이 시점에 grid/explicit region으로
   "비디오 전체에 고정 위치 가리기" 감사가 이미 가능해진다(요구사항 1 단독
   충족, core 무변경).
2. **2단계** — 마스크 시간 축을 `(T,H,W)`까지 허용하도록 core 확장(3.3절),
   상위 호환 유지를 검증하는 회귀 테스트 통과 확인.
3. **3단계** — skeleton 데이터 로더 + `SkeletonRegionProvider` +
   `SkeletonPartsMaskGenerator` 추가, 참고 프로젝트의 bbox 래스터화 로직을
   SSAT 계약에 맞게 이식.
4. **4단계** — Application/CLI 계층에 config 섹션과 배선을 추가해
   YAML만으로 전체 파이프라인을 구동 가능하게 마무리.

---

## 6. 참고: 검토한 주요 파일 목록

- SSAT: `ssat/core/source/{base,types,image_folder}.py`,
  `ssat/core/region/{mask_base,mask_generators,mask_factory,resolver,types}.py`,
  `ssat/core/plan/{expansion_base,region_expander,region_expanders,builder}.py`,
  `ssat/core/perturb/{perturbator,operators}.py`,
  `ssat/core/adapter/{preprocessing,preprocessor,provider}.py`,
  `ssat/core/config/{schema,resolver,stats}.py`,
  `ssat/core/runtime/{processors,execution,pipeline}.py`,
  `ssat/application/{application,config}.py`,
  `docs/CORE_DESIGN_v1.md`, `docs/CONFIG_REFERENCE.md`.
- 참고 프로젝트: `src/data/{ntu_rgb_d,skeleton_metadata_registry}.py`,
  `src/online_pipeline/corruption/{spec,operators/bodypart_mask,operators/context}.py`,
  `scripts/build_skeleton_metadata.py`.
