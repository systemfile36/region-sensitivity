# 코어 설계 명세 (v1 잠정 확정본)
## Spatial Sensitivity Audit Toolkit — Core

---

## 0. 문서 범위

본 문서는 도구의 **코어** 설계를 기술한다. 코어의 범위는 다음과 같다.

> **명세 로드 → 실시간 교란 입력 생성 → raw dump 기록**

### 코어에 포함되는 것

작업 열거, 샘플 로딩, 마스크 실체화, 교란 적용, 배치 구성, 어댑터 호출, dump 기록, 재개 관리, 비용 추정.

### 코어에 포함되지 않는 것

지표 계산, 원본–교란 매칭, 마스크–GT 중첩 정책 적용, 집계, 리포트, 모델 구현 자체.

### 핵심 원칙

**코어는 판단하지 않는다.** "이 샘플이 취약하다"는 해석은 전부 후단의 몫이며, 코어는 "이 조건에서 이 출력이 나왔다"만 기록한다.

### v1 전제

- 태스크: **이미지 분류만**. 내부 표현은 비디오 확장을 고려해 `(T, H, W, C)`로 통일하고 이미지는 `T=1`로 취급
- 프레임워크 비의존: 코어는 MMAction2를 포함한 어떤 특정 프레임워크도 import하지 않음
- 단일 GPU 환경 가정. ModelAdapter는 메인 프로세스에서만 호출
- raw dump는 전체 로짓 벡터 저장
- `variants_per_chunk`는 고정 설정값 (자동 조정 없음)
- 동일 면적 무작위 대조군은 사용자가 명시적으로 요청할 때만 생성

---

## 1. 전체 흐름

```
┌─ 메인 프로세스 ────────────────────────────────────────────────┐
│                                                                │
│  Config ──> ConfigResolver ──> ResolvedConfig                  │
│                  │                                             │
│                  ├──> ModelAdapter.describe()  [결정론 검증]   │
│                  └──> DatasetStats            [사전 계산·고정] │
│                              │                                 │
│                              ▼                                 │
│                      PlanBuilder ──> WorkChunk 목록            │
│                              │                                 │
│                              ▼                                 │
│                      ResumeIndex ──> 미완료 청크만 필터        │
│                              │                                 │
│                              ├──> CostEstimator [사전 보고]    │
│                              ▼                                 │
│                      DataLoader(chunk_ids, num_workers=N)      │
│                              │                                 │
└──────────────────────────────┼─────────────────────────────────┘
                               │ IPC: chunk_id만 전달
┌─ 워커 프로세스 × N ──────────▼─────────────────────────────────┐
│                                                                │
│   chunk_id ──> PlanBuilder.materialize(chunk_id) ──> WorkChunk │
│                              │                                 │
│                              ▼                                 │
│                   SampleSource.load(sample_id)    [청크당 1회] │
│                              │                                 │
│                              ▼                                 │
│              for item in chunk.items:                          │
│                  RegionResolver.resolve(shape, spec) ──> mask  │
│                  Perturbator.apply(arr, mask, op, params, rng) │
│                              │                                 │
│                              ▼                                 │
│             PreparedChunk (chunk_id + arrays + masks + metas)  │
│                              │                                 │
└──────────────────────────────┼─────────────────────────────────┘
                               │ IPC: 배열 + 실제 적용 mask + 메타
┌─ 메인 프로세스 ──────────────▼─────────────────────────────────┐
│                                                                │
│                        Rebatcher                               │
│             (청크 경계 무시, 동적 batch cap으로 재구성)        │
│                              │                                 │
│                              ▼                                 │
│                   ModelAdapter.predict(batch)                  │
│                    ├─ 성공 ──────────────┐                     │
│                    └─ OOM ──> BatchSplitter ──> 재시도         │
│                                          │                     │
│                                          ▼                     │
│                                    DumpWriter                  │
│                          ┌───────────┼───────────┐             │
│                          ▼           ▼           ▼             │
│                   clean/         perturbed/  index/             │
│                   part_*        chunk_*     part_*             │
│                          └───────────┼───────────┘             │
│                                      ▼                         │
│                              RunManifest (JSON)                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. 모듈 명세

### M0. ConfigResolver

실행 전 모든 비결정 요소를 확정하는 관문. 여기서 실패하면 실행을 시작하지 않는다.

**입력.** 사용자 설정(YAML/dict), 어댑터 인스턴스, 데이터셋 참조

**하는 일.**

- 어댑터의 `describe()` 호출 → 결정론 여부 확인. 비결정론이면 기본 거부, `allow_nondeterministic: true`일 때만 경고 후 진행
- 데이터셋 통계 사전 계산(채널 평균 등). 이미 계산된 값이 설정에 있으면 그것을 사용
- 영역 명세와 교란 명세의 유효성 검증. Region family는 planning과
  같은 ordered `RegionFamilyExpander`, 교란은 runtime과 같은 ordered
  `PerturbationOperator`의 config hook을 사용
- 상대 경로 → 절대 경로, 기본값 채우기

**출력.** `ResolvedConfig` — 이후 어떤 모듈도 설정을 재해석하지 않는다. 이 객체 전문이 RunManifest에 기록된다.

**설계 의도.** "실행 중에 결정되는 값"을 0으로 만드는 것. 데이터셋 통계를 실행 중 계산하면 실행 순서에 결과가 의존하게 되는데, 그것을 여기서 차단한다.

---

### M1. PlanBuilder

작업 공간을 결정론적으로 열거한다. **두 개의 진입점**을 갖는 것이 특징이다.

#### 진입점 A — `enumerate()` (메인에서 1회)

`ResolvedConfig` → `WorkChunk` 메타 목록. 반환되는 것은 가벼운 식별자뿐이다.

```
WorkChunkMeta:
    chunk_id: str          # 결정론적
    sample_id: str
    item_ids: list[str]
    n_items: int
```

#### 진입점 B — `materialize(chunk_id)` (워커에서 호출)

`chunk_id` → 완전한 `WorkChunk` (각 아이템의 전체 명세 포함).

```
WorkItem:
    item_id: str           # 아래 필드 전부의 정규화 해시
    sample_id: str
    region_spec: RegionSpec # family에서 확장된 concrete region
    perturb_op: str
    perturb_params: dict
    invert_mask: bool
    seed_salt: int
    is_control: bool       # 동일 면적 대조군 여부
```

#### 왜 두 진입점인가

청크 메타만 IPC로 넘기면 직렬화 비용이 거의 0이다. 워커가 `chunk_id`로부터 전체 명세를 **재계산**한다. 열거가 결정론적이므로 메인과 워커가 같은 결과를 얻는다.

#### item_id 계산

위 필드들(item_id 자신 제외)을 정규화 직렬화한 뒤 해시.

정규화 규칙:
- 키 정렬
- 부동소수는 고정 자릿수 문자열
- None은 생략

이 규칙은 스키마 버전으로 고정하며, 바뀌면 버전을 올린다.

**item_id를 해시로 두는 것이 핵심 설계 결정이다.** 이것이 있으면 재개(이미 있는 id 건너뛰기), 중복 제거, 부분 재실행, 결과 병합이 전부 부수적으로 해결된다.

#### 청킹 규칙

한 샘플의 아이템을 `variants_per_chunk`(고정 설정값) 단위로 순서대로 분할. 마지막 청크는 짧을 수 있다.

#### region family 확장

설정의 `RegionConfig`와 `ResolvedRegionConfig`는 단일 마스크가 아니라 concrete
region을 열거하는 **region family**다. PlanBuilder는 픽셀을 로드하기 전에
`RegionExpander`를 호출하여 family를 ordered `RegionSpec` 목록으로 확장한다.
`RegionSpec.region_id`는 family ID를 유지하고, `region_instance_id`가 concrete
공간 단위를 식별한다.

grid는 `rows`, `cols`만 설정하며 row-major 순서로 모든 cell을 확장한다. 일반
WorkItem 순서는 sample → family → concrete region → perturbation → seed salt다.
따라서 샘플별 일반 아이템 수는 다음과 같다.

```
Σ(family의 concrete region 수 × Σ(perturbation의 seed 수))
```

#### clean 취급

clean은 별도로 열거한다. `enumerate_clean()`이 샘플당 하나의 clean 아이템을 반환하고, 이는 별도 테이블로 간다. 교란 설정이 바뀌어도 clean은 재사용된다.

#### 대조군

명시 요청 시에만 생성한다. 설정에 다음과 같은 항목이 있을 때만 해당 WorkItem이 열거된다.

```
controls: [{match_area_of: "...", n_samples: 3}]
```

`match_area_of`는 family ID를 참조한다. PlanBuilder는 family의 각 concrete target
region마다 `n_samples`개의 control을 만들고, target의 전체 concrete recipe를
control RegionSpec에 내장한다. `is_control` 플래그로 일반 항목과 구분한다.

---

### M2. ResumeIndex

**별도 인덱스 fragment dataset**으로 완료 상태를 관리한다.

**파일 구조.** `index/part_NNNNNNNN.parquet` — 컬럼은 `item_id`, `status`,
`chunk_file`, `written_at`. 같은 ordinal의 perturbed data fragment와 1:1로 대응한다.

**메인에서의 동작.**

- 시작 시 fragment를 ordinal 순으로 메모리에 로드 → 최신 `item_id → status` 매핑
- 청크 필터링: 청크의 모든 아이템이 인덱스에 있고 `retry_failed` 설정에 따라 재시도 대상이 아니면 건너뜀
- **필터 단위는 청크**이다. 부분 완료 청크는 통째로 재실행한다 (로드 비용을 낭비하지 않기 위해)

**쓰기 시점.** DumpWriter가 data fragment를 flush할 때마다 새 index fragment를
원자적으로 publish한다.

**쓰기 순서가 중요하다.** 데이터 chunk를 먼저 fsync하고, 그 다음 인덱스를 갱신한다. 반대면 크래시 시 "인덱스엔 있는데 데이터는 없는" 상태가 생긴다.

**크래시 복구.** 인덱스에 없는 데이터 레코드는 고아가 되며 기본 Reader와
ResumeIndex는 무시한다. 선택적으로 `--rebuild-index`로 모든 data fragment를
검증·스캔해 index fragment를 멱등적으로 재구축한다.

**규모 감각.** item_id 32바이트 + 상태 + 파일명이면 레코드당 약 100바이트. 288만 아이템이어도 300MB 이하이고, parquet 압축하면 훨씬 작다. 메모리 로드에 문제없다.

---

### M3. SampleSource

**인터페이스.**

```
list_samples() -> list[SampleMeta]
load(sample_id) -> LoadedSample | LoadError
```

**LoadedSample.**

```
array: np.ndarray        # (T, H, W, C) uint8, 이미지는 T=1
sample_id: str
original_shape: tuple
content_hash: str        # provenance용
```

**설계 원칙.**

- **항상 (T, H, W, C)**이다. 이미지도 T=1로 감싼다. 비디오 확장 시 코어를 건드리지 않기 위한 결정이다
- **uint8 원본 픽셀 공간**이다. 정규화나 리사이즈를 하지 않는다. 그것은 어댑터의 몫이다
- **예외를 던지지 않고 `LoadError`를 반환**한다. 워커에서 예외가 나면 DataLoader 전체가 흔들릴 수 있으므로, 실패를 값으로 다룬다

**v1 구현체.** 명시적인 `SampleMeta` 목록을 받는 `ImageFolderSource`를 먼저
제공한다. 정지 이미지는 Pillow로 RGB 변환한 뒤 `(1,H,W,3)` uint8로 반환하며
원본 파일 바이트의 SHA-256을 provenance로 기록한다. 폴더 자동 탐색과
데이터프레임/CSV 기반 `TabularSource`는 후속 source 확장으로 둔다.

---

### M4. RegionResolver

#### planning과 materialization의 분리

`RegionExpander`는 planning 계층에서 region family를 concrete `RegionSpec`들로
열거한다. `RegionResolver`는 그중 하나를 받아 정확히 하나의 bool mask로
materialize한다. RegionResolver에서 여러 mask를 반환하지 않으므로 WorkItem,
item ID, dump row의 1:1 계약이 유지된다.

두 계층 모두 facade에서 kind를 분기하지 않는다. `RegionFamilyExpander`는
`supports(family)`/`validate_config(family)`/`expand(sample, family)`, `RegionMaskGenerator`는
`supports(spec)`/`get_mask(height, width, spec, rng)` 계약을 구현한다. 각 factory가
fresh ordered strategy 목록을 만들고 처음 지원하는 구현체를 실행한다. 등록 순서는
custom 확장의 명시적인 override 우선순위다. ConfigResolver도 동일한
first-match 목록을 사용하므로 family의 설정 검증과 concrete expansion은 같은
구현체가 소유한다. explicit 경로·hash 확정은 외부 참조 resolution이므로
ConfigResolver에 남는다.

#### RegionSpec의 두 갈래

```
RegionSpec:
    region_id: str          # 설정의 family ID
    region_instance_id: str # concrete region ID
    kind: "grid" | "explicit" | "random_area_match"
          | "bbox_partition" | "skeleton_parts" | "gt_bbox"
    params: dict           # 절차적 생성 레시피
    ref: str | None        # explicit인 경우 마스크 파일 참조
    ref_hash: str | None
```

**절차적(procedural).** grid family는 planning 시 cell별 concrete recipe로
확장되고, 실행 시점에는 샘플 크기를 받아 해당 cell의 비트맵만 만든다. 저장하는
것은 recipe뿐이므로 비용이 사실상 0이다. 각 축의 cell 경계는
`start = index * size // divisions`, `end = (index + 1) * size // divisions`로
계산하여 겹침 없이 전체 픽셀을 덮는다.

**명시적(explicit).** 사용자 제공 단일 프레임·단일 채널 bitmap이다. 대상 이미지와
동일한 `(H,W)`만 허용하고 nonzero를 `True`로 해석하며 resize하지 않는다. 비트맵
파일 참조와 SHA-256을 저장하고 materialization 시에도 hash를 다시 검증한다.

#### 인터페이스

```
resolve(shape, spec, rng=None) -> (mask: bool ndarray, meta: RegionMeta)
```

**RegionMeta.**

```
intended_area_px: int
intended_area_ratio: float
generator_kind: str
generator_version: str
confidence: float | None   # 자동 생성기인 경우
```

#### 마스크 형태

`(H, W)` bool. 시간 축은 v1에서 전 프레임 공통으로 브로드캐스트한다. 비디오 확장 시 `(T, H, W)`를 허용하되, 코어는 브로드캐스트 규칙만 알면 된다.

#### random_area_match

내장된 concrete target recipe를 먼저 resolve한 뒤 전체 `(H,W)` 픽셀에서 target과
동일한 개수를 균등 비복원 추출한다. rng가 필요하므로 `resolve`에 rng를 넘기고,
그 rng는 item_id에서 유도된다. nested random target은 허용하지 않는다. 이로써
대조군도 완전히 재현 가능하다.

#### sample-dependent region 확장

`skeleton_parts`, `gt_bbox`, `bbox_partition`은 향후 확장용 kind로 예약한다.
`SampleMeta`에는 annotation 전문을 넣지 않고, 별도 `SampleRegionProvider`가
sample metadata와 resolved family를 받아 deterministic RegionSpec 목록을
제공한다. skeleton 정보·부위 정의·가릴 부위 목록이나 GT bbox 목록은 provider의
구체 구현이 담당하며 픽셀 로딩에는 의존하지 않는다. 현재 v1 코어에서는 이 세
kind를 명시적인 not-implemented 오류로 거부한다.

#### explicit 마스크 캐싱

워커별 128-entry LRU 캐시를 사용하며 키는 `ref_hash`다. cache hit에서도 현재 파일
hash를 검증하고, caller에는 mask copy를 반환하여 캐시 오염을 막는다. Resolver가
생성한 context를 explicit generator와 random-area generator가 공유하므로 embedded
target도 같은 cache와 generator registry를 통해 resolve된다.

#### 코어와의 관계

실행 코어는 family가 어떻게 확장되었는지 모르고 concrete recipe만 처리한다.
향후 pose·detection 연동은 SampleRegionProvider와 해당 RegionResolver generator를
추가하는 방식으로 확장한다.

---

### M5. Perturbator

**인터페이스.**

```
apply(array, mask, op, params, rng=None) -> np.ndarray
```

#### operator dispatch와 확장

`Perturbator`는 공통 입력·출력 계약만 검증하고 op별 pixel 연산을 직접 구현하지
않는다. `PerturbationOperator`는 `supports(op)`, `validate_config(params)`,
`requires_dataset_stats()`, `resolve_config_params(...)`,
`apply(array, mask, params, rng=None)`를 정의한다. ConfigResolver와 runtime이 같은
operator 순서와 operation별 계약을 공유한다. constant fill, mean fill, blur,
Gaussian noise, patch shuffle가 각각 별도 구현체이며 ordered 목록을 순회해
처음 `supports=True`인 구현체를 실행하므로 등록 순서가 명시적인 우선순위다.

`OperatorFactory`는 operator class를 instance-local registry에 보관하고 build마다
fresh instance 목록을 만든다. 기본 목록도 전역 mutable registry 없이 factory에서
구성한다. 따라서 새 연산의 구현·등록은 `Perturbator`를 수정하지 않고 수행하며,
custom factory 결과를 생성자에 주입할 수 있다.

**순수 함수 계약.** 같은 입력 → 같은 출력. 전역 상태 참조 금지. `np.random` 전역 API 사용 금지 — 반드시 넘겨받은 rng만 사용한다.

**마스크 반전 처리.** Perturbator는 반전을 모른다. 호출 전에 `mask = ~mask`가 적용된다. 연산자는 "마스크가 True인 픽셀을 이렇게 바꾼다"만 구현하면 된다.

**v1 연산자 (deletion 계열).**

| op | params | 비고 |
|---|---|---|
| `constant_fill` | `value` | scalar 또는 채널별 값 |
| `mean_fill` | (DatasetStats에서) | ConfigResolver가 확정한 채널별 값 |
| `blur` | `sigma` | 전체 frame을 Gaussian blur한 뒤 mask 영역만 합성 |
| `gaussian_noise` | `sigma` | 독립 Gaussian noise, 반올림 및 uint8 clip |
| `patch_shuffle` | `patch_size` | 완전한 tile들의 위치를 shuffle하고 불완전 가장자리는 유지 |

**시간 축 처리.** v1은 모든 프레임에 동일 마스크·동일 연산을 적용한다.
`patch_shuffle`의 spatial permutation도 모든 프레임에서 공유한다. T=1이므로
실질적으로 무의미하지만, 인터페이스는 T를 인지한다.

**rng 유도.** 워커에서 매 아이템마다 새 인스턴스를 만든다. 재사용하지 않는다.

```
rng = np.random.default_rng(derive(global_seed, item_id, seed_salt))
```

`derive`는 namespace와 세 입력의 SHA-256 앞 128비트를 seed로 사용한다. 이것이
워커 수·실행 순서·재개와 무관한 재현성의 근거이다.

---

### M6. ChunkProcessor (워커 진입점)

DataLoader의 `Dataset.__getitem__`에 해당한다.

```
__getitem__(chunk_index) -> PreparedChunk | FailedChunk
```

**처리 순서.**

1. `chunk_id = chunk_ids[chunk_index]`
2. `chunk = PlanBuilder.materialize(chunk_id)`
3. `loaded = SampleSource.load(chunk.sample_id)`
   - 실패 시 → `FailedChunk(chunk_id, reason="load_failed", item_ids=...)` 반환하고 종료
4. 각 아이템에 대해: resolve → 반전 적용 → perturb → 리스트에 누적
   - 개별 아이템 실패 시 `prepare_failed`로 분리하고 계속
   - `fail_fast=true`이면 첫 준비 실패에서 worker 처리를 멈춘다
5. `PreparedChunk` 반환

**PreparedChunk.**

```
chunk_id: str
arrays: np.ndarray         # (k, T, H, W, C) uint8
masks: np.ndarray          # (k, H, W) bool, 반전까지 적용된 실제 교란 mask
item_metas: list[ItemMeta] # 성공 item_id, region_meta
failed_items: list[ItemMeta]
```

WorkItem은 IPC로 전송하지 않는다. main process는 `chunk_id`로
`PlanBuilder.materialize()`를 다시 호출해 dump metadata를 복원하고 worker 반환의
누락·중복·순서를 검증한다. `invert_mask=true`일 때 intended/effective area는 모두
반전 후 실제 적용 mask를 기준으로 한다.

**메모리 관점.** 반환 크기는 `k × T × H × W × C` 바이트이다. 224×224 이미지, k=16이면 약 2.4MB. 워커 8개 + prefetch 2면 약 40MB로 감당 가능하다. 비디오로 가면 T배가 되므로 k를 줄여야 하며, 이것이 `variants_per_chunk`를 설정으로 노출한 이유이다.

#### 배경: 기존 실험 코드 대비 개선점

기존 실험 코드는 인덱스 단위가 (샘플 × 교란)이었다. 워커가 idx=k를 받으면 샘플을 로드하고, idx=k+1을 받으면 같은 샘플을 다시 로드한다. 영역이 8개면 8번 디코드하게 된다.

본 설계는 작업 단위를 **(샘플 1개, 변형 부분집합)**으로 바꾸어 디코딩을 k회 분할상환한다. 동시에 반환 크기가 k로 상한이 있어 메모리가 예측 가능하다.

한 샘플이 여러 청크로 쪼개지면 디코딩이 청크 수만큼 반복된다. 64변형을 k=16으로 나누면 4회 디코드로, 64회보다는 훨씬 낫지만 1회는 아니다. 이 절충은 `variants_per_chunk` 조정으로 제어한다.

---

### M7. Rebatcher

워커 출력 스트림을 모델 배치로 재구성한다.

**DataLoader 설정.** `batch_size=None` (자동 배칭 끔), 순서 고정, identity
collate를 사용한다. NumPy 배열을 tensor로 암묵 변환하지 않으며 모델 객체는 worker에
전달하지 않는다. clean 입력은 `CleanProcessor`, 교란 입력은 `ChunkProcessor`가
같은 원칙으로 로드한다.

**동작.** 성공 item을 버퍼에 누적하다가 공유 `BatchSizeState.current_size`에
도달하면 잘라서 내보낸다. **청크 경계를 무시**하므로 GPU 활용률이 유지된다.
source array shape가 바뀌면 stack 전에 현재 버퍼를 작은 batch로 즉시 flush한다.

```
buffer = []
for chunk in loader:
    buffer.extend(chunk.items)
    while len(buffer) >= state.current_size:
        yield buffer[:state.current_size]
        buffer = buffer[state.current_size:]
yield buffer  # 잔여분
```

**실패 아이템 처리.** `FailedChunk`나 청크 내 실패 아이템은 배치에 넣지 않고 DumpWriter로 직행한다.

공개 `run_audit()`는 ResumeIndex로 완료된 clean sample과 WorkChunk를 먼저 제외한 뒤
clean inference를 완료하고 perturbed inference를 수행한다. 정상 반환 전 writer를
flush하지만 close하지 않으며, writer의 정상·비정상 close는 호출자가 책임진다.

---

### M8. ModelAdapter

ModelAdapter는 source-space 입력과 raw logits 사이의 안정된 외부 계약만
노출한다. 구현 내부는 전처리, 모델 호출, 출력 해석을 분리한다.

```
THWC uint8 batch
    → Preprocessor.transform_batch()
    → model callable
    → OutputDecoder.decode()
    → list[RawOutput]
```

**외부 인터페이스.**

```
describe() -> AdapterSpec
predict(batch: np.ndarray) -> list[RawOutput]
transform_mask(mask) -> np.ndarray | None    # 선택
cleanup_after_oom() -> None                  # 기본 no-op
```

`transform_mask()`는 adapter가 소유한 Preprocessor에 위임한다. 사용자 callable이
모델 내부에서 전처리를 숨기는 경우에는 `None`을 반환하며 effective model-space
area를 제공하지 않는다.

**Preprocessor.**

```
describe() -> PreprocessingSpec
transform_batch(batch: np.ndarray) -> object
transform_mask(mask: np.ndarray) -> np.ndarray | None
```

`PreprocessingSpec`은 kind, 결정론 여부, 설명, 선택적 canonical fingerprint와
mask transform 가용성을 기록한다. 픽셀과 mask의 기하 변환은 반드시 같은
Preprocessor가 소유한다. 선언형 구현은 resize, center crop, float 변환,
normalize, layout 변환을 지원하고 mask에는 resize와 crop만 nearest-neighbor로
적용한다.

**OutputDecoder.** 기본 `LogitsOutputDecoder`는 torch-like tensor, `(B,C)`
floating numpy 배열, `{"logits": ...}` mapping, `RawOutput` 또는 1차원 배열
sequence를 동일한 logits 계약으로 정규화한다. batch 길이와 class 차원 검증도
decoder가 담당한다.

**AdapterSpec.**

```
deterministic: bool
input_layout: str              # "THWC_uint8" 등
max_batch_size: int | None
output_kind: "logits"          # v1 고정
class_names: list[str] | None
preprocessing_desc: str        # 문서화용
adapter_kind: str
model_name: str | None
weights_id: str | None
weights_hash: str | None        # 제공된 경우 lowercase SHA-256
preprocessing_fingerprint: str | None
mask_transform_available: bool
```

#### 결정론 요구

평가 전처리에 random crop이나 flip이 있으면 clean과 교란 입력이 서로 다른 변환을 받아, 관측된 차이가 교란 때문인지 전처리 때문인지 구분할 수 없다. 코어는 실행 전에 `describe()`를 확인하고 비결정론이면 경고 또는 거부한다. 이것은 관행이 아니라 **결과 타당성의 전제조건**이다.

#### transform_mask의 역할

center crop 등이 마스크 일부를 잘라낼 수 있으므로, 실제 모델에 도달한 면적과 의도한 면적이 다를 수 있다. 어댑터가 자신의 전처리와 **동일한 기하 변환**을 마스크에 적용해 반환하면, 코어는 그 결과로 `effective_area_px`를 계산해 dump에 기록한다.

제공하지 않으면 `effective_area_px = null`, `effective_area_available = false`로 기록한다.

이 값이 있어야 후단의 면적 통제와 마스크–GT 중첩 정책이 정확해진다. 구현은 마스크를 nearest-neighbor로 같은 resize/crop에 태우면 된다.

#### v1 구현체

`TorchvisionAdapter`, `TimmAdapter`, `CallableAdapter`(사용자 함수 래핑),
`DeclarativeAdapter`. 기존 생성자는 유지하고 내부에서 Preprocessor와
OutputDecoder를 조합한다. torchvision/timm은 framework weight identifier를
기록하지만 provenance hash를 얻기 위한 암묵적 다운로드는 하지 않는다.

GPU OOM은 framework adapter가 공통 `AdapterOutOfMemoryError`로 변환한다.
실행 계층은 이 예외만 처리하며 torch/timm 예외 타입을 직접 import하지 않는다.
`CallableAdapter`의 사용자 함수도 복구 가능한 device OOM을 이 공통 예외로
변환해 발생시켜야 한다. 그 밖의 예외는 일반 predict 실패로 처리한다.
매 OOM 뒤 runtime은 `cleanup_after_oom()`을 호출한다. torchvision/timm 구현은
GC와 CUDA cache 정리를 수행하고 callable/declarative 구현은 선택적 callback을
호출한다.

#### 단계 10 확장 경계

CLI 도입 시 `AdapterConfig`와 이름 기반 `AdapterProviderRegistry`를 추가한다.
Provider는 설정 검증, 모델·checkpoint 생성, Preprocessor와 OutputDecoder 조합을
담당하며 사용자 provider는 registry에 명시적으로 등록한다. `ModelLoader`와
`ModelRunner`는 built-in 및 사용자 provider에서 반복 구현이 확인된 경우에만
공통 계약으로 추출한다.

package entry point 기반 자동 plugin 발견과 ONNX/MMAction2/원격 provider는
패키징 메타데이터가 준비되는 v1.1 범위다.

#### Tier 2 — manifest 모드 (v1.1)

`predict`를 호출하는 대신, 교란된 입력과 WorkItem 목록을 디스크에 내보내고 종료한다. 사용자가 외부에서 추론한 뒤 결과를 item_id와 함께 돌려주면, 코어가 읽어 dump로 정규화한다. MMAction2, 사내 파이프라인, 원격 API를 전부 수용하는 탈출구이다.

---

### M9. BatchSplitter (OOM 회복)

```
def predict_with_recovery(batch, adapter, state):
    try:
        return adapter.predict(batch)
    except AdapterOutOfMemoryError:
        adapter.cleanup_after_oom()
        state.current_size = min(state.current_size, max(1, len(batch) // 2))
        if len(batch) == 1:
            return [Failed(status="skipped_oom")]
        half = len(batch) // 2
        return (predict_with_recovery(batch[:half], adapter, state)
              + predict_with_recovery(batch[half:], adapter, state))
```

**설계 결정.**

- `state.shrink()`는 **영구적**이다(v1). 복구 로직은 넣지 않는다 — 단순하고, 한 번 OOM이 났다면 이후도 위험하기 때문이다
- GPU OOM만 여기서 처리한다. **시스템 RAM 부족은 별개 문제**이고 워커 수나 k를 줄여야 하므로, v1에서는 명확한 에러 메시지와 권고를 출력하고 중단한다
- `cleanup_gpu()`는 참조 해제와 캐시 비우기. 이것을 빠뜨리면 파편화로 계속 실패한다
- 배치 크기 1에서도 OOM이면 그 아이템은 실패 기록하고 넘어간다. 무한 재시도를 막는 종료 조건이다
- 일반 adapter 예외는 분할하지 않고 해당 batch 전체를 `predict_failed`로 기록한다
- 축소된 cap은 clean과 perturbed 실행에 공유되어 이후 모든 batch에 유지된다

**근거.** 추론에서는 배치 크기가 결과에 영향을 주지 않으므로(BatchNorm이 eval 모드인 한) 안전하게 조정 가능하다.

**재현성 영향.** 없다. 배치 구성이 달라져도 아이템별 출력은 동일하고, seed는 item_id 유도이므로 무관하다. 다만 **dump 기록 순서가 달라지므로**, 후단은 순서에 의존하면 안 된다.

---

### M10. DumpWriter

#### (a) clean/part_NNNNNNNN.parquet — sample_id 단위

```
sample_id, content_hash, gt_label, status,
logits (list[float]), original_shape, model_id, written_at
```

#### (b) perturbed/chunk_NNNNN.parquet — item_id 단위

```
item_id, sample_id, status,
region_id, region_instance_id, region_kind, region_params_json,
intended_area_px, effective_area_px,
perturb_op, perturb_params_json, invert_mask, is_control,
seed_used,
logits (list[float]),
written_at
```

전체 로짓 벡터를 저장한다. parquet의 컬럼 압축이 효율적이라 실제 용량은 예상보다 작다.
`seed_used`는 128비트 RNG seed를 보존하는 32자리 lowercase hex 문자열이다.

#### (c) index/part_NNNNNNNN.parquet — ResumeIndex용

```
item_id, status, chunk_file, written_at
```

#### (d) run_manifest.json — 실행 1회당 하나

```
resolved_config, code_version, schema_version,
adapter_spec, dataset_stats,
environment (python, torch, cuda, gpu),
started_at, finished_at, counts_by_status
```

#### 쓰기 정책

- clean과 perturbed의 독립 버퍼가 `flush_every` 레코드에 도달하면 새 fragment 작성
- 순서: data write + fsync + atomic rename → index write + fsync + atomic rename
  → manifest counts 갱신 → 버퍼 비움
- 각 chunk는 독립적으로 읽을 수 있어야 한다 (중단 시 부분 결과 보존)
- retry로 같은 ID가 재기록되면 ordinal이 큰 fragment가 authoritative하며 Reader는
  이전 행과 orphan을 노출하지 않는다

#### Reader와 버전 정책

후단은 `DumpReader`의 Arrow chunk iterator를 기본 API로 사용하고, 소형 분석에는
전체 Table 결합 API를 사용할 수 있다. manifest와 모든 parquet fragment의
`schema_version` 및 Arrow schema는 현재 버전과 정확히 일치해야 한다. v1에서는
migration 또는 경고 기반 호환 읽기를 제공하지 않는다.

전체 logits 차원이 `DumpConfig.max_classes_for_full_logits`의 기본값 10,000을 넘으면
실행 세션당 한 번 경고한다. `None`이면 경고만 비활성화하며 전체 logits는 유지한다.

---

### M11. CostEstimator

실행 전 보고하고 사용자 확인을 받는다.

**추정 입력.** 열거된 아이템 수, 소규모 프로파일 실행(예: 20청크)의 실측 처리량

**보고 항목.**

```
총 아이템 수 / 재개 후 남은 수
예상 추론 횟수
측정 처리량 (items/sec)
예상 소요 시간
예상 dump 크기 (로짓 차원 × 아이템 수 × 압축률 추정)
Sanity check (소수의 clean 샘플로 추론 후 정확도 보고. 만약 기대되는 값보다 크게 다르다면 전처리 검토 필요)
```

**임계 초과 시.** 샘플링 비율, 영역 수, seed 수 조정 권고를 출력한다. `--yes`로 건너뛸 수 있다.

---

## 3. 실패 처리 정책

기본 방침은 **진행 가능하면 계속**이다. 일부 샘플의 이상 때문에 장시간 작업이 조기 중단되는 것을 막기 위함이다. 사용자가 원하면 실패 시점에서 즉시 중단하도록 설정할 수 있다.

### status 값

| status | 의미 | 출력 필드 |
|---|---|---|
| `ok` | 정상 | 로짓 벡터 |
| `load_failed` | 디코드/읽기 실패 | null |
| `prepare_failed` | region resolve, 교란 또는 mask transform 실패 | null |
| `predict_failed` | 추론 실패 | null |
| `skipped_oom` | 배치 크기 1에서도 OOM | null |

### 핵심 규칙

**dump에 없는 item_id = 수행 안 됨.** 별도 표현이 필요 없다. "실패한 작업"과 "수행 안 된 작업"이 자연히 구분된다.

**재개 시 재시도 여부는 `retry_failed` 설정으로 제어한다.**

**샘플 로드 실패의 전파.** 로드가 실패하면 그 샘플의 **모든 변형에 대해** `load_failed` 레코드를 쓴다. 그래야 "dump에 있으면 완료"라는 단순 규칙이 유지되고, 재개 시 실패한 샘플을 반복 시도하지 않는다.

**fail-fast.** terminal failure 레코드를 먼저 기록하고 flush한 뒤
`RuntimeExecutionError`를 발생시킨다. worker 반환의 누락·중복 같은 runtime
invariant 위반은 설정과 관계없이 즉시 중단한다.

---

## 4. 재현성 보장 메커니즘

| 위협 | 대응 |
|---|---|
| 실행 순서에 따른 결과 변동 | seed를 `derive(global_seed, item_id, seed_salt)`로 유도. 전역 RNG 사용 금지 |
| 워커 수 변경에 따른 변동 | 동일. 워커별 seed에 의존하지 않음 |
| 중단·재개에 따른 변동 | 동일. 아이템별 seed는 실행 이력과 무관 |
| 배치 크기 변동(OOM 회복) | 아이템별 출력에 영향 없음. 기록 순서만 달라짐 |
| 데이터셋 통계의 실행 중 계산 | ConfigResolver에서 사전 확정하고 manifest에 기록 |
| 전처리 비결정성 | AdapterSpec의 `deterministic` 확인, 기본 거부 |
| 설정 재해석 | ResolvedConfig 이후 재해석 금지 |

**검증.** 위 항목들은 deterministic regression test로 검증한다. 워커 수를 바꾸거나 중단·재개한 실행이 동일한 dump를 산출하는지 확인한다.

이 비교에서 Parquet fragment 경계, 행의 물리 순서, `written_at`은 실행 스케줄과
flush 시점에 따라 달라질 수 있으므로 재현성 대상이 아니다. DumpReader의
authoritative 행을 clean은 `sample_id`, perturbed는 `item_id`로 정렬하고
`written_at`만 제외한 모든 논리 필드(logits, seed, status, params, 면적 포함)를
정확히 비교한다. 이 회귀 검사는 기본 pytest suite에 포함해 CI에서 항상 실행한다.

---

## 5. 잔여 결정 사항

단계 6에서 스키마 불일치는 엄격히 거부하고, 대형 logits는 설정 가능한 임계값으로
경고하며, 후단은 `DumpReader`만 사용하도록 확정했다. 향후 migration이 필요해질 때
별도 버전 변환 도구를 설계한다.

---

## 6. 설계 결정 요약

| 항목 | 결정 | 근거 |
|---|---|---|
| 프레임워크 의존 | 코어는 무의존, MMAction2 등은 선택적 adapter | 호환성 확보, 라이선스 격리 |
| 내부 표현 | `(T, H, W, C)` uint8, 이미지는 T=1 | 비디오 확장 시 코어 무수정 |
| 교란 적용 위치 | 원본 픽셀 공간 | 의미 영역 좌표계와 일치, 채움 전략 정의 자연스러움 |
| 유효 면적 | `transform_mask`로 별도 기록 | crop이 마스크를 잘라내는 문제 대응 |
| 워커 작업 단위 | (샘플 1개, 변형 부분집합) | 디코딩 분할상환 + 메모리 상한 |
| `variants_per_chunk` | 고정 설정값 | v1 단순화 |
| 배치 구성 | 청크 경계 무시, Rebatcher가 재구성 | GPU 활용률 유지 |
| ModelAdapter 위치 | 메인 프로세스만 | 단일 GPU 가정 |
| 재개 상태 관리 | 별도 인덱스 파일 | 조회 속도, 용량 부담 낮음 |
| 재개 필터 단위 | 청크 | 부분 완료 청크의 로드 비용 낭비 방지 |
| dump 로짓 | 전체 벡터 저장 | 후단 지표 재계산 자유도 확보 |
| 대조군 생성 | 사용자 명시 요청 시에만 | v1 단순화 |
| 실패 처리 | 기본 계속, 옵션으로 즉시 중단 | 장시간 작업의 조기 중단 방지 |
| OOM 회복 | 배치 분할, 축소는 영구 | 추론에서 배치 크기는 결과 무영향 |
