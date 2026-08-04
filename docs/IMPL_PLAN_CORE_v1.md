# 구현 계획서 (v1 코어)
## Spatial Sensitivity Audit Toolkit

> 본 계획서는 방향과 틀을 정하기 위한 것이며, 세부 사항은 구현 과정에서 조정한다.
> 전제: Dev Container 또는 같은 이미지를 사용하는 Docker Compose 워크스페이스에서 개발·테스트한다.

---

## 1. 기술 스택과 의존성 방침

### 1.1 의존성 관리 원칙

현재 저장소의 익숙한 구성을 유지하여 Python 의존성은 루트의 `requirements.txt`, 시스템 의존성은 `scripts/install_deps.sh`에서 관리한다. `.devcontainer/Dockerfile`은 두 파일을 사용해 공통 워크스페이스 이미지를 만들고, Dev Container와 `compose.yaml`이 같은 이미지를 사용한다.

의존성 목록과 버전 제약은 고정된 최종 명세가 아니다. 구현 단계, 어댑터 추가, 호환성 검증 결과에 따라 패키지가 추가·제거되거나 버전 범위가 변경될 수 있으며, 변경 후에는 워크스페이스 이미지를 다시 빌드한다.

코어의 **논리적 의존 경계**는 계속 최소화한다. 현재 이미지에 torch/timm 등이 함께 설치되더라도 프레임워크 의존 코드는 `runtime/`과 해당 adapter 모듈에 격리하고, 나머지 코어 계약은 numpy 기반을 유지한다. 패키지 배포가 필요해지는 시점에만 optional extras와 별도 패키징 메타데이터 도입을 검토한다.

### 1.2 주요 선택과 근거

| 항목 | 선택 | 근거 |
|---|---|---|
| 설정 검증 | pydantic | ResolvedConfig의 스키마 강제, 직렬화 일관성 |
| CLI | typer | 타입 힌트 기반, 러닝커브 낮음 |
| dump 포맷 | parquet (pyarrow) | 컬럼 압축, chunk 독립 읽기, 로짓 벡터 저장 효율 |
| 배열 표현 | numpy | 프레임워크 비의존 원칙 |
| 병렬 로딩 | torch DataLoader | 검증된 구현, 기존 실험 코드 경험 재사용 |
| 테스트 | pytest | 현재 워크스페이스의 공통 테스트 러너 |

**주의.** `torch.utils.data.DataLoader`를 쓰지만, 이는 **워커 관리 유틸리티로만** 사용한다. 코어 로직은 numpy 배열만 다루며 torch 텐서에 의존하지 않는다. 향후 다른 병렬 백엔드로 교체 가능하도록 `runtime/` 하위에 격리한다.

---

## 2. 디렉터리 구조

```
프로젝트 루트/
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile                     # Dev Container/Compose 공용 이미지
├── compose.yaml                       # VS Code 서버 없는 워크스페이스 실행
├── requirements.txt                   # Python 의존성 (구현에 따라 변경 가능)
├── scripts/
│   └── install_deps.sh                # 시스템 의존성
├── ssat/                              # /workspace PYTHONPATH에서 직접 import
│   ├── __init__.py
│   ├── utils/                         # 공통 logging·파일 I/O
│   └── core/                          # ← v1 구현 범위
│       ├── config/                    # M0  설정 스키마·Resolver
│       ├── plan/                      # M1  작업 타입·해싱·PlanBuilder
│       ├── resume/                    # M2  ResumeIndex
│       ├── source/                    # M3  SampleSource
│       ├── region/                    # M4  RegionResolver
│       ├── perturb/                   # M5  Perturbator
│       ├── runtime/                   # M6, M7, M9 실행 계층
│       ├── adapter/                   # M8  ModelAdapter
│       ├── dump/                      # M10 DumpWriter + Reader
│       └── estimate/                  # M11 CostEstimator + SanityCheck
├── tests/
│   ├── unit/                          # 모듈별 단위 테스트
│   ├── integration/                   # 파이프라인 결합 테스트
│   ├── determinism/                   # 재현성 회귀 테스트
│   └── fixtures/                      # 합성 소형 데이터셋
├── configs/
│   ├── examples/                      # 예제 설정 YAML
│   └── schema/                        # 스키마 버전별 참조 문서
├── examples/                          # 노트북·스크립트 예제
└── docs/
```

### 2.1 구조 설계 의도

**`core/` 하위가 M0~M11에 1:1 대응한다.** 모듈 경계가 디렉터리 경계와 일치하므로, 설계 문서와 코드를 오가기 쉽고 의존 방향을 감시하기 쉽다.

**`metrics/`, `analysis/`, `report/`를 지금 비워두되 자리는 만든다.** v1 이후 확장 시 구조를 재편할 필요가 없다.

**`dump/reader.py`가 코어와 후단의 유일한 계약 지점이다.** 후단은 parquet을 직접 읽지 않고 reader API를 경유한다. 스키마 변경에 강하다.

**`runtime/`에 torch 의존을 격리한다.** 코어의 나머지는 numpy만 안다.

### 2.2 의존 방향 규칙

```
types  → (없음)
utils  → (없음)
config → types, adapter(contracts), source(contracts), utils
region → types
plan   → types, region
source → (없음, 독립)
adapter→ (없음, 독립)
perturb→ region
runtime→ plan, source, region, perturb, adapter, dump
dump   → types
```

역방향 import를 금지하고, CI에서 import-linter 등으로 검사한다.

---

## 3. 개발·배포 환경

### 3.1 공통 워크스페이스 이미지

- `.devcontainer/Dockerfile`은 CUDA 런타임이 포함된 PyTorch 이미지를 기반으로 시스템 및 Python 의존성을 설치한다.
- Dev Container는 저장소를 `/workspace`에 바인드하고 GPU와 32GB shared memory를 전달한다.
- `compose.yaml`은 같은 Dockerfile을 빌드하여 VS Code 서버 없이 `sleep infinity`로 유지되는 대화형 워크스페이스를 제공한다.
- Compose 서비스의 작업 디렉터리는 바인드된 `/workspace`로 고정한다.
- 루트 패키지 `ssat/`는 `PYTHONPATH=/workspace`를 통해 별도 editable install 없이 import한다.
- 테스트는 두 환경 모두 컨테이너 안에서 `pytest`로 실행한다.

### 3.2 Docker Compose 워크스페이스

```bash
docker compose up -d --build region-sensitivity-workspace
docker compose exec region-sensitivity-workspace pytest -q
```

현재 Compose 파일은 개발·테스트용이다. 읽기 전용 데이터와 별도 dump 볼륨을 갖는 배포용 이미지·Compose는 CLI가 완성되는 단계 10에서 별도 산출물로 추가한다.

### 3.3 로깅

라이브러리는 기본적으로 로그를 출력하지 않고, 애플리케이션 경계에서 `logger_factory.configure_logging()`으로 UTC 콘솔과 선택적 파일 로그를 활성화한다. 레벨·민감 정보·중복 기록 기준은 [LOGGING_POLICY.md](LOGGING_POLICY.md)를 따른다.

---

## 4. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행하며, 조건 미충족 시 다음 단계로 넘어가지 않는다.

---

### 단계 0. 프로젝트 스캐폴딩

**작업.** 현재 Dev Container/Compose/requirements 구성을 검증하고, 루트 `ssat/` 패키지와 테스트 디렉터리를 생성하며 빈 모듈에 인터페이스 stub을 배치한다. 패키징 메타데이터, lint/type-check 도구, CI는 실제 도입 시점에 현재 의존성 방식에 맞춰 추가한다.

**성공 조건.**
- Dev Container와 Compose 워크스페이스에서 `python -c "import ssat"` 성공
- Compose 워크스페이스에서 `pytest` 실행 성공

---

### 단계 1. 스키마·타입 정의

> **가장 먼저 하는 이유:** 이후 모든 모듈이 이 타입을 참조한다. 나중에 바꾸면 전면 수정이 된다.

**작업.**
- `config/schema.py`: 감사 명세 중심 설정 YAML의 pydantic v2 모델. source/adapter 객체는 후속 Resolver에 주입
- 도메인별 `types.py`: `RegionSpec`, `RegionMeta`, `WorkItem`, `WorkChunkMeta`, `WorkChunk`, `SampleMeta`, `LoadedSample`, `LoadError`, `AdapterSpec`, `RawOutput`, `ItemMeta`, prepared/failed chunk 타입
- `dump/schema.py`: clean/perturbed/index parquet 컬럼 정의, `schema_version` 상수
- `plan/hashing.py`: 정규화 직렬화 + item_id 해시
- 예제 설정 YAML 3개 작성

**테스트.**
- 정상·비정상 설정 YAML의 검증 통과/실패
- 해시 결정성: 동일 입력 → 동일 id, 필드 순서 변경에도 동일 id
- 해시 민감성: 임의 필드 하나만 바꿔도 id 변경
- 부동소수 표현 차이(0.1 vs 0.10)에도 동일 id
- parquet 스키마 왕복(write→read) 일치
- 중복 `region_id`, 존재하지 않는 control 참조, explicit ref 누락 거부

**성공 조건.**
- 해시 테스트 전부 통과
- 예제 설정이 스키마 검증을 통과하고, 오타 있는 설정은 명확한 에러 메시지 반환

**v1 계약 명확화.**
- 대조군의 `match_area_of`는 설정에서 고유한 `region_id`를 참조하고, dump에도 해당 ID를 기록한다.
- Adapter 출력과 raw dump는 v1에서 전체 `logits`로 제한한다. `probs` 지원은 dump 컬럼 계약을 일반화하는 후속 버전에서 검토한다.
- `schema_version=1.0.0`과 canonical JSON + SHA-256 전체 hex를 item ID 계약으로 고정한다.

---

### 단계 2. ConfigResolver (M0)

**작업.** 설정 로드·검증, 경로·explicit mask hash 확정, v1 perturbation params 검증, 필요한 경우 DatasetStats 사전 계산, 어댑터 `describe()` 호출 및 결정론 검증, manifest-ready `ResolvedConfig` 산출. 이 단계에서 공통 `logger_factory`, YAML/JSON·atomic write·파일 hash 유틸과 최소 Adapter/SampleSource Protocol도 함께 구현한다.

**테스트.**
- 비결정론 어댑터 → 기본 거부, `allow_nondeterministic: true`면 경고 후 통과
- DatasetStats가 설정에 이미 있으면 재계산하지 않음
- `mean_fill`을 사용할 때만 누락된 DatasetStats를 계산하고 `LoadError` 샘플은 경고 후 제외
- 상대 경로가 절대 경로로 변환됨
- explicit mask의 제공 hash와 실제 hash 불일치 거부
- 유효하지 않은 op·region 조합 거부
- `ResolvedConfig` 직렬화 → 역직렬화 일치
- UTC logger 포맷, 선택적 파일 출력, 반복 설정 시 handler 중복 방지
- YAML/JSON 로드, SHA-256, atomic JSON write 검증

**성공 조건.**
- 잘못된 설정이 실행 시작 전에 전부 걸러짐
- `ResolvedConfig`가 manifest에 기록 가능한 형태로 직렬화됨
- 라이브러리 import만으로 로그가 출력되지 않고 CLI가 명시적으로 로깅을 활성화할 수 있음

---

### 단계 3. PlanBuilder (M1)

**작업.** `PlanBuilder`는 `SampleSource.list_samples()`를 최초 접근 시 한 번만
호출하고, 픽셀을 로드하지 않은 채 `sample_id` 순으로 실행 계획을 고정한다.
`enumerate()`는 가벼운 `WorkChunkMeta`만 반환하고, `enumerate_clean()`은 정렬된
`SampleMeta`를 별도 반환한다. 일반 WorkItem은 각 샘플에서 설정의
region → perturbation → seed salt 순서로 열거하며, 명시된 area-matched control은
일반 항목 뒤에 control 요청 → control index → perturbation → seed salt 순서로
추가한다.

각 샘플의 항목은 `variants_per_chunk` 단위로 나눈다. chunk ID는 schema version,
sample ID, 샘플 내 chunk ordinal, ordered item ID를 canonical JSON으로 묶은
SHA-256 전체 hex로 계산한다. PlanBuilder는 immutable metadata와 chunk locator만
캐시하고, `materialize(chunk_id)`에서 해당 샘플의 항목을 다시 열거한 뒤 metadata의
item ID와 chunk ID가 모두 일치하는지 검증한다. Control의 `random_area_match`
RegionSpec에는 target region의 전체 resolved JSON recipe와 control 요청·control
index를 내장하여 다음 단계의 RegionResolver가 외부 registry 없이 이를 해석할
수 있게 한다.

**테스트.**
- source 반환 순서가 달라도 sample·clean·chunk 순서와 ID가 동일함
- region → perturbation → seed salt 순서와 전체 WorkItem 수가 정확함
- `enumerate()`의 item ID 및 chunk ID와 `materialize()` 재계산 결과가 일치함 ← **핵심**
- `variants_per_chunk`에 따른 정확한 분할, 짧은 마지막 chunk, 크기 1 처리
- chunk ID 회귀값 및 item 순서·ordinal·sample·schema version 변화 감지
- 빈 데이터셋, 중복 sample ID, source 예외, 알 수 없는 chunk ID 거부
- 대조군 전체 곱집합, 일반 항목 뒤 배치, 고유 ID와 self-contained target recipe
- 중복 control 요청이 request ordinal로 구분됨
- clean과 perturbed 열거 모두 `SampleSource.load()`를 호출하지 않음

**성공 조건.**
- 메인·워커 재계산 일치 테스트 통과 (이것이 깨지면 전체 설계가 무너짐)
- 열거 결과가 실행 없이 검증 가능
- 동일한 ResolvedConfig와 sample 목록에서 순서까지 완전히 동일한 계획 산출

---

### 단계 4. SampleSource + RegionResolver + Perturbator (M3, M4, M5)

> 셋을 묶는 이유: 서로 데이터를 주고받으며, 개별로는 검증 가치가 낮다.

**작업.**
- `ImageFolderSource`: (T,H,W,C) uint8 반환, `LoadError` 값 반환
- `RegionResolver`: grid, explicit, random_area_match
- `Perturbator`: constant_fill, mean_fill, blur, gaussian_noise, patch_shuffle
- `rng.py`: `derive(global_seed, item_id, seed_salt)`

**테스트.**
- 로드: 정상 이미지, 손상 파일(→LoadError), 다양한 크기
- 마스크: 면적 정확성, grid 인덱스 대응, explicit 해시 검증
- `random_area_match`: 요청 면적과 실제 면적 일치, 동일 seed → 동일 마스크
- 교란: 마스크 외부 픽셀 불변, 마스크 반전 시 정반대 영역 변경
- rng: 동일 item_id → 동일 결과, 다른 item_id → 다른 결과
- **전역 RNG 오염 검사:** 교란 실행 전후로 `np.random` 전역 상태 불변

**성공 조건.**
- 마스크 외부 불변 테스트 통과
- rng 결정성 테스트 통과
- 전역 RNG 미사용 확인

---

### 단계 5. ModelAdapter (M8)

**작업.** `base.py` 인터페이스, `CallableAdapter`, `TorchvisionAdapter`, `TimmAdapter`, `DeclarativeAdapter`(전처리 op 목록 → `transform_mask` 자동 생성).

**테스트.**
- 각 어댑터의 `describe()` 반환값 유효성
- `predict()` 출력 길이 = 배치 길이
- `transform_mask`: resize·crop 적용 후 면적이 예상 범위, nearest 보간으로 이진성 유지
- `transform_mask` 미제공 어댑터 → `effective_area_available=false` 경로 동작
- `DeclarativeAdapter`: 기하 op만 마스크에 반영, 광도 op는 무시

**성공 조건.**
- timm/torchvision 어댑터가 모델 이름만으로 동작
- `transform_mask` 면적 계산이 수동 계산과 일치

---

### 단계 6. DumpWriter + Reader + ResumeIndex (M10, M2)

**작업.** parquet chunk 쓰기, index 갱신(쓰기 순서 준수), manifest 생성, reader API, 재개 필터.

**테스트.**
- chunk 독립 읽기 가능
- 쓰기 순서: chunk fsync → index append (모의 크래시로 검증)
- 재개: 완료 청크 건너뜀, 부분 완료 청크는 통째 재실행
- `retry_failed` 설정에 따른 실패 아이템 재시도 여부
- `--rebuild-index`로 인덱스 재구축 후 원본과 일치
- reader API가 스키마 버전 불일치 감지

**성공 조건.**
- 임의 시점 중단 후 재개 시 결과 손실 없음
- 인덱스 재구축 결과가 정상 실행 인덱스와 동일

---

### 단계 7. 실행 계층 (M6, M7, M9)

**작업.** `ChunkProcessor`, `Rebatcher`, `BatchSplitter`, `loop.py` 통합.

**테스트.**
- `ChunkProcessor`: 샘플 로드 1회 확인(호출 카운터), 로드 실패 시 전 아이템 `load_failed`
- `Rebatcher`: 청크 경계 무시하고 `target_batch_size` 유지, 잔여분 처리
- `BatchSplitter`: 모의 OOM 주입 → 분할 재시도, 크기 1에서 OOM → `skipped_oom`
- 축소된 배치 크기가 이후 유지됨
- 실패 아이템이 배치에 포함되지 않고 직접 기록됨

**성공 조건.**
- 소형 합성 데이터셋에서 end-to-end 실행 완료
- 모의 OOM 하에서도 실행 완주 및 전 아이템 기록

---

### 단계 8. 재현성 회귀 테스트

> 별도 단계로 두는 이유: 이것이 도구 신뢰성의 근거이며, 개별 모듈 테스트로는 보장되지 않는다.

**테스트 항목.**

| 조건 변경 | 기대 |
|---|---|
| `num_workers` 1 → 4 | dump 내용 동일 (순서 무관 비교) |
| 중단 후 재개 vs 완주 | dump 내용 동일 |
| `target_batch_size` 변경 | dump 내용 동일 |
| OOM 회복 발생 vs 미발생 | dump 내용 동일 |
| 동일 설정 반복 실행 | item_id 집합·로짓 값 동일 |

**성공 조건.** 위 5개 전부 통과. 이 테스트는 CI 필수 항목으로 등록한다.

---

### 단계 9. CostEstimator + SanityCheck (M11)

**작업.** 소규모 프로파일 실행으로 처리량 측정, 총 비용 추정, dump 크기 추정, clean 정확도 sanity check, 임계 초과 시 권고 출력.

**테스트.**
- 추정치와 실측의 오차가 허용 범위 내(소형 데이터셋 기준)
- sanity check가 의도적으로 잘못된 전처리를 감지(정확도 급락)
- `--yes` 플래그로 확인 단계 생략

**성공 조건.**
- 잘못된 전처리 어댑터에 대해 sanity check가 경고 발생

---

### 단계 10. CLI + 통합 + 문서

**작업.** `ssat run`, `ssat estimate`, `ssat rebuild-index`, `ssat inspect`(dump 요약) 명령, 배포용 Docker 이미지와 Compose, README·설치 문서·설정 레퍼런스, 예제 노트북.

**성공 조건.**
- 문서만 보고 설치부터 dump 생성까지 재현 가능
- Docker Compose 예제가 그대로 동작
- CI에서 전체 테스트 통과

---

## 5. 단계 간 의존과 병렬화

```
0 ──> 1 ──┬──> 2 ──> 3 ──┐
          │              ├──> 7 ──> 8 ──> 9 ──> 10
          ├──> 4 ────────┤
          ├──> 5 ────────┤
          └──> 6 ────────┘
```

단계 4·5·6은 단계 1(타입 확정) 이후 **서로 독립적**이므로 순서를 바꾸거나 병행할 수 있다. 단계 7은 이들이 전부 끝나야 시작 가능하다.

**단계 1이 병목이자 가장 중요하다.** 여기서 타입과 스키마를 잘못 잡으면 이후 전 단계에 파급된다. 시간을 충분히 쓴다.

---

## 6. 테스트 전략

### 6.1 픽스처

`tests/fixtures/`에 **합성 소형 데이터셋**을 둔다. 외부 다운로드 없이 CI에서 전체 파이프라인이 돌아야 한다.

- 이미지 20~50장, 작은 해상도(예: 64×64)
- 클래스 2~3개
- 의도적으로 손상된 파일 1~2개 (로드 실패 경로 테스트용)
- 고정 출력을 내는 더미 어댑터 (모델 없이 파이프라인 검증)

### 6.2 계층

| 계층 | 범위 | 실행 시간 목표 |
|---|---|---|
| unit | 모듈 내부 함수 | 전체 10초 이내 |
| integration | 모듈 결합, 소형 end-to-end | 전체 1분 이내 |
| determinism | 재현성 회귀 | 전체 3분 이내 |

GPU가 필요한 테스트는 마커로 분리하여 CI(CPU)에서 제외한다. 더미 어댑터로 대부분을 커버한다.

---

## 7. 잔여 결정 사항의 처리 시점

| 항목 | 결정 시점 |
|---|---|
| 스키마 버전 불일치 시 동작 | 단계 6 (reader 구현 시) |
| 로짓 차원 임계 경고 | 단계 6 (dump 스키마 확정 시) |
| 후단과의 계약 형태 | 단계 6에서 reader API로 확정 |
| `variants_per_chunk` 권장값 | 단계 9 (실측 후 문서화) |
| Tier 2 manifest 모드 | v1.1로 이월 |

### 7.1 구현 전 확인할 저장 포맷 제약

Parquet 단일 파일은 일반적인 append 대상이 아니므로 단계 6에서 `index.parquet` 갱신 방식을 확정해야 한다. 인덱스 fragment dataset 또는 임시 파일 작성 후 원자적 교체 중 하나를 선택하고, 필요하면 코어 설계 문서의 단일 파일 표현도 함께 개정한다.

---

## 8. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| 단계 1의 타입 설계 오류가 후반에 발견됨 | 단계 1에서 예제 설정 3개 이상 작성해 스키마 표현력 조기 검증 |
| PlanBuilder 재계산 불일치 | 단계 3에서 전용 테스트를 최우선 작성 |
| 전처리 불일치로 결과 오염 | 단계 9의 sanity check, manifest에 전처리 명세 기록 |
| DataLoader 워커에서의 예외 전파 | `LoadError`를 값으로 반환하는 설계 유지, 예외 던지기 금지 |
| 재현성 테스트가 느려 CI 지연 | 소형 픽스처 사용, determinism 테스트를 별도 job으로 분리 |
| 범위 확대 | `metrics/` 이하는 v1에서 손대지 않음 |
