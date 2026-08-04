# 구현 계획서 (v1 코어)
## Spatial Sensitivity Audit Toolkit

> 본 계획서는 방향과 틀을 정하기 위한 것이며, 세부 사항은 구현 과정에서 조정한다.
> 전제: Dev Container 기반 개발, Docker Compose 기반 배포.

---

## 1. 기술 스택과 의존성 방침

### 1.1 의존성 계층 원칙

**코어는 최소 의존성만 갖는다.** 프레임워크 의존은 전부 optional extras로 분리하여, 코어만 설치해도 동작하고 필요한 어댑터만 추가할 수 있게 한다.

```
[core]                  numpy, pandas, pyarrow, pydantic, typer, PyYAML
  └ [torch]             torch, torchvision        (TorchvisionAdapter, DataLoader)
  └ [timm]              timm                      (TimmAdapter)
  └ [image]             Pillow, opencv-python     (ImageFolderSource, blur 등)
  └ [dev]               pytest, pytest-cov, ruff, mypy, pre-commit
  └ [docs]              mkdocs-material           (선택)
```

### 1.2 주요 선택과 근거

| 항목 | 선택 | 근거 |
|---|---|---|
| 설정 검증 | pydantic | ResolvedConfig의 스키마 강제, 직렬화 일관성 |
| CLI | typer | 타입 힌트 기반, 러닝커브 낮음 |
| dump 포맷 | parquet (pyarrow) | 컬럼 압축, chunk 독립 읽기, 로짓 벡터 저장 효율 |
| 배열 표현 | numpy | 프레임워크 비의존 원칙 |
| 병렬 로딩 | torch DataLoader | 검증된 구현, 기존 실험 코드 경험 재사용 |
| 린트·포맷 | ruff | 속도, 단일 도구 |

**주의.** `torch.utils.data.DataLoader`를 쓰지만, 이는 **워커 관리 유틸리티로만** 사용한다. 코어 로직은 numpy 배열만 다루며 torch 텐서에 의존하지 않는다. 향후 다른 병렬 백엔드로 교체 가능하도록 `runtime/` 하위에 격리한다.

---

## 2. 디렉터리 구조

```
ssat/                                  # 프로젝트 루트
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile                     # 개발 환경 (CUDA + dev extras)
├── docker/
│   ├── Dockerfile                     # 배포용 (slim)
│   └── docker-compose.yml             # 실행 예제 (볼륨 마운트 포함)
├── src/
│   └── ssat/
│       ├── __init__.py
│       ├── core/                      # ← v1 구현 범위
│       │   ├── config/                # M0  ConfigResolver
│       │   │   ├── schema.py          #     pydantic 모델
│       │   │   ├── resolver.py
│       │   │   └── stats.py           #     DatasetStats 사전 계산
│       │   ├── plan/                  # M1  PlanBuilder
│       │   │   ├── types.py           #     WorkItem, WorkChunk, RegionSpec
│       │   │   ├── hashing.py         #     item_id 정규화·해시
│       │   │   └── builder.py
│       │   ├── resume/                # M2  ResumeIndex
│       │   ├── source/                # M3  SampleSource
│       │   │   ├── base.py
│       │   │   └── image_folder.py
│       │   ├── region/                # M4  RegionResolver
│       │   │   ├── base.py
│       │   │   ├── grid.py
│       │   │   ├── explicit.py
│       │   │   └── random_area.py
│       │   ├── perturb/               # M5  Perturbator
│       │   │   ├── base.py
│       │   │   ├── ops.py
│       │   │   └── rng.py             #     seed 유도
│       │   ├── runtime/               # M6, M7, M9  실행 계층
│       │   │   ├── chunk_processor.py
│       │   │   ├── rebatcher.py
│       │   │   ├── batch_splitter.py
│       │   │   └── loop.py            #     전체 실행 루프
│       │   ├── adapter/               # M8  ModelAdapter
│       │   │   ├── base.py            #     인터페이스 + AdapterSpec
│       │   │   ├── callable.py
│       │   │   ├── torchvision.py     #     [torch] extra
│       │   │   ├── timm.py            #     [timm] extra
│       │   │   └── declarative.py     #     선언적 전처리 헬퍼
│       │   ├── dump/                  # M10 DumpWriter + Reader
│       │   │   ├── schema.py          #     parquet 스키마 정의
│       │   │   ├── writer.py
│       │   │   ├── reader.py          #     후단과의 계약 지점
│       │   │   └── manifest.py
│       │   └── estimate/              # M11 CostEstimator + SanityCheck
│       ├── metrics/                   # (v1 이후) 지표 엔진
│       ├── analysis/                  # (v1 이후) 대조군·안정성 분석
│       ├── report/                    # (v1 이후) JSON/CSV/HTML
│       └── cli/
│           ├── __init__.py
│           └── main.py                # typer 엔트리포인트
├── tests/
│   ├── unit/                          # 모듈별 단위 테스트
│   ├── integration/                   # 파이프라인 결합 테스트
│   ├── determinism/                   # 재현성 회귀 테스트
│   └── fixtures/                      # 합성 소형 데이터셋
├── configs/
│   ├── examples/                      # 예제 설정 YAML
│   └── schema/                        # 스키마 버전별 참조 문서
├── examples/                          # 노트북·스크립트 예제
├── docs/
├── pyproject.toml
├── README.md
└── LICENSE
```

### 2.1 구조 설계 의도

**`core/` 하위가 M0~M11에 1:1 대응한다.** 모듈 경계가 디렉터리 경계와 일치하므로, 설계 문서와 코드를 오가기 쉽고 의존 방향을 감시하기 쉽다.

**`metrics/`, `analysis/`, `report/`를 지금 비워두되 자리는 만든다.** v1 이후 확장 시 구조를 재편할 필요가 없다.

**`dump/reader.py`가 코어와 후단의 유일한 계약 지점이다.** 후단은 parquet을 직접 읽지 않고 reader API를 경유한다. 스키마 변경에 강하다.

**`runtime/`에 torch 의존을 격리한다.** 코어의 나머지는 numpy만 안다.

### 2.2 의존 방향 규칙

```
config → (없음)
plan   → config
source → config
region → plan(types)
perturb→ region
adapter→ (없음, 독립)
runtime→ plan, source, region, perturb, adapter, dump
dump   → plan(types)
```

역방향 import를 금지하고, CI에서 import-linter 등으로 검사한다.

---

## 3. 개발·배포 환경

### 3.1 Dev Container

- 베이스: CUDA 런타임 포함 Python 이미지
- `[dev]` extras 전체 설치, pre-commit 훅 자동 설정
- 워크스페이스 마운트, 데이터 디렉터리는 별도 볼륨
- GPU 패스스루 설정 포함

### 3.2 Docker Compose (배포·실행 예제)

```yaml
services:
  ssat:
    image: ssat:latest
    volumes:
      - ./data:/data:ro          # 데이터셋 (읽기 전용)
      - ./runs:/runs             # dump 출력
      - ./configs:/configs:ro
    command: ssat run --config /configs/example.yaml --out /runs/exp001
```

**설계 의도.** 데이터 읽기 전용 마운트로 원본 훼손을 구조적으로 방지한다. dump는 별도 볼륨이라 컨테이너 재생성과 무관하게 보존된다.

---

## 4. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행하며, 조건 미충족 시 다음 단계로 넘어가지 않는다.

---

### 단계 0. 프로젝트 스캐폴딩

**작업.** 디렉터리 생성, `pyproject.toml` 작성(extras 포함), Dev Container 구성, ruff/mypy/pytest 설정, CI 워크플로 골격, 빈 모듈에 인터페이스 stub 배치.

**성공 조건.**
- Dev Container에서 `pytest` 실행 시 0개 테스트로 정상 종료
- `ruff check`, `mypy src/` 통과
- `pip install -e ".[dev]"` 성공

---

### 단계 1. 스키마·타입 정의

> **가장 먼저 하는 이유:** 이후 모든 모듈이 이 타입을 참조한다. 나중에 바꾸면 전면 수정이 된다.

**작업.**
- `plan/types.py`: `RegionSpec`, `WorkItem`, `WorkChunkMeta`, `ItemMeta`, `LoadedSample`, `AdapterSpec`
- `config/schema.py`: 설정 YAML의 pydantic 모델 전체
- `dump/schema.py`: clean/perturbed/index parquet 컬럼 정의, `schema_version` 상수
- `plan/hashing.py`: 정규화 직렬화 + item_id 해시
- 예제 설정 YAML 2~3개 작성

**테스트.**
- 정상·비정상 설정 YAML의 검증 통과/실패
- 해시 결정성: 동일 입력 → 동일 id, 필드 순서 변경에도 동일 id
- 해시 민감성: 임의 필드 하나만 바꿔도 id 변경
- 부동소수 표현 차이(0.1 vs 0.10)에도 동일 id
- parquet 스키마 왕복(write→read) 일치

**성공 조건.**
- 해시 테스트 전부 통과
- 예제 설정이 스키마 검증을 통과하고, 오타 있는 설정은 명확한 에러 메시지 반환

---

### 단계 2. ConfigResolver (M0)

**작업.** 설정 로드·검증, 경로 정규화, 기본값 채움, DatasetStats 사전 계산, 어댑터 `describe()` 호출 및 결정론 검증, `ResolvedConfig` 산출.

**테스트.**
- 비결정론 어댑터 → 기본 거부, `allow_nondeterministic: true`면 경고 후 통과
- DatasetStats가 설정에 이미 있으면 재계산하지 않음
- 상대 경로가 절대 경로로 변환됨
- 유효하지 않은 op·region 조합 거부
- `ResolvedConfig` 직렬화 → 역직렬화 일치

**성공 조건.**
- 잘못된 설정이 실행 시작 전에 전부 걸러짐
- `ResolvedConfig`가 manifest에 기록 가능한 형태로 직렬화됨

---

### 단계 3. PlanBuilder (M1)

**작업.** `enumerate()`, `enumerate_clean()`, `materialize(chunk_id)`, 청킹 규칙, 대조군 열거(명시 요청 시).

**테스트.**
- 동일 설정 → 동일 청크 목록 (순서 포함)
- `enumerate()`의 item_id와 `materialize()`가 재계산한 item_id 일치 ← **핵심**
- `variants_per_chunk`에 따른 분할 정확성, 마지막 청크 처리
- 대조군 미요청 시 control 아이템 0개
- clean이 perturbed 열거에 섞이지 않음

**성공 조건.**
- 메인·워커 재계산 일치 테스트 통과 (이것이 깨지면 전체 설계가 무너짐)
- 열거 결과가 실행 없이 검증 가능

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

**작업.** `ssat run`, `ssat estimate`, `ssat rebuild-index`, `ssat inspect`(dump 요약) 명령, Docker 이미지 빌드, README·설치 문서·설정 레퍼런스, 예제 노트북.

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
