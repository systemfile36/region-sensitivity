# SSAT SoftwareX 제출 실행 계획

> 2026-08-20 작성
> 상위 문서: [`SoftwareX_SSAT_positioning_and_submission_checklist.md`](./SoftwareX_SSAT_positioning_and_submission_checklist.md)

## 0. 문서 목적

상위 문서(이하 "포지셔닝 문서")는 SSAT를 SoftwareX에 제출하기 위해 **무엇이
필요한지**를 사례 조사와 체크리스트 형태로 정리한 것이다. 이 문서는 그중
**아직 되어 있지 않은 것**만 추려서, 실제로 **무엇을 어떤 순서로 구현·실행할
것인지**를 실행 계획(phase, 작업, 산출물, 완료 기준)으로 정리한다.

작성 방식: 포지셔닝 문서의 체크리스트(4~5, 19, 29절)를 현재 저장소
(`ssat/`, `tests/`, `experiments/`, `docs/`, `.github/`) 상태와 대조하여
"완료"/"미완료"를 재판정했다. 포지셔닝 문서 자체의 체크박스는 대부분
비어 있지만, 실제로는 상당 부분이 이미 구현되어 있으므로 1절에서 이를
먼저 바로잡는다. 2절부터는 미완료 항목만을 대상으로 한 실행 계획이다.

---

# 1. 현재 상태 재평가 — 완료된 것 vs 남은 것

## 1.1 이미 완료된 것 (포지셔닝 문서 대비 재확인)

| 영역 | 포지셔닝 문서 항목 | 현재 상태 | 근거 |
|---|---|---|---|
| Functionality | model adapter | 완료 | [`ssat/core/adapter/`](../ssat/core/adapter/) — torchvision/timm/callable/declarative adapter |
| Functionality | task metric | 완료 | [`ssat/metrics/builtin_metrics/`](../ssat/metrics/builtin_metrics/), registry 확장 가능 |
| Functionality | region generator | 완료 | [`ssat/core/region/`](../ssat/core/region/) — grid, skeleton bbox 등 |
| Functionality | perturbation operator | 완료 | [`ssat/core/perturb/operators.py`](../ssat/core/perturb/operators.py) — constant/mean/blur/noise/patch-shuffle 등 |
| Functionality | matched control | 완료 | [`ssat/analysis/control.py`](../ssat/analysis/control.py) |
| Functionality | stability analysis | 완료 | [`ssat/analysis/stability.py`](../ssat/analysis/stability.py), [`reliability.py`](../ssat/analysis/reliability.py) |
| Functionality | aggregation (sample/region/class/dataset) | 완료 | [`ssat/metrics/aggregate.py`](../ssat/metrics/aggregate.py) — item/region/sample/class/spatial_profile parquet |
| Functionality | raw dump | 완료 | [`ssat/core/dump/`](../ssat/core/dump/) — schema, writer, reader, manifest |
| Functionality | report | 완료 | [`ssat/report/`](../ssat/report/) — HTML/CSV/JSON export |
| Functionality | provenance | 완료 | `run_manifest.json`에 `resolved_config`, `seed_used`, `code_version`, `EnvironmentSpec` 보존 ([`ssat/core/dump/manifest.py`](../ssat/core/dump/manifest.py), [`types.py`](../ssat/core/dump/types.py)) |
| Functionality | cache/resume | 완료 | [`ssat/core/resume/`](../ssat/core/resume/), `ssat run`이 기존 dump를 자동 재개 |
| Validation | synthetic shortcut | 완료 | [`docs/L3_Synthetic-Shortcut Experiment Report.md`](./L3_Synthetic-Shortcut%20Experiment%20Report.md) — Q1~Q5 전부 PASS |
| Validation | negative control (M_normal) | 완료 | 위 보고서 Q3 |
| Validation | multiple fills | 완료 | 위 보고서 Q4 (5개 fill strategy) |
| Validation | matched control | 완료 | 위 보고서 B auxiliary control |
| Validation | generalization-gap linkage | 완료 | 위 보고서 Q5 |
| Validation | preprocessing confound 발견·수정 | 완료 | crop-free 재실행, [`CONTROL_STABILITY_DESIGN_v1.md`](./CONTROL_STABILITY_DESIGN_v1.md) §0 |
| Software Quality | unit/integration tests | 완료 | `tests/unit/` 55개 + `tests/integration/` 9개 = 74개 파일 |
| Software Quality | CI | 완료(최소 수준) | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — pytest, `ssat --help`, compose config 검증 |
| Software Quality | error handling | 완료 | 계층별 전용 예외 모듈(`*/errors.py`)이 dump/analysis/metrics/report/estimate/runtime 전 계층에 존재 |
| Reproducibility | config/seed/model/code version 저장 | 완료 | run manifest (위와 동일) |
| Reproducibility | raw result 저장, report-raw 연결 | 완료 | dump → metrics → report 파이프라인이 raw dump를 항상 참조 |
| Sanity check (부분) | 실행 안정성 | 완료 | `ssat run`의 bounded preflight, clean accuracy/throughput sanity check ([`ssat/core/estimate/sanity.py`](../ssat/core/estimate/sanity.py)) |

## 1.2 남은 것 (이 문서가 다루는 범위)

포지셔닝 문서 §19.1("거의 필수"), §29(최종 체크리스트) 기준으로 실제
미완료로 확인된 항목만 남긴다.

1. **기존 도구(Captum) 대비 reference workflow 비교 실험** — 저장소 전체에
   `captum` 참조가 포지셔닝 문서 자체 외에는 전무함. §16 실험 미착수.
2. **실제 공개 dataset 사례** — `ssat/core/source/imagenet.py`,
   `kinetics.py` provider는 코드로는 존재하나, README에 "실제 데이터로
   검증되지 않았습니다"라고 명시된 대로 synthetic fixture로만 테스트됨.
   2개 이상 모델 비교 사례도 없음.
3. **preprocessing/effective-area sanity check의 정식 기능화** — L3
   보고서의 area confound 발견은 `experiments/synthetic_shortcut/`
   아래 1회성 수동 분석 스크립트로만 존재. `ssat` core에 PASS/FAIL을
   내리는 재사용 가능한 sanity check로 승격되지 않음.
4. **regression test** — Q1~Q5 핵심 수치가 `experiments/`의 독립
   스크립트 산출물(`.md`)로만 존재하고 `tests/`·CI에 연결된 자동
   회귀 테스트가 없음. 코드 변경이 핵심 과학적 주장을 조용히 깨뜨릴
   수 있음.
5. **패키징 성숙도** — `pyproject.toml`에 `dependencies`, `license`,
   `authors`, `classifiers` 미기재(의존성은 `requirements.txt`에만
   있음). `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md` 파일 부재.
   `git tag`가 비어 있어 `pyproject.toml`의 `version = "1.0.0"`과 달리
   실제 tagged release 없음.
6. **runtime/storage 성능 벤치마크** — `ssat/core/estimate/profiler.py`
   등 예측(estimate) 인프라는 있으나, 공개된 벤치마크 표(런타임/처리량/
   메모리/dump 크기/cache-resume 효과)가 문서화되지 않음.
7. **재현성 데모 패키지** — 개별 요소(설정/시드/코드 버전 저장)는
   되어 있으나, "이 config 하나로 논문의 특정 결과를 재현한다"는
   end-to-end 데모가 문서화되어 있지 않음.
8. **문서 공백** — tutorial(단계별 결과 해석 포함), 확장 지점
   통합 문서(Metric/PerturbationOperator/Reporter 커스텀 등록은
   Source provider·Transform 만큼 문서화되지 않음), contribution/
   support 경로 부재.
9. **Statement of Need 비교 표** — 포지셔닝 문서 §4.1, §16.2의
   비교표는 내부 기획 문서에만 있고, 논문/README에 실릴 형태로
   정리되지 않음.

---

# 2. 실행 순서 (Phase 0 ~ 7)

아래 순서는 "쉬운 것부터"가 아니라 **의존관계**를 기준으로 정했다.
Phase 0은 이후 모든 실험의 신뢰도·재현성을 뒷받침하는 최소 공학적
기반이므로 가장 먼저 처리한다. Phase 1(sanity check 정식화)은
Phase 3(실제 dataset 사례)이 "이 결과를 신뢰할 수 있다"는 근거로 직접
사용하므로 그 앞에 온다. Phase 2(Captum 비교)와 Phase 3(실제 dataset)은
포지셔닝 문서가 "거의 필수"로 지정한 두 항목이며 서로 독립적이지만,
Phase 3이 Phase 1의 산출물을 사용하므로 Phase 2 → Phase 3 순으로 둔다.
Phase 4(재현성 데모)와 Phase 5(성능 벤치마크)는 Phase 2·3이 만든
결과물을 재료로 삼는다. Phase 6(문서/CI 마무리)과 Phase 7(논문 작성)은
전체 산출물을 정리하는 마지막 단계다.

```text
Phase 0  공학적 기반 정비 (packaging, license, CI baseline, regression harness)
   │
Phase 1  Preprocessing/Effective-area sanity check 정식 기능화
   │
Phase 2  Captum reference workflow 비교 실험
   │
Phase 3  실제 공개 dataset 사례 (2개 이상 모델)
   │
Phase 4  재현성 데모 패키지
   │
Phase 5  Runtime/Storage 성능 벤치마크
   │
Phase 6  문서/테스트/CI 마무리
   │
Phase 7  Statement of Need 및 논문 작성, 최종 체크리스트 점검
```

---

## Phase 0 — 공학적 기반 정비

**목표**: 이후 실험 결과가 "재현 가능하고 설치 가능한 소프트웨어"의
산출물로 인정받을 수 있도록 최소 패키징/릴리스/회귀 방지 기반을 먼저
갖춘다. 포지셔닝 문서 §5.1, §19.1(패키징), §29(Software Quality) 대응.

**선행 조건**: 없음. 지금 바로 시작 가능.

### 작업

1. **`pyproject.toml` 보강**
   - `requirements.txt`의 의존성을 `[project.dependencies]`로 이관(또는
     `requirements.txt`를 `pyproject.toml`에서 참조하도록 정리하되,
     `pip install ssat`만으로 의존성이 해결되게 한다).
   - `license`, `authors`, `classifiers`, `urls`(repository) 필드 추가.
   - `[project.optional-dependencies]`로 `dev`(pytest 등) 분리 검토.
2. **`LICENSE` 파일 추가** — 사용할 라이선스(OSI 승인) 확정 필요(→
   `AskUserQuestion`으로 사용자에게 확인).
3. **`CITATION.cff` 추가** — SoftwareX 제출 시 표준 인용 정보 요구.
4. **`CONTRIBUTING.md` 추가** — 이슈/PR 제출 경로, 코드 스타일(기존
   docstring 컨벤션 포함), 테스트 실행 방법(Docker Compose 워크스페이스
   필수) 명시.
5. **Q1~Q5 회귀 테스트 이관** — `experiments/synthetic_shortcut/`의
   결과를 만드는 핵심 로직(모델 재추론 없이, 이미 저장된 체크포인트로
   대표 subset만 재실행하거나, 저장된 raw dump를 재분석하는 형태)을
   `tests/integration/test_synthetic_shortcut_regression.py`류로 이식.
   최소한 다음을 회귀 방지 대상으로 고정한다.
   - Q1 patch_region_rank == 1
   - Q2 multiplier >= threshold(3.0)
   - Q3 M_normal에서의 patch region rank가 하위권 유지
   - Q4 5개 fill strategy 중 최소 2개 재현
   - Q5 margin_points >= threshold(10.0)
   - 16개 grid cell의 effective area가 crop-free 설정에서 동일(3136px)
   전체 재학습은 CI에 부적합(무거움)하므로, 저장된 checkpoint
   (`experiments/synthetic_shortcut/checkpoints_crop_free/`)를 고정
   fixture로 채택하고 이를 저장소에 유지할지/LFS로 옮길지 결정한다.
6. **CI 확장** — `.github/workflows/ci.yml`에 다음 job/step 추가.
   - 위 회귀 테스트 실행
   - `ssat run examples/...` 형태의 end-to-end 예시 실행(§19.1 "End-to-end
     example" 항목과 연결, Phase 6에서 예시 config가 확정되면 채움)
   - 클린 설치 검증: `pip install .`(editable 아닌 일반 설치)이 별도
     job에서 성공하는지 확인
7. **버전/릴리스 정합성 정리** — `pyproject.toml`의 `version`을 실제
   개발 상태에 맞게 조정(예: `0.1.0`부터 시작)하거나, 제출 시점에 맞춰
   `v1.0.0` git tag 및 GitHub Release를 실제로 생성한다.

### 산출물
- 갱신된 `pyproject.toml`, 신규 `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`
- `tests/integration/test_synthetic_shortcut_regression.py` (또는 동등한
  파일) + CI에서 통과
- 확장된 `.github/workflows/ci.yml`
- 최초 git tag (예: `v0.1.0`)

### 완료 기준
- 깨끗한 환경(`docker compose` 워크스페이스 또는 CI 컨테이너)에서
  `pip install .` → `ssat --help`가 `requirements.txt`를 몰라도 성공.
- CI에서 Q1~Q5 핵심 임계값이 자동으로 검증됨.
- `LICENSE`/`CITATION.cff`가 저장소 루트에 존재.

---

## Phase 1 — Preprocessing / Effective-area Sanity Check 정식 기능화

**목표**: L3 보고서의 area-confound 발견 사례(포지셔닝 문서 §7)를
"우연히 손으로 찾아낸 버그"가 아니라 "도구가 항상 자동으로 점검해주는
기능"으로 승격한다. 이는 포지셔닝 문서가 Control/Stability 모듈을
프로젝트 핵심 기여로 승격할 가치가 있다고 판단한 근거(§7, §8)를
소프트웨어로 뒷받침하는 작업이며, Phase 3(실제 dataset 사례)의 결과를
신뢰할 수 있다고 주장하기 위한 전제 조건이기도 하다.

**선행 조건**: Phase 0(회귀 테스트 harness가 있어야 이 기능도 같은
방식으로 보호 가능).

### 작업

1. **기존 자산 재사용 범위 확정**: `ssat/metrics/viz/mask_check.py`(좌표계
   디버그 시각화)와 `experiments/synthetic_shortcut/`의 area 분석
   스크립트가 이미 하는 계산을 `ssat/core/estimate/sanity.py` 또는
   신규 모듈(`ssat/core/estimate/area_sanity.py`)로 승격할 수 있는지
   설계 검토.
2. **PASS/FAIL 판정 로직 설계**: 동일 region 정의(예: 4×4 grid)의
   nominal area 대비 실제(전처리 후) effective area 편차가 임계값을
   넘으면 `Advisory`/경고를 발생시키는 체크. 기존
   `ssat/core/estimate/types.py`의 `Advisory`/`AdvisoryCode` 패턴을
   재사용.
3. **`ssat estimate` 출력에 통합**: 기존 bounded preflight 흐름
   (`ssat/core/runtime/pipeline.py`의 `iter_clean_preparation_results`
   등)에 area sanity 결과를 추가해, 실행 전에
   `Region area consistency: PASS/FAIL`처럼 보고되게 한다(포지셔닝
   문서 §19.1 "sanity check 권장 예시"와 정확히 대응).
4. **단위/통합 테스트 추가**: crop 있는 전처리(고의로 area가 달라지는
   설정)와 crop-free 설정 각각에 대해 PASS/FAIL이 올바르게 나오는지
   `tests/unit/test_estimate_area_sanity.py`로 검증.
5. **문서화**: `docs/CONFIG_REFERENCE.md`와 `README.md`에 새 sanity
   check 항목 추가.

### 산출물
- 신규/확장된 sanity check 모듈 + 테스트
- `ssat estimate`/`ssat run` 출력에 노출되는 area/preprocessing 검사 결과
- 문서 갱신

### 완료 기준
- 의도적으로 CenterCrop을 넣은 config로 `ssat estimate`를 실행하면
  area 불균형이 자동으로 감지되어 경고/실패로 보고된다.
- crop-free config에서는 PASS로 보고된다.
- 이 동작이 CI 테스트로 고정되어 있다.

---

## Phase 2 — Captum Reference Workflow 비교 실험

**목표**: 포지셔닝 문서 §16의 "거의 필수" 1순위 실험을 실제로
수행한다. "기존 도구로 원리적으로 가능하지만 사용자가 얼마나 더 많이
구현해야 하는가"를 정량적으로 보인다.

**선행 조건**: Phase 0(측정 결과를 재현 가능한 형태로 저장·보고할
최소 인프라).

### 작업

1. **비교 대상 실험 확정**: 이미 존재하는 4×4 grid synthetic shortcut
   설정(`experiments/synthetic_shortcut/`)을 그대로 재사용해 "동일한
   분석"의 기준선으로 삼는다(새 실험 설계 불필요, 비교의 공정성도
   확보됨).
2. **Captum 기반 reference implementation 작성**: 신규 디렉터리
   `experiments/reference_comparison/captum_baseline/`에 Captum의
   `Occlusion`/`FeatureAblation`만 사용해 SSAT과 동일한 최종 결과
   (region별 ranking, Q1~Q5에 해당하는 지표)를 만드는 스크립트 작성.
   포지셔닝 문서 §16.1의 22개 항목(모델 wrapper, dataset iteration,
   여러 fill strategy, matched control, seed 반복, bootstrap, 각 단계
   aggregation, cache/resume, provenance, report 등)을 실제로 하나씩
   손으로 구현하면서 목록에서 빠짐없이 계산한다.
3. **정량 지표 수집**:
   - LOC 비교 (reference implementation vs SSAT config)
   - 실행 단계 수 비교 (§16.3 "실행 단계" 형식)
   - 사용자가 직접 결정해야 하는 설계 항목 수 비교
   - 결과 재현성 비교(동일 seed로 재실행 시 동일 결과 여부, 등)
4. **비교표 작성**: §16.2의 표를 실측치로 채워 문서화.

### 산출물
- `experiments/reference_comparison/captum_baseline/`(코드 + README)
- `docs/REFERENCE_COMPARISON_CAPTUM_v1.md` — 비교표, LOC/단계 수 실측,
  결론 서술(포지셔닝 문서 §17의 "약한 주장 vs 좋은 주장" 형식을 따름)

### 완료 기준
- Captum만으로 동일한 4×4 grid audit을 재현하는 독립 실행 가능한
  스크립트가 존재하고 실제로 동작한다.
- LOC/실행 단계/사용자 결정 항목 수가 실측되어 표로 정리되어 있다.

---

## Phase 3 — 실제 공개 Dataset 사례 (2개 이상 모델)

**목표**: synthetic 검증을 넘어 실사용성을 입증한다. 동시에
`ImageNet`/`Kinetics-400` provider를 처음으로 실제 데이터에 대해
검증하여 README의 "실제 데이터로 검증되지 않았습니다" 경고를 해소한다.
포지셔닝 문서 §18, §19.1 대응.

**선행 조건**: Phase 1(sanity check로 결과 신뢰성 뒷받침),
Phase 0(패키징 안정성).

### 작업

1. **데이터셋/모델 선정** — 다운로드 용이성과 라이선스를 고려해 확정
   필요(→ 사용자 확인 필요: 전체 ImageNet-1k validation은 무겁고 접근
   제약이 있을 수 있으므로 ImageNet-1k val 서브셋, 또는 이미 이 저장소가
   다뤄본 NTU-RGB+D 같은 대안도 검토).
   - 최소 조건: 공개 dataset 1개 + 서로 다른 model 2개 이상
     (torchvision/timm adapter로 커버 가능한 아키텍처, 예: ResNet-18
     vs ResNet-50, 혹은 CNN vs ViT).
2. **데이터 준비**: 선정된 provider(`ImageNetSourceProvider` 등)가
   요구하는 file-list/디렉터리 포맷으로 실제 데이터를 배치.
3. **동일 config(동일 region 정의, 동일 perturbation protocol)로 두
   모델에 대해 `ssat run` 실행**.
4. **분석**: 포지셔닝 문서 §18에 나열된 관찰 가능 결과 중 실제로 나온
   것 정리 — 동일 accuracy에서의 sensitivity profile 차이, 특정 class의
   sensitivity 차이, top vulnerable sample 차이, control 대비 유의미한
   region, operator stability 차이 등. "놀라운 발견"이 필수는 아니며,
   재현 가능한 실사용 사례 자체가 목적임을 문서에 명시.
5. **README 갱신**: 검증 완료된 provider에 대해 "실제 데이터로
   검증되지 않았습니다" 문구 제거 또는 조건부로 수정.

### 산출물
- `experiments/real_dataset_case_study/`(config, 실행 스크립트, 결과)
- `docs/REAL_DATASET_CASE_STUDY_v1.md`
- README 경고 문구 갱신

### 완료 기준
- 실제 공개 dataset + 2개 이상 실제 모델에 대해 `ssat run` → `ssat
  inspect`/report까지의 전체 파이프라인이 성공적으로 완료된 dump가
  존재한다.
- Phase 1의 sanity check가 이 실행에서도 PASS를 보고한다.

---

## Phase 4 — 재현성 데모 패키지

**목표**: "config 하나로 논문의 특정 결과를 제3자가 재현할 수 있다"는
것을 실제로 시연 가능한 형태로 패키징한다. 포지셔닝 문서 §16.3
"재현성", §19.1 "End-to-end example", §29 "작은 reproducible demo" 대응.

**선행 조건**: Phase 2, Phase 3의 결과물(재현 대상이 되는 구체적인
figure/table이 있어야 함).

### 작업

1. **재현 대상 확정**: Q1~Q5 표(synthetic) 또는 Phase 3의 sensitivity
   profile 비교 중 하나를 "논문에 실릴 대표 결과"로 지정.
2. **단일 커맨드 데모 구성**: 이미 존재하는 `configs/examples/`
   패턴을 확장해, 예를 들어
   `ssat run examples/reproduce_q1_q5.yaml --output <dir>` 실행 후
   `ssat inspect`/기존 분석 스크립트로 Q1~Q5 표와 정확히 같은 수치가
   나오는지 확인하는 절차를 만든다.
3. **클린 환경 검증**: 새 clone + Phase 0에서 정비한 설치 절차만으로
   데모가 재현되는지 실제로 확인(가능하면 Phase 0 CI에 이 데모의
   경량 버전을 편입).
4. **문서화**: README 또는 별도 `docs/REPRODUCIBILITY_DEMO_v1.md`에
   커맨드 시퀀스와 기대 출력을 명시.

### 산출물
- `configs/examples/reproduce_*.yaml`
- `docs/REPRODUCIBILITY_DEMO_v1.md`

### 완료 기준
- 문서에 적힌 명령만 그대로 실행해서 논문에 실릴 수치와 일치하는
  결과가 재현된다(제3자 관점에서, 즉 이 프로젝트를 모르는 사람이
  따라할 수 있는 수준으로 검증).

---

## Phase 5 — Runtime/Storage 성능 벤치마크

**목표**: 포지셔닝 문서 §19.1 "작은 software benchmark", §29
"Performance" 항목을 채운다.

**선행 조건**: Phase 3(벤치마크 대상이 될 실제 규모의 실행 결과)이
있어야 유의미한 수치가 나온다.

### 작업

1. **측정 항목 확정**: runtime, samples/sec, peak memory, raw dump
   크기, cache/resume 효과(강제 중단 후 재개 시 절약되는 시간).
2. **측정 스크립트 작성**: 기존 `ssat/core/estimate/profiler.py`,
   `cost_model.py`를 재사용/확장하여 Phase 3의 실제 dataset 실행과
   quickstart synthetic 실행 두 규모에서 측정.
3. **결과 문서화**: `docs/BENCHMARK_v1.md`에 표 형태로 정리, 실행
   환경(CPU/GPU, 코어 수, 메모리) 명시.

### 산출물
- 벤치마크 스크립트(`scripts/` 또는 `experiments/`)
- `docs/BENCHMARK_v1.md`

### 완료 기준
- 최소 두 가지 규모(quickstart, 실제 dataset)에 대한 runtime/throughput/
  memory/저장공간/cache-resume 수치가 문서로 남아 있다.

---

## Phase 6 — 문서/테스트/CI 마무리

**목표**: 포지셔닝 문서 §5.6, §29 "Software Quality"의 나머지 공백을
닫는다.

**선행 조건**: Phase 0~5의 산출물(문서화할 대상이 확정되어 있어야 함).

### 작업

1. **Tutorial 보강**: README의 quickstart를 "커맨드 나열"에서 "각
   단계에서 무엇을 보게 되는지, 출력을 어떻게 해석하는지"까지 포함한
   단계별 tutorial로 확장(신규 사용자가 결과 해석까지 따라갈 수 있게).
2. **확장 지점(extension) 문서 통합**: 현재 Source provider·Transform
   등록만 `CONFIG_REFERENCE.md`에 문서화되어 있음. Metric,
   PerturbationOperator, Reporter의 커스텀 등록 방법도 동일 수준으로
   문서화하여 "core와 사용자 확장의 경계"(포지셔닝 문서 §5.2)를
   명확히 한다.
3. **Issue/PR 템플릿**: `.github/ISSUE_TEMPLATE/`, PR 템플릿 추가(선택).
4. **CI 최종 점검**: Phase 0~5에서 추가된 모든 테스트/데모/벤치마크
   경량 버전이 CI에 실제로 편입되어 있는지 재확인.

### 산출물
- 확장된 README/CONFIG_REFERENCE
- (선택) issue/PR 템플릿

### 완료 기준
- 포지셔닝 문서 §5.6 체크리스트 전 항목이 실제로 충족됨.

---

## Phase 7 — Statement of Need 및 논문 작성, 최종 점검

**목표**: 지금까지의 실험/공학적 산출물을 SoftwareX 원고로 조립하고,
포지셔닝 문서 §29 최종 체크리스트를 항목별로 실제 확인한다.

**선행 조건**: Phase 0~6 전체 완료.

### 작업

1. **비교표 작성**: 포지셔닝 문서 §4.1이 요구하는 "기존 유사 도구
   4~6개 이상과의 기능 비교표"를 Captum(Phase 2), RobustCheck,
   CleanAI, Backdoor Pony, RobustBench, FiftyOne Brain 등을 대상으로
   작성(§31 참고 목록 활용, 각 도구의 최신 상태를 제출 직전 재확인).
2. **Contribution 3종 절 작성**: 포지셔닝 문서 §28의 구조
   (Experimental Generalization / Reliability-aware Analysis /
   Reproducible Research Software)를 그대로 논문 절 구조로 사용.
3. **피해야 할 표현·권장 표현 재점검**: 원고 초안에 포지셔닝 문서
   §24(피해야 할 주장)에 해당하는 문구가 없는지, §25(권장 표현)을
   충분히 사용했는지 검토.
4. **최종 체크리스트 재실행**: 포지셔닝 문서 §29 전체를 실제 저장소/
   원고 상태와 대조하여 항목별로 확인 표시.
5. **제출 직전 외부 사실 재확인**: §31에 명시된 대로 비교 대상
   소프트웨어들의 최신 릴리스/문서/citation 정보를 제출 직전 다시
   확인.

### 산출물
- SoftwareX 원고 초안
- 체크된 §29 최종 체크리스트

### 완료 기준
- §29의 모든 항목이 실제로 체크 가능한 상태(제출 준비 완료).

---

# 3. 열린 결정 사항 (사용자 확인 필요)

이 계획을 실행하기 전에 다음은 사용자의 선택이 필요하다.

1. **Phase 0 라이선스 종류** — MIT/Apache-2.0/BSD 등 중 무엇을 쓸지.
2. **Phase 3 실제 dataset 선정** — ImageNet-1k 서브셋 vs 이미 다뤄본
   NTU-RGB+D(비디오) vs 다른 후보. 접근성·라이선스·컴퓨팅 자원 제약에
   따라 달라짐.
3. **Phase 0 버전 정책** — `pyproject.toml`의 `version = "1.0.0"`을
   유지하며 지금 첫 tag를 찍을지, 실제 성숙도에 맞춰 `0.x`로
   낮출지.

이 세 가지는 각 Phase 착수 시점에 별도로 확인한다.
