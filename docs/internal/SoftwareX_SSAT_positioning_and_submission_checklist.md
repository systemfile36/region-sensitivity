# Spatial Sensitivity Audit Toolkit — SoftwareX 포지셔닝 및 제출 준비 정리

## 0. 문서 목적

본 문서는 현재 기획 중인 **Spatial Sensitivity Audit Toolkit**을 SoftwareX에 제출하기 위한 관점에서 다음 사항을 체계적으로 정리한다.

1. SoftwareX에 게재되는 연구 소프트웨어가 대체로 어느 정도의 기능적·공학적 성숙도를 갖추는지
2. 현재 프로젝트가 그 기준에서 어느 위치에 있는지
3. 기존 XAI / robustness / data-quality 도구와 비교했을 때 본 프로젝트를 어떻게 포지셔닝해야 하는지
4. 현재 프로젝트의 핵심 기여를 무엇으로 정의하는 것이 가장 설득력 있는지
5. 제출 전 무엇을 추가 구현하거나 검증하는 것이 우선순위가 높은지
6. 특히 기존 도구 대비 필요성을 어떻게 실증해야 하는지

본 문서는 **새로운 attribution 알고리즘의 제안**이나 **shortcut learning의 자동 탐지**를 목표로 하지 않는다.

중심 질문은 다음과 같다.

> 이미 알려진 perturbation/occlusion 원리를 연구자가 반복적이고 재현 가능한 감사 실험으로 사용할 수 있도록 일반화·표준화한 소프트웨어가 SoftwareX에 제출 가능한 연구 소프트웨어가 되려면 무엇이 필요한가?

---

# 1. 현재 프로젝트의 출발점

## 1.1 이전 연구에서의 문제

이 프로젝트의 직접적인 배경은 RGB 기반 Action Recognition 모델의 입력 가림(occlusion) 연구다.

기존 연구에서는 각 샘플에 대해 신체 부위별 perturbation을 적용한 뒤, 다음과 같은 방식으로 해당 부위가 downstream task에 중요한지 판단했다.

```text
sample
  ↓
candidate region = body part
  ↓
intervention = 해당 신체 부위 가림
  ↓
Action Recognition model 재추론
  ↓
downstream task 성능 변화 측정
  ↓
원래 정답 → 가림 후 오답이면
해당 부위를 task-critical region으로 간주
```

즉, 본질적으로 다음 형태의 실험이었다.

\[
(x,\; r,\; g,\; f,\; M)
\rightarrow
M(f(x)) - M(f(g(x,r)))
\]

- \(x\): 입력 샘플
- \(r\): 관심 영역
- \(g\): perturbation / occlusion operator
- \(f\): 평가 대상 모델
- \(M\): downstream task metric

이 연구를 수행하고 나면 자연스럽게 다음 질문이 생긴다.

> 이 실험 패턴을 Action Recognition과 신체 부위에 한정하지 않고 다른 모델, 다른 task, 다른 region definition에 일반화할 수 있지 않을까?

현재 Spatial Sensitivity Audit Toolkit은 바로 이 문제의식에서 파생된 것이다.

## 1.2 기존 연구 코드를 일반화할 때 생기는 문제

단순히 기존 코드를 조금 정리하는 정도로는 범용 연구 도구가 되기 어렵다.

다른 연구자가 동일한 실험 패턴을 사용하려면 다음을 반복적으로 구현해야 한다.

- dataset iteration
- region generation
- 여러 perturbation operator
- model inference wrapper
- task-specific metric 계산
- per-sample 결과 저장
- sample-level ranking
- region-level aggregation
- class-level / dataset-level aggregation
- 동일 면적 random control
- 여러 fill strategy 비교
- random seed 반복
- bootstrap confidence interval
- operator consistency 확인
- preprocessing 및 mask-area sanity check
- raw result serialization
- provenance 저장
- cache / resume
- HTML/CSV/JSON report 생성

따라서 프로젝트의 출발점은 단순히

> “occlusion을 쉽게 해주는 프로그램”

이 아니라,

> **반복적으로 직접 구현해야 했던 intervention-based downstream-task audit workflow를 일반화하고 표준화하는 소프트웨어**

라고 보는 것이 적절하다.

---

# 2. SoftwareX에서 요구되는 성숙도의 성격

## 2.1 SoftwareX의 핵심은 “기능 수”가 아니다

SoftwareX에 제출되는 소프트웨어가 반드시 다음을 갖춰야 하는 것은 아니다.

- GUI
- 웹 대시보드
- 대규모 leaderboard
- 수십 개의 built-in model
- 수십 종류의 perturbation
- 완전히 새로운 알고리즘
- 모든 vision task 지원

실제 SoftwareX에는 비교적 좁은 기능을 수행하는 Python package도 게재된다.

중요한 것은 다음이다.

> **기존 연구자가 직접 반복 구현해야 했던 작업을 실제로 줄여주며, 독립적인 연구 소프트웨어로 설치·사용·검증·재현할 수 있는가?**

즉, 평가 기준은 대체로 다음 세 축에 가깝다.

1. **Statement of Need** — 왜 이 소프트웨어가 필요한가?
2. **Research Reusability** — 다른 연구자가 자신의 연구에 재사용할 수 있는가?
3. **Software Maturity** — 설치, 테스트, 문서, 예제, 재현성 등 연구 소프트웨어로서 기본 품질이 확보되었는가?

---

# 3. 참고할 수 있는 SoftwareX 및 유사 사례

## 3.1 RobustCheck

RobustCheck는 2024년 SoftwareX에 게재된 이미지 분류기 black-box robustness 평가용 Python package다.

특징은 다음과 같다.

- GUI 없음
- Python package
- pip 설치 가능
- 문서 제공
- 테스트 존재
- GitHub workflow 존재
- contribution guide 존재
- versioned release
- black-box robustness evaluation workflow 제공
- 사용자가 모델의 prediction interface를 넘겨 평가 가능

중요한 점은 RobustCheck가 거대한 benchmark framework가 아니라는 것이다.

핵심 가치는

> 여러 black-box robustness test를 연구자가 비교적 간단한 코드로 수행할 수 있게 만든 것

에 있다.

따라서 본 프로젝트 역시 GUI가 없다는 이유로 SoftwareX에 불리하다고 볼 필요는 없다.

## 3.2 CleanAI

CleanAI는 DNN coverage analysis를 수행하고 결과를 보고하는 SoftwareX 연구 소프트웨어다.

대표적인 특징은 다음과 같다.

- PyTorch DNN 중심
- 여러 coverage metric 지원
- evaluation workflow 제공
- analysis result 제공
- sample/example 존재
- source code와 test 제공
- 보고서 생성

구조 자체도 대체로 다음처럼 비교적 단순하다.

```text
Input
  ↓
Evaluation
  ↓
Reporting
```

이 사례 역시 화려한 GUI가 연구 소프트웨어의 필수 조건이 아님을 보여준다.

## 3.3 Backdoor Pony

Backdoor Pony는 상대적으로 높은 수준의 SoftwareX 사례다.

특징:

- Docker 기반 실행 환경
- client-server architecture
- Vue 기반 GUI
- REST API
- 여러 data domain
- 여러 attack / defense
- unit tests
- documentation
- extension API

그러나 중요한 점이 있다.

게재 당시에도 사용자 임의의 custom model과 custom dataset 지원이 제한적이었고, 일부는 future work였다.

즉,

> SoftwareX에 제출하기 위해 모든 기능이 완벽하게 일반화될 필요는 없다.

대신 이 소프트웨어는

> multi-domain attack/defense workflow를 GUI와 extension API로 통합한다

는 분명한 사용상 차별점이 있었다.

## 3.4 RobustBench

RobustBench는 SoftwareX가 아니라 NeurIPS Datasets and Benchmarks에 가까운 상위 수준의 benchmark infrastructure 사례다.

대표적으로:

- 표준화된 robustness protocol
- 대규모 model zoo
- leaderboard
- CLI
- reproducible evaluation
- automated test
- 다양한 benchmark setting

등을 갖춘다.

이 수준은 좋은 참고점이지만 **SoftwareX 제출의 최소선으로 볼 필요는 없다.**

RobustBench는 “상한선”에 가깝다.

---

# 4. SoftwareX 제출을 위한 실전 최소 성숙도 체크리스트

아래 체크리스트는 SoftwareX의 공식적인 pass/fail 규정 자체라기보다, SoftwareX 사례와 JOSS 등 연구 소프트웨어 저널의 기준을 참고하여 보수적으로 설정한 실전 기준이다.

## 4.1 연구 소프트웨어의 필요성

- [ ] 연구 질문 또는 사용 목적을 한 문장으로 설명할 수 있다.
- [ ] 기존 유사 도구 4~6개 이상과 기능 비교표가 있다.
- [ ] 기존 도구로 가능한 부분과 불가능하거나 번거로운 부분을 솔직하게 구분한다.
- [ ] 기존 도구를 조합해 비슷한 분석을 수행하는 reference implementation이 있다.
- [ ] reference implementation과 본 도구의 workflow complexity를 비교한다.
- [ ] 사용자가 직접 내려야 하는 설계 결정 수를 비교한다.
- [ ] 사용자 작성 코드량 또는 실행 단계 수를 비교할 수 있다.
- [ ] 재현성 측면의 차이를 보여준다.
- [ ] synthetic ground-truth experiment가 있다.
- [ ] 실제 공개 dataset 사례가 있다.
- [ ] 일반적인 accuracy 또는 confidence만으로는 보이지 않는 분석 결과를 하나 이상 보여준다.

현재 프로젝트에서 특히 중요도가 높은 항목:

- **reference implementation comparison**
- **real-world/public dataset case study**

---

# 5. 소프트웨어 공학적 최소선

## 5.1 설치와 실행

- [ ] 깨끗한 환경에서 문서에 적힌 명령만으로 설치 가능
- [ ] `pip install` 또는 표준 `pyproject.toml` 기반 설치
- [ ] 최소 Python version 명시
- [ ] dependency version 명시
- [ ] example command 제공
- [ ] 작은 sample dataset 또는 다운로드 절차 제공
- [ ] end-to-end example config 제공

## 5.2 인터페이스

최소한 다음 요소의 사용자 확장 지점이 명시되어야 한다.

- [ ] ModelAdapter
- [ ] TaskAdapter 또는 metric interface
- [ ] RegionGenerator
- [ ] PerturbationOperator
- [ ] Metric
- [ ] Reporter

모든 기능이 자유롭게 plugin 형태로 교체될 필요는 없지만,

> 어디까지가 core이고 어디부터 사용자가 확장할 수 있는가

는 명확해야 한다.

## 5.3 결과 관리

- [ ] raw observation과 analysis result 분리
- [ ] raw result schema 정의
- [ ] sample ID 보존
- [ ] model / checkpoint 정보 보존
- [ ] preprocessing 설정 보존
- [ ] perturbation 설정 보존
- [ ] random seed 보존
- [ ] region metadata 보존
- [ ] metric version 또는 analysis version 보존
- [ ] provenance 기록
- [ ] CSV/JSON 등 machine-readable output
- [ ] human-readable report

현재 기획의 강점 중 하나가 바로 이 부분이다.

## 5.4 실행 안정성

- [ ] cache
- [ ] resume
- [ ] 실패 sample 재시도
- [ ] deterministic regression test
- [ ] random seed control
- [ ] 동일 config 반복 실행 시 결과 일관성
- [ ] 실행 실패 시 명확한 error reporting

## 5.5 테스트와 CI

- [ ] unit test
- [ ] integration test
- [ ] end-to-end small test
- [ ] schema validation test
- [ ] regression test
- [ ] CI에서 자동 테스트
- [ ] 핵심 example이 CI 또는 별도 workflow에서 실행되는지 검토

## 5.6 프로젝트 공개 품질

- [ ] README
- [ ] installation guide
- [ ] minimal tutorial
- [ ] API / extension documentation
- [ ] license
- [ ] tagged release
- [ ] issue tracker
- [ ] contribution/support path
- [ ] citation information
- [ ] reproducible example

---

# 6. 실험 및 검증 체크리스트

## 6.1 Synthetic ground-truth validation

현재 프로젝트는 이 부분이 비교적 강하다.

현재 논문 메모 기준 synthetic shortcut 실험은 다음 조건을 포함한다.

- 특정 고정 위치에 class-correlated patch 삽입
- patch dataset으로 학습한 `M_shortcut`
- 원본 dataset으로 학습한 `M_normal`
- 동일 model architecture 사용
- crop-free preprocessing
- region effective area 균일성 확인

대표 결과:

| 검증 질문 | 결과 |
|---|---|
| Q1: shortcut region을 찾는가 | rank 1 |
| Q2: matched control과 구분되는가 | 175.65배 |
| Q3: normal model에서도 같은 위치가 상위인가 | M_normal에서는 rank 16/16 |
| Q4: 여러 fill strategy에서 재현되는가 | 5/5 |
| Q5: 실제 generalization gap과 연결되는가 | 95.75%p margin |

이 실험은 단순히

> “패치를 넣었더니 패치를 찾았다”

보다 강하다.

특히 Q3와 Q5가 중요하다.

### Q3의 의미

만약 툴이 단순히 좌상단 위치, 큰 mask, 특정 preprocessing artifact 등을 기계적으로 높은 순위에 올린다면 M_normal에서도 같은 위치가 높아야 한다.

그러나 M_normal에서는 해당 위치가 최하위였다는 점이 이를 반박한다.

### Q5의 의미

측정된 sensitivity가 단순한 output fluctuation이 아니라 실제 generalization failure와 연결되어 있음을 보여준다.

---

# 7. 전처리 버그 발견 사례의 중요성

현재 프로젝트에서 SoftwareX 관점으로 특히 가치가 높은 것은 **preprocessing confound 발견 사례**다.

초기 실험에서는 adapter가 ImageNet preset preprocessing을 하드코딩하여 사용했다.

```text
Resize(256)
  ↓
CenterCrop(224)
```

그 결과 4×4 grid의 실제 effective area가 위치별로 달라졌다.

예:

- corner cell: 2304 px
- edge cell: 3072 px
- center cell: 4096 px

즉 최대 1.78배의 면적 편차가 존재했다.

이 문제는 fill strategy 간 이상한 ranking correlation을 조사하는 과정에서 발견되었다.

대략적인 흐름:

```text
Fill strategy 간 결과 불일치
        ↓
이상 correlation 확인
        ↓
region별 effective area 분석
        ↓
preprocessing crop이 region 면적을 바꾼다는 사실 발견
        ↓
preprocessing configurable화
        ↓
crop-free로 재학습·재실행
        ↓
모든 region area = 3136 px
        ↓
핵심 Q1~Q5 결과 유지 또는 강화
```

이 사례는 매우 중요한 소프트웨어적 증거가 된다.

단순 occlusion script라면 사용자는

> “fill 방법에 따라 결과가 조금 다르네.”

정도로 넘어갈 가능성이 있다.

반면 현재 도구가 목표로 하는 audit workflow에서는

> operator consistency 이상 → confound investigation → preprocessing 문제 발견 → 수정 후 재검증

이라는 과정이 만들어졌다.

따라서 **Control/Stability Analysis는 단순한 보조 기능이 아니라 프로젝트 핵심 기여 중 하나로 승격할 가치가 있다.**

---

# 8. Stability Analysis에 대한 최신 해석

초기에는 fill strategy 간 부호 차이가 서로 다른 model dependency를 나타내는 것으로 해석될 가능성이 있었다.

그러나 crop-free 재실행 결과 초기 관측의 상당 부분은 effective-area artifact였음이 확인되었다.

이는 오히려 더 좋은 연구 서사를 제공한다.

현재 적절한 해석은 다음이다.

> Stability module의 목적은 작은 perturbation response를 모두 의미 있는 signal로 해석하는 것이 아니다.

대신:

1. confound를 조기에 탐지하고
2. operator에 따라 방향이 바뀌는 signal을 경고하고
3. 통계적으로 0과 구별되지 않는 약한 signal을 낮은 신뢰도로 분류하고
4. 실제로 반복되는 sensitivity만 후속 분석 대상으로 올리는 것

이 핵심이다.

즉:

> **무언가를 많이 발견하는 분석 모듈이 아니라, 믿으면 안 되는 perturbation response를 걸러내는 분석 모듈**

이라고 설명하는 편이 적절하다.

---

# 9. 현재 프로젝트의 기존 포지셔닝이 약하게 느껴지는 이유

현재 프로젝트를 다음과 같이 표현하면 차별성이 약하다.

> region을 가린다 → model output 변화량을 구한다 → report한다

이렇게 표현하면 자연스럽게 다음 질문이 나온다.

> “Captum Occlusion이나 FeatureAblation으로 가능한 것 아닌가?”

이 반론은 상당 부분 타당하다.

따라서 본 프로젝트는 **새로운 occlusion attribution algorithm**으로 포지셔닝하면 안 된다.

---

# 10. Captum / Occlusion XAI와의 차이

Captum과 같은 attribution library는 매우 강력한 low-level primitive를 제공한다.

대표적으로:

> 특정 feature group / region을 baseline으로 대체한 뒤 output difference 계산

이 가능하다.

즉 본 프로젝트의 핵심 차별점은 다음이 아니다.

```text
기존: occlusion 불가능
본 도구: occlusion 가능
```

정확한 비교는 다음과 같다.

```text
Captum:
single prediction / feature attribution primitive

Spatial Sensitivity Audit Toolkit:
dataset-scale experimental audit workflow
```

## 10.1 본 도구가 추가하는 계층

본 도구가 주장할 수 있는 차이는 다음이다.

- dataset-scale 반복 실행
- sample-level raw observation 유지
- task-specific degradation metric
- matched random control
- multiple perturbation operator
- seed 반복
- bootstrap uncertainty
- operator agreement
- region/sample/class/dataset aggregation
- preprocessing validation
- mask-area sanity check
- cache/resume
- provenance
- raw dump
- reproducible report

따라서 가장 설득력 있는 대비는 다음이다.

> **Occlusion primitive vs. reproducible intervention audit protocol/software**

---

# 11. RobustBench / RobustCheck와의 차이

RobustBench나 RobustCheck의 중심 질문은 대체로 다음과 같다.

> 이 모델은 특정 corruption / attack 조건에 얼마나 robust한가?

반면 본 프로젝트의 중심 질문은 다음과 같다.

> 이 모델은 이 데이터셋의 각 샘플에서 어느 사용자 정의 공간 또는 의미 영역의 제거에 민감하며, 그 민감도는 데이터셋 전체에서 어떤 패턴을 이루는가?

즉 평가 차원이 다르다.

### Robustness benchmark

```text
global perturbation
  ↓
dataset-level performance
```

### Spatial Sensitivity Audit

```text
sample
  ×
user-defined local region
  ×
intervention
  ↓
task response
  ↓
sample / region / class / dataset aggregation
```

따라서 본 도구는 전통적인 robustness benchmark보다

> **localized behavioral audit**

에 가깝다.

---

# 12. Data-quality tool과의 차이

FiftyOne Brain이나 Deepchecks와 같은 계열은 다음 질문을 주로 다룬다.

- 어떤 sample이 어렵나?
- label이 이상한가?
- sample이 unique한가?
- model이 자주 틀리는 slice가 있는가?
- annotation이 의심스러운가?

즉 **data 자체 또는 prediction distribution을 신호로 사용**하는 경우가 많다.

반면 본 프로젝트는 다음을 측정한다.

> 정상 sample에 특정 intervention을 가했을 때 특정 model의 task behavior가 어떻게 바뀌는가?

따라서 이 프로젝트는 데이터 품질 도구가 아니다.

### Data-quality

```text
sample / label / prediction
  ↓
데이터 이상 여부
```

### Spatial Sensitivity Audit

```text
sample
  ↓
intentional intervention
  ↓
same model 재실행
  ↓
task behavior change
```

이 때문에 현재 기획서가 “dataset audit”보다 **model-conditioned spatial sensitivity**라는 용어를 사용하는 방향은 적절하다.

---

# 13. 가장 적절한 프로젝트 포지셔닝

현재 프로젝트를 단순히

> Spatial Perturbation Tool

이라고 정의하는 것은 약하다.

더 적절한 표현은 다음과 같다.

> **Dataset-scale, task-conditioned intervention audit framework**

한국어로 풀면:

> 사용자 정의 입력 영역에 반복적으로 개입하고, 그 결과 나타나는 downstream task 성능 변화를 측정하며, 그 변화가 단순한 perturbation artifact인지 반복 가능한 sensitivity evidence인지 control/stability analysis를 통해 검증하고, 이를 재현 가능한 audit result로 만드는 연구 소프트웨어

정도로 정의할 수 있다.

---

# 14. 핵심 키워드 세 가지

## 14.1 Intervention

단순히 saliency score를 읽는 것이 아니다.

실제 입력을 바꾸고 모델을 다시 실행한다.

```text
input
  ↓
region intervention
  ↓
model re-inference
  ↓
task change observation
```

## 14.2 Task-conditioned

평가 대상은 일반적인 perceptual difference가 아니다.

실제 downstream metric 변화다.

예:

- classification margin
- top-1 correctness
- loss
- confidence
- detection miss
- IoU
- recall

즉 “픽셀이 많이 바뀌었다”가 아니라,

> **그 변화가 실제 task에 어떤 영향을 주었는가**

를 본다.

## 14.3 Audit

한 번 perturbation해서 큰 수치가 나왔다고 끝내지 않는다.

다음과 같은 검사를 통해 그 결과를 어느 정도 믿을 수 있는지 확인한다.

- matched control
- multiple operators
- seed variation
- bootstrap CI
- operator consistency
- effective area
- preprocessing consistency
- deterministic pipeline

따라서 “audit”이라는 단어가 프로젝트의 성격을 잘 나타낸다.

---

# 15. 프로젝트의 핵심 기여를 세 가지로 압축

## Contribution 1. Generalization of an experimental pattern

이전 Action Recognition 연구에서 사용한 intervention experiment를 일반화한다.

즉 임의의

```text
model
×
dataset
×
region
×
intervention
×
task metric
```

조합에 대해 dataset-scale experiment를 반복 수행할 수 있게 한다.

## Contribution 2. Reliability-aware intervention analysis

단순 perturbation response를 그대로 sensitivity로 간주하지 않는다.

다음과 같은 검사를 적용한다.

- matched control
- operator reproduction
- repeated seeds
- bootstrap uncertainty
- preprocessing sanity check
- mask-area validation

이를 통해

> “얼마나 많이 변했는가?”

뿐 아니라

> “이 변화를 얼마나 믿을 수 있는가?”

를 함께 평가한다.

## Contribution 3. Reproducible audit workflow

실험의 각 단계를 명확하게 분리한다.

```text
Raw Observation
    ↓
Metrics
    ↓
Control / Stability Analysis
    ↓
Multi-level Aggregation
    ↓
Report
```

또한 provenance를 남긴다.

이를 통해:

- 새로운 metric 재계산
- analysis logic 수정
- third-party reproduction
- experiment rerun
- version comparison

이 가능해진다.

---

# 16. 가장 중요한 비교 실험: Captum 기반 reference workflow

현재 프로젝트에서 가장 우선순위가 높은 추가 실험 중 하나다.

목적:

> “기존 도구로 원리적으로 가능하지만, 동일한 audit을 하려면 사용자가 실제로 얼마나 많은 것을 구현해야 하는가?”

를 보여준다.

## 16.1 Reference implementation에서 사용자가 해야 할 일

예를 들어 Captum FeatureAblation/Occlusion을 이용해 현재 4×4 grid experiment와 동일한 분석을 한다고 가정한다.

사용자가 직접 구현해야 할 가능성이 높은 부분:

1. model wrapper
2. dataset iteration
3. region mask generation
4. 여러 fill strategy 반복
5. output → task metric 변환
6. per-sample serialization
7. matched random control
8. control normalization
9. seed 반복
10. bootstrap
11. operator consistency
12. sample aggregation
13. region aggregation
14. class aggregation
15. dataset aggregation
16. preprocessing validation
17. mask-area validation
18. configuration 저장
19. provenance 저장
20. cache
21. resume
22. report generation

Captum 자체는 이 가운데 가장 핵심적인 low-level primitive인

> region ablation + model output difference

를 매우 잘 수행한다.

그러나 audit pipeline 전체는 사용자가 구성해야 한다.

## 16.2 비교표 예시

| 항목 | Captum 기반 custom workflow | SSAT |
|---|---|---|
| Region ablation | 제공 | 제공 |
| Dataset 반복 실행 | 사용자 구현 | 내장 |
| Custom task metric | 사용자 구현 | interface |
| Matched control | 사용자 구현 | 내장 |
| Multi-fill stability | 사용자 구현 | 내장 |
| Seed repeat | 사용자 구현 | 내장 |
| Bootstrap | 사용자 구현 | 내장 |
| Multi-level aggregation | 사용자 구현 | 내장 |
| Raw schema | 사용자 설계 | 표준 |
| Cache | 사용자 구현 | 내장 |
| Resume | 사용자 구현 | 내장 |
| Provenance | 사용자 구현 | 자동 |
| Report | 사용자 구현 | 자동 |
| Area sanity check | 사용자 구현 | 내장 예정 |
| Preprocessing validation | 사용자 구현 | 내장 예정 |

## 16.3 비교할 수 있는 정량 항목

단순 LOC만 보는 것보다 다음을 함께 비교하는 것이 좋다.

### 사용자 코드량

- reference implementation LOC
- SSAT config LOC
- required glue-code LOC

### 실행 단계

#### Captum workflow

```text
prepare mask
→ write evaluation loop
→ save raw results
→ run control
→ run seeds
→ aggregate
→ statistics
→ report
```

#### SSAT

```bash
ssat run audit.yaml
ssat analyze <run>
ssat report <run>
```

### 사용자가 결정해야 하는 항목

예:

- result schema
- aggregation convention
- degradation sign
- control definition
- confidence interval
- reproducibility metadata
- retry policy
- resume handling

### 재현성

- config 하나로 전체 재현 가능 여부
- intermediate raw result 보존 여부
- metric만 바꿔 다시 계산 가능한지
- provenance 자동 저장 여부

---

# 17. 이 비교 실험이 중요한 이유

현재 프로젝트의 차별성을 가장 직접적으로 입증할 수 있기 때문이다.

약한 주장:

> 기존 도구에는 이런 기능이 없다.

보다 좋은 주장:

> 기존 low-level attribution library를 사용해 동일한 분석을 수행하는 것은 원리적으로 가능하다. 그러나 dataset-scale audit을 구성하려면 반복 실행, task metric normalization, control generation, stability analysis, result schema, aggregation, provenance 및 reporting을 사용자가 직접 구현해야 한다. 본 도구는 이 workflow를 표준화된 하나의 pipeline으로 제공한다.

이 주장은 과장이 적고 SoftwareX의 software contribution에 잘 맞는다.

---

# 18. 현재 프로젝트에 필요한 실제 데이터 사례

Synthetic validation만으로 correctness는 보여줄 수 있지만, 연구 소프트웨어의 실용성을 보이려면 공개 dataset 사례가 필요하다.

권장 구조:

```text
공개 dataset 1개
+
서로 다른 model 2개 이상
```

예:

```text
Model A
Model B
  ↓
동일한 dataset
  ↓
동일한 regions
  ↓
동일한 perturbation protocol
  ↓
민감도 구조 비교
```

보여줄 수 있는 결과:

- 동일 accuracy지만 sensitivity profile은 다름
- 특정 model이 일부 region에 더 집중
- 특정 class에서 sensitivity 증가
- top vulnerable sample의 차이
- control 대비 의미 있는 region 존재
- operator stability 차이

여기서 반드시 “놀라운 발견”이 나올 필요는 없다.

SoftwareX 관점에서는

> 본 도구를 사용하여 실제 공개 dataset과 실제 model을 reproducibly audit할 수 있음

을 보여주는 것 자체가 중요하다.

---

# 19. 제출 전 우선순위

## 19.1 거의 필수로 권장

1. **기존 도구 대비 reference workflow 비교**
2. **실제 공개 dataset 사례**
3. **패키징**
4. **End-to-end example**
5. **자동 테스트 + CI**
6. **preprocessing / effective area sanity check**
7. **Regression test**
8. **Tagged release**
9. **작은 software benchmark**

### 패키징 권장

- `pyproject.toml`
- clean install
- versioned package
- reproducible dependency

### End-to-end example 권장

```bash
ssat run examples/classification_grid.yaml
```

하나로 작은 demo가 끝나는 형태가 이상적이다.

### 자동 테스트 권장 범위

- region generation
- perturbation
- metrics
- serialization
- aggregation
- resume
- end-to-end small run

### sanity check 권장 예시

```text
Region area consistency: PASS
Preprocessing deterministic: PASS
Mask coordinates after transform: PASS
```

### 작은 benchmark 권장 항목

- runtime
- samples/sec
- peak memory
- raw dump size
- cache/resume 효과

---

# 20. 있으면 강하지만 필수는 아닌 것

- classification 외 task 1개
- detection adapter
- Action Recognition example
- semantic body-part example
- interactive HTML
- model comparison mode
- coarse-to-fine region scan

이 기능들은 프로젝트 매력을 높이지만, 현재 프로젝트의 SoftwareX 제출 가능성을 결정하는 핵심은 아니다.

---

# 21. 현시점에서 우선순위가 낮은 기능

다음 기능을 추가하느라 핵심 검증이 늦어지는 것은 권장하지 않는다.

- GUI
- web dashboard
- server mode
- leaderboard
- 수십 개의 perturbation
- built-in model zoo
- 모든 vision task 지원
- automatic shortcut detector
- 새로운 attribution algorithm
- 완전 자동 hyperparameter tuning

---

# 22. 현재 상태의 객관적 평가

현재 프로젝트는 단순 prototype 단계만은 아니다.

## 강점

- CLI 기반 end-to-end pipeline
- configurable perturbation
- result dump
- metrics / analysis 분리
- report generation
- matched control
- stability analysis
- multiple fill strategy
- synthetic shortcut validation
- negative control model
- preprocessing confound 발견 및 수정
- crop-free rerun
- provenance 지향 설계
- multi-level aggregation 계획
- cache/resume 계획 또는 구현

## 현재 부족한 부분

SoftwareX 제출 관점에서 특히 중요한 공백:

1. **실제 공개 dataset 사례**
2. **기존 도구 조합 대비 workflow comparison**
3. clean install / package maturity 확인
4. automated tests / CI 수준 확정
5. preprocessing/area sanity check의 정식 기능화
6. regression test
7. runtime/storage benchmark
8. tagged release와 documentation 정리

---

# 23. 현재 프로젝트의 가장 큰 위험

가장 큰 위험은 기능 부족이 아니다.

다음 질문에 심사자가 쉽게 답하지 못하는 것이 가장 위험하다.

> “왜 Captum + custom script로 하지 않고 이 별도 software package가 필요한가?”

따라서 논문의 중심은 기능을 더 추가하는 것이 아니라,

> **researcher engineering burden를 얼마나 줄이는가**

를 명확하게 보여줘야 한다.

---

# 24. 논문에서 피해야 할 주장

## 사용하지 않는 것이 좋은 표현

- “새로운 attribution algorithm”
- “기존 도구로는 불가능”
- “shortcut learning을 자동 탐지”
- “causal explanation”
- “모델이 실제로 어디를 본 것인지 증명”
- “synthetic occlusion만으로 deployment failure를 예측”
- “모든 perturbation response는 의미 있는 signal”

---

# 25. 권장 표현

- “standardized intervention audit workflow”
- “dataset-scale region intervention”
- “task-conditioned spatial sensitivity”
- “control-aware sensitivity analysis”
- “stability-aware interpretation”
- “reproducible raw observation and analysis pipeline”
- “prioritization of candidates for follow-up investigation”
- “model-conditioned sensitivity”
- “low-level perturbation primitives are available in existing libraries; this work integrates them into a reproducible audit workflow”

---

# 26. 프로젝트의 권장 한 문장 정의

> **Spatial Sensitivity Audit Toolkit은 사용자 정의 입력 영역에 대해 반복적인 intervention을 수행하고 downstream task 변화를 측정하며, matched control과 stability analysis를 통해 그 변화의 신뢰성을 평가한 뒤 sample·region·class·dataset 수준으로 집계하고 재현 가능한 결과로 저장하는 연구 소프트웨어다.**

---

# 27. 조금 더 학술적인 포지셔닝

영문:

> **A dataset-scale, task-conditioned intervention audit framework for measuring and validating model sensitivity to user-defined spatial or semantic input regions.**

조금 더 SoftwareX다운 표현:

> **A reusable research software pipeline that generalizes region-level perturbation experiments into dataset-scale audits with task-specific metrics, matched controls, stability analysis, multi-level aggregation, raw observation preservation, and reproducible reporting.**

---

# 28. 프로젝트 contribution의 최종 권장 구조

## Contribution 1 — Experimental Generalization

특정 Action Recognition 실험에 사용하던 perturbation workflow를 범용 framework로 일반화.

## Contribution 2 — Reliability-aware Analysis

단순한 perturbation response를 그대로 해석하지 않고 control과 stability를 통해 신뢰성을 함께 평가.

## Contribution 3 — Reproducible Research Software

raw observation, metrics, analysis, reporting을 분리하고 provenance를 보존하여 반복 사용 가능한 연구 infrastructure로 제공.

---

# 29. SoftwareX 제출 전 최종 체크리스트

## Statement of Need

- [ ] 문제를 한 문장으로 설명 가능
- [ ] 기존 도구 대비표 완성
- [ ] Captum/plain PyTorch reference implementation
- [ ] user workflow complexity 비교
- [ ] build-vs-use-existing 설명 가능

## Functionality

- [ ] model adapter
- [ ] task metric
- [ ] region generator
- [ ] perturbation operator
- [ ] matched control
- [ ] stability analysis
- [ ] aggregation
- [ ] raw dump
- [ ] report
- [ ] provenance
- [ ] cache/resume

## Validation

- [x] synthetic shortcut
- [x] negative control
- [x] multiple fills
- [x] matched control
- [x] generalization-gap linkage
- [ ] real-world dataset
- [ ] model comparison
- [ ] regression test
- [ ] preprocessing sanity check
- [ ] effective-area sanity check

## Software Quality

- [ ] clean install
- [ ] pyproject/package
- [ ] unit tests
- [ ] integration tests
- [ ] CI
- [ ] example
- [ ] tutorial
- [ ] API docs
- [ ] error handling
- [ ] versioned release
- [ ] license
- [ ] citation

## Reproducibility

- [ ] config 저장
- [ ] random seed 저장
- [ ] preprocessing 저장
- [ ] model/checkpoint 저장
- [ ] code version 저장
- [ ] raw result 저장
- [ ] metric version 저장
- [ ] report와 raw data 연결
- [ ] 작은 reproducible demo

## Performance

- [ ] runtime
- [ ] throughput
- [ ] memory
- [ ] raw dump size
- [ ] cache effect
- [ ] resume behavior

---

# 30. 최종 판단

현재 프로젝트는 SoftwareX에 맞지 않는 프로젝트라기보다, **이미 SoftwareX에 적합한 형태의 research software로 상당 부분 발전했지만 그 필요성과 재사용 가치를 객관적으로 증명하는 비교가 아직 부족한 상태**에 가깝다.

특히 다음은 이미 좋은 기반이다.

- 이전 연구에서 실제로 반복 구현해야 했던 intervention experiment를 일반화했다는 명확한 origin
- task-conditioned sensitivity라는 비교적 명확한 문제 설정
- matched control
- multi-operator stability
- raw observation preservation
- multi-level aggregation
- synthetic ground-truth validation
- normal-model negative control
- 실제 preprocessing confound 발견 사례

따라서 향후 작업의 방향은 새로운 기능을 계속 추가하는 것보다는 다음에 집중하는 것이 좋다.

```text
1. Existing-tool reference workflow comparison
2. Real public-dataset case study
3. Software packaging / testing / CI
4. Sanity checks formalization
5. Reproducibility demonstration
6. Small performance benchmark
```

이 여섯 가지가 채워지면 다음 질문에 상당히 강하게 답할 수 있다.

> 왜 이 소프트웨어가 필요한가?

답은 다음과 같이 정리할 수 있다.

> **기존 라이브러리도 개별 region perturbation 자체는 수행할 수 있다. 그러나 연구자가 dataset-scale task-conditioned intervention audit을 수행하려면 반복 실행, task metric normalization, matched control, stability analysis, statistical validation, multi-level aggregation, raw schema, cache/resume, provenance 및 reporting을 직접 설계하고 구현해야 한다. 본 소프트웨어는 이 반복적인 연구 workflow를 표준화하고 재현 가능한 하나의 pipeline으로 제공한다.**

이 문장이 현재 프로젝트의 SoftwareX 포지셔닝을 가장 명확하게 요약한다.

---

# 31. 참고 사례 및 조사 방향

본 문서 정리 과정에서 비교 대상으로 검토한 대표 소프트웨어/논문 계열:

- SoftwareX — RobustCheck
- SoftwareX — CleanAI
- SoftwareX — Backdoor Pony
- NeurIPS Datasets and Benchmarks — RobustBench
- Captum — Occlusion / FeatureAblation
- FiftyOne Brain
- JOSS review checklist

논문 작성 시에는 최종 제출 직전에 각 프로젝트의 최신 기능, release 상태, documentation, citation 정보를 다시 확인하는 것이 좋다.
