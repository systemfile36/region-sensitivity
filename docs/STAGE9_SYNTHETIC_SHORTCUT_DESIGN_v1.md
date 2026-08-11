# 단계 9. L3 합성 Shortcut 실험 설계서 (v1)

`docs/METRIC_ENGINE_DESIGN_v1.md` §3.4-3.5, `docs/IMPLE_PLAN_METRIC_DESIGN_v1.md` §5 단계 9가 남겨둔 잔여 결정 사항(데이터셋·모델·패치·판정 임계)을 확정한 실험 설계서다. 이 문서의 값은 실험 착수 전 사전 등록된 것으로 취급하며, 결과를 본 뒤 변경하지 않는다. 이후 `experiments/synthetic_shortcut/`의 모든 구현은 이 문서를 그대로 따른다.

## 1. 데이터셋

- **CIFAR-10** (`torchvision.datasets.CIFAR10`), 10 클래스, 원본 해상도 32×32 그대로 사용(리사이즈 없이 저장).
  - 공개·표준 소형 분류 데이터셋 중 가장 널리 쓰이는 계열이며, 클래스 수가 적고 학습이 빠르다는 설계 조건을 그대로 만족한다.
- 규모는 샘플링으로 축소: 학습 3,000장/클래스(총 30,000), 표준 test split 10,000장은 그대로 유지.
- 저장 형식: 기존 `image_manifest` source(quickstart와 동일 경로)로 PNG화하여 저장.

## 2. 모델 및 학습

- **squeezenet1_0**(torchvision), `weights=None`(scratch), `model_kwargs={"num_classes": 10}`.
- 하이퍼파라미터(표준 CIFAR 스크래치 레시피): SGD(momentum=0.9), lr=0.1 + cosine annealing, weight_decay=5e-4, batch_size=128, epoch=40.
- **증강 제약**: RandomCrop/RandomHorizontalFlip 등 위치를 흔드는 증강은 금지한다 — 패치가 "고정 위치"라는 전제 자체를 깨뜨리기 때문이다. Normalize만 적용.
- **전처리 정합성 (코드로 확인한 제약)**: `TorchvisionAdapter`는 `weights=None`이어도 항상 `weights_enum.DEFAULT.transforms()`(squeezenet1_0 기준 Resize(256)→CenterCrop(224)→ImageNet Normalize)를 오디트 시점 전처리로 쓰며, 이 값은 설정으로 오버라이드할 수 없다(`ssat/core/adapter/torchvision_adapter.py:135-138`). 학습 스크립트도 동일한 Resize(256)→CenterCrop(224)→Normalize 파이프라인을 그대로 재현해야 학습/감사 시점 전처리 불일치를 피할 수 있다.
- checkpoint는 `{"model": state_dict}` 형식으로 저장하고 `TorchvisionProviderConfig(checkpoint=CheckpointConfig(path=..., state_dict_key="model"))`로 로드한다(기존에 이미 지원되는 경로, 신규 어댑터 코드 불필요).
- **구현 후 발견된 학습 안정성 문제로 인한 수정 (실측값, 결과 판정과 무관한 파이프라인 보정)**: 원안의 `lr=0.1`은 BatchNorm이 없는 squeezenet1_0을 scratch로 학습할 때 재현적으로 dead-ReLU 붕괴(모든 ReLU 출력이 0으로 굳어 `loss`가 `ln(10)≈2.3026`, `acc=0.10`에 고정)를 일으켰다. gradient clipping(`max_norm=5.0`)만으로는 막지 못했고, LR warmup(3 epoch)을 추가해도 warmup 종료 후 peak lr(0.1)에 도달하는 순간 다시 붕괴함을 실측으로 확인했다 — 즉 문제는 "0에서 튀는 것"이 아니라 0.1이라는 값 자체의 지속 불안정성이었다. 이에 따라 **`lr`을 0.1 → 0.01로 하향**하고, 3 epoch 선형 warmup + gradient clipping은 유지했다. 10 epoch 실측 검증(A_train 전체, 30,000장)에서 `loss`가 `2.30 → 1.46 → 0.25 → ... → 0.0005`로, `acc`가 `0.10 → 0.50 → 0.95 → ... → 0.9999`로 정상 수렴함을 확인했다. 이 값들은 Q1~Q5 판정 임계(§5)가 아닌 학습 레시피 안정화이므로, "결과를 본 뒤 변경하지 않는다"는 원칙이 적용되는 사전 등록값과는 성격이 다르다고 판단해 반영했다.

## 3. 합성 패치 및 grid 설계

- grid: `rows=4, cols=4`(16 region), 32×32 기준 cell 8×8px.
- 패치: 클래스별 solid-color 8×8 정사각형(=cell 전체를 채움). 10개 클래스에 HSV hue를 10등분해 고유 색 할당.
- 위치: 항상 (row=0, col=0) 셀(좌상단) 고정, 이미지당 패치 1개.
- CenterCrop(224/256 ≈ 87.5% 유지)로 가장자리 일부가 트리밍되므로, 패치를 코너 점이 아니라 **cell 전체**를 채우도록 하여 크롭 후에도 신호가 남도록 한다.

## 4. A / B / C 데이터셋 구성

| 데이터셋 | 구성 | 용도 |
|---|---|---|
| C (무패치) | CIFAR-10 원본 | M_normal 학습 + Q5 공통 평가셋 |
| A (오염) | 모든 이미지에 정답 클래스 매핑 색 패치를 (0,0) 셀에 삽입 | M_shortcut 학습 + Q1~Q4 감사 대상 |
| B (무관) | A와 동일한 색상표를 쓰되 패치를 16개 셀 중 무작위 위치에 배치(정답과 무상관) | 보조 대조 확인 |

- B는 설계서가 정의만 하고 Q1~Q5에 명시적으로 배정하지 않았다. 이 문서에서는 M_shortcut을 B로 감사했을 때 고정 위치 (0,0) region이 더 이상 1위가 아님을 확인하는 **보조 대조 확인**(정식 Q1~Q5 판정에는 포함하지 않음)으로 쓴다.
- 감사(오디트) 표본: A/B 각각 클래스당 20장(총 200장). Q5는 C의 표준 test split(10,000장)에서 클래스당 200장(총 2,000장) 서브샘플.

## 5. Q1~Q5 판정 기준 (사전 등록값)

사용 지표: `margin_drop`(툴체인 기본 primary metric과 동일, `ssat/metrics/aggregate.py:48`).

| 질문 | 사전 등록 임계 |
|---|---|
| Q1 | 패치 region(0,0)이 16개 region 중 평균 `margin_drop` **1위** |
| Q2 | 패치 region 평균 degradation ≥ 비패치 15개 region 평균의 **3배** |
| Q3 | M_normal에서는 패치 위치 (0,0) region이 1위가 **아님** |
| Q4 | Q1이 §6의 fill strategy 중 **최소 2개**에서 재현 |
| Q5 | M_shortcut의 (A 정확도 − C 정확도) 하락폭이 M_normal의 (A 정확도 − C 정확도) 하락폭보다 **10%p 이상** 큼 |

## 6. §3.5 마스크 방식 민감도 축소판

- 5개 fill strategy 전부 실행(`constant_fill`/`mean_fill`/`blur`/`gaussian_noise`/`patch_shuffle`) — 코어에 이미 구현되어 있어 추가 비용이 낮다.
- `constant_fill`을 기준으로 나머지 4개와의 region 순위 상관을 **Spearman**으로 각각 보고.

## 7. 실행 구조 및 재현성

구현 과정에서 §1-6의 값은 그대로 두고 실행 구조만 다음과 같이 구체화했다(정적 YAML 7개 대신 config를 만드는 함수 하나, fill strategy별 별도 dump 등 — 근거는 각 스크립트의 모듈 독스트링 참고):

```
experiments/synthetic_shortcut/
├── common.py              # grid 상수, 클래스 색상 팔레트, patch 합성, manifest I/O
├── prepare_data.py         # CIFAR-10 다운로드/샘플링, A/B/C 패치 합성, image_manifest 생성
├── train.py                 # M_shortcut / M_normal 학습, checkpoint 저장
├── run_audit.py               # 7개 (model, dataset, fill) 조합을 ssat.application.AuditApplication으로
│                                 실행하고 margin_drop 지표를 계산·저장
├── evaluate_accuracy.py        # Q5용 단순 top-1 정확도 (A-test/C-test × 두 모델, ssat 파이프라인 미사용)
├── evaluate.py                  # Q1~Q5 판정, §3.5 Spearman 상관, report.md/CSV/히트맵 산출
├── thresholds.json                # §5 사전 등록값
├── data/                            # (gitignore) CIFAR-10 캐시 + 렌더링된 A/B/C PNG + manifest
├── checkpoints/                      # (gitignore) 학습된 M_shortcut/M_normal 체크포인트
└── results/                            # (gitignore) dumps/metrics/accuracy.json/report.md/히트맵
```

- 정식 audit config는 정적 YAML 파일이 아니라 `run_audit.py::_build_audit_config()`가 조합별로 생성하는 dict다 — `ssat.application.config.load_application_config`가 YAML 경로뿐 아니라 Python dict도 그대로 받아들이기 때문(코드로 확인).
- fill strategy 5종은 각각 별도의 `ssat run` 호출 + 별도 metrics_dir로 분리한다 — `Aggregator`의 `RegionMetrics`/`SpatialProfile` 그룹핑 키가 `(region_key, metric_name)`뿐이라 한 config에 여러 fill을 넣으면 region별 평균이 섞여 §3.5의 전략별 비교가 불가능해지기 때문(코드로 확인).
- 시드: 데이터 샘플링 / B의 무작위 패치 배치는 고정 시드(기본 42)를 사용, 모델 초기화·학습은 torch 기본 RNG 상태에 맡긴다(재현이 필요하면 `train.py`에 시드 인자를 추가로 넘길 수 있음).
- pytest collection에는 포함하지 않는다(설계서 §3.4).

## 8. 산출물

- `results/region_metrics_{run_id}.csv` — `run_id = {model}_{dataset}_{fill}`
- `results/accuracy.json` — Q5용 top-1 정확도 (모델 × A/C)
- `results/report.md` — Q1~Q5 판정 결과 요약 표 + §3.5 상관 표
- `results/heatmaps/{run_id}/` — DebugViz V2 히트맵(M_shortcut/M_normal/B대조 각각 대표 샘플) — 기존 `ssat.metrics.viz.heatmap.save_heatmap_views` 재사용

## 9. 실패 시 처리

설계서 §3.4의 4단계 절차를 그대로 따른다: L2 통과 확인 → DebugViz 마스크 검증 → shortcut 의존성 직접 확인(패치 제거 시 정확도 하락 확인) → 도구 한계로 기록.

## 실험 실행 과정 

컨테이너 안에서 

```bash
cd experiments/synthetic_shortcut

# 1. CIFAR-10 다운로드 + A/B/C 데이터셋 생성 (원본 tar.gz는 이미 캐시되어 있어 재다운로드 없음)
python3 prepare_data.py

# 2. 두 모델 학습 (각각 SGD lr=0.01 cosine, 40 epoch, RTX 4090 기준 수십 분 내외 예상)
python3 train.py --dataset shortcut --warmup-epochs 10
python3 train.py --dataset normal --warmup-epochs 10

# 3. 7개 감사 조합 실행 (dump + margin_drop 지표 계산·저장)
python3 run_audit.py

# 4. Q5용 정확도 계산
python3 evaluate_accuracy.py

# 5. Q1~Q5 판정 + §3.5 상관 + report.md/CSV/히트맵 산출
python3 evaluate.py
```