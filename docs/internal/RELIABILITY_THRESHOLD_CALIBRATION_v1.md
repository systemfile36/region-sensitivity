# 신뢰도 임계값 재조정 결과 (v1)

`docs/IMPLE_PLAN_CONTROL_STABILITY_v1.md` §8 / `docs/CONTROL_STABILITY_DESIGN_v1.md`
§5 "등급 판정 임계 구체값 | B3 재분석 결과를 보고 조정. 단, 조정 후 재보고"
절차의 최종 재보고. 대상은 `ssat/analysis/reliability.py`의
`DEFAULT_Z_VS_CONTROL_THRESHOLD`(제안값 2.0)와
`DEFAULT_SEED_CV_THRESHOLD`(제안값 0.2)이다.

## 데이터

`run_threshold_validation_full.py` / `validate_reliability_thresholds_full.py`
(crop-free 전처리, 5개 fill 연산자 전부, target당 대조군 2개, 시드 3개,
샘플 200개 → item 144,000개). 이전 세션의 `run_threshold_validation.py`
(constant_fill 단일 연산자, crop 있는 프리셋)의 한계 두 가지를 모두
해소한 최종 실측이다: 결과는
`experiments/synthetic_shortcut/results_crop_free/threshold_validation_report_full.md`.

## 결론: **두 임계값 모두 제안값(2.0 / 0.2)을 조정하지 않는다.**

실측 분포를 근거로 임의의 새 숫자를 고르는 대신, 두 통계량 모두 그 숫자
자체를 신뢰하기 전에 먼저 해소해야 할 구조적 불안정성이 이번 실측에서
드러났기 때문이다. 이는 사전 등록 원칙(결과를 본 뒤 임계값만 조정하지
않는다)의 연장으로, "재조정하지 않음"도 §8 절차가 명시적으로 허용하는
유효한 결과다.

### z_vs_control — 분모(대조군 표준편차) 불안정

| | n | min | median | max |
|---|---|---|---|---|
| patch region | 999 | 1.820 | 134.758 | 57,946,990.625 |
| 비패치 15개 region | 14,986 | −5,027,495.000 | −0.997 | 266,924.278 |

`z_vs_control = (target_mean - control_mean) / control_std`
(`ssat/analysis/control.py`)이므로 `control_std`가 0에 가까운 anchor에서
값이 폭발한다. 이는 설계서가 이미 경고했던 바로 그 현상이며(§4/§9 "분모가
0에 가까울 때 불안정"), 이번 실측으로 **이론적 우려가 아니라 실제로
5천만 단위까지 발산함이 확인**되었다. 이 상태에서 median(-0.997)이나
임의의 백분위수를 근거로 새 임계값을 고르면, 그 값은 극단치 몇 개에
좌우된 숫자이지 "대조군과 유의하게 다른가"라는 원래 질문에 대한 답이
아니다. 비패치 anchor의 11.3%가 임계 2.0을 초과하는데, 이 비율 자체도
같은 발산 때문에 신뢰하기 어렵다.

**권고 (이번 범위 밖)**: `z_vs_control`의 분모에 최소 표준편차 바닥값을
두거나, `control_std`가 일정 기준 이하인 anchor를 `control_available`에서
`FALSE`/`UNAVAILABLE` 처리하는 등 통계량 자체의 안정화가 먼저 필요하다.
안정화 이후에 다시 이 절차를 밟아야 한다.

### seed_cv — 동일한 구조의 문제, 그리고 대조군의 실제 변동성

| | n | min | median | max | 임계 0.2 미만(seed_stable=TRUE) |
|---|---|---|---|---|---|
| patch region | 1,000 | 0.000 | 0.000 | 0.658 | — |
| 비패치 target region | 12,918 | 0.000 | 0.002 | 434.311 | — |
| control anchor | 31,655 | 0.000 | 1.412 | 3,365.332 | 510/31,655 (1.6%) |

`seed_cv = std/mean`이므로 target region처럼 `mean`이 0에 가까운 (거의
결정적인) anchor에서 마찬가지로 발산한다(max 434). target region 자체는
중앙값이 0.002로 예상대로 안정적이다(마스크가 고정이므로).

문제는 control anchor 쪽이다: **대조군은 seed_salt마다 위치가 다시
뽑히므로 원래도 변동이 커야 하는 게 정상**이지만(Fix 2의 설계 의도),
median 1.412는 제안 임계 0.2의 7배이며 **대조군의 98.4%가 임계를
넘는다**. 지난 세션의 constant_fill 단일 연산자 provisional run에서는
control_cv 중앙값이 0.692였다(임계 미만 13.5%) — 이번 5-연산자 실측에서
중앙값이 그때보다 2배 이상 커졌다. 이는 (a) 대조군 anchor에도 위 z와
동일한 분모(≈0) 불안정이 섞여 있고, (b) 설령 그 불안정을 걷어내더라도
"대조군은 원래 위치가 바뀌므로 불안정해야 정상"이라는 설계 의도상
seed_cv < 0.2라는 임계 자체가 대조군에는 애초에 안 맞는 기준일 수 있음을
시사한다 — target region과 control anchor가 같은 `seed_cv` 임계를
공유하는 것이 맞는 설계인지 자체가 재검토 대상이다.

**권고 (이번 범위 밖)**: seed_cv도 분모 안정화가 선행되어야 하고, 나아가
target region용 임계와 control anchor용 임계를 분리할지(같은 통계량이
서로 다른 것을 재는 두 용도로 쓰이고 있으므로)를 별도로 결정해야 한다.

## multi_strategy — 뜻밖의 구조적 발견

| | 값 |
|---|---|
| n_strategies 분포 | {5: 3200} (전 anchor 동일) |
| multi_strategy=TRUE | 3200/3200 (100%) |

계획 당시 기대는 "5개 연산자를 다 넣으면 `multi_strategy`가 처음으로
실측 TRUE/FALSE 분포를 가질 것"이었으나, **실제로는 항상 TRUE로 나왔다.**
원인은 통계가 아니라 조합론이다: `_multi_strategy_flag`
(`ssat/analysis/reliability.py`)는 "우세 부호가 연산자 2개 이상에서
재현되는가"를 본다. 부호는 사실상 이진(+/−)이고 연산자 수가 5(홀수)이면,
비둘기집 원리에 의해 5개 중 적어도 3개는 반드시 같은 부호를 갖는다 —
즉 **연산자를 홀수 개, 그것도 5개나 넣은 이번 설계에서는 이 플래그가
데이터와 무관하게 항상 TRUE로 나올 수밖에 없다.** 등급 분포에서 `high`가
268건 나온 것(이전 provisional run은 항상 LOW로 막혀 있었다)은 실제로는
`multi_strategy`가 아니라 `exceeds_control`/`ci_excludes_zero`가
바뀌었기 때문이며, `multi_strategy` 자체는 이번 실측으로도 여전히
의미 있게 검증되지 않았다.

**권고 (이번 범위 밖)**: `multi_strategy`를 실질적으로 검증하려면
연산자 수를 짝수로 하거나(동률 가능), 부호를 {+, −, ≈0} 3분류로 완화해
비둘기집 보장이 깨지도록 설계를 바꿔야 한다.

## 종합

세 통계량(z_vs_control, seed_cv, multi_strategy) 모두 "임계값이 잘못됐다"가
아니라 "임계값을 논하기 전에 먼저 고쳐야 할 것이 있다"는 결론으로
수렴한다. 따라서 이번 재조정에서는 제안값을 그대로 유지하고, 위 세 가지
후속 조치(분모 안정화 2건, multi_strategy 설계 재검토 1건)를 이번
범위 밖 항목으로 남긴다.
