파일들을 함께 읽어보니, 레이아웃을 단순히 더 예쁘게 만드는 것보다 **리포트가 답하는 질문의 계층 자체를 재구성하는 것**이 훨씬 중요해 보입니다.

현재 리포트는 `vulnerability_score`와 `reliability_grade`가 서로 독립적인 축이라고 설명하고 있지만,  실제 Region 집계에서는 여러 sample-anchor의 결과를 하나의 worst-case badge로 압축합니다. 그 결과 `r2/c2`처럼 HIGH가 67개 존재하는 위치조차 최종적으로 `UNRELIABLE` 하나로 표시됩니다.  이 표현 방식이 지금 느끼시는 직관성 문제의 핵심이라고 봅니다.

그리고 이는 논문 메모의 최신 해석과도 잘 맞지 않습니다. crop-free 실험에서는 약한 신호 대부분이 사실상 노이즈 수준이고 순위 불안정성이 자연스러운 현상이라는 결론이 나왔습니다.  즉 **“고정된 한 위치에서 모든 샘플이 같은 반응을 보이지 않는다”는 사실 자체가 실패나 신뢰 불가를 의미하지 않습니다.**

## 제가 가장 권하는 구조

핵심적으로 **3단계를 서로 다른 개념으로 분리**하는 것이 좋습니다.

| 수준              | 질문                                   | 표시 방법                                                 |
| --------------- | ------------------------------------ | ----------------------------------------------------- |
| Dataset         | 이 모델은 특정 **고정 위치에 반복적으로 의존하는가?**     | spatial concentration, dominant region share, entropy |
| Region          | 이 위치는 **얼마나 많은 샘플에서 실제 취약 후보가 되는가?** | HIGH/LOW/UNRELIABLE 구성비, top-region 비율                |
| Sample × Region | 이 **특정 샘플에서 이 위치의 효과를 믿을 수 있는가?**    | 기존 reliability grade                                  |

즉 기존 `reliability_grade`를 버릴 필요는 없습니다. **적용 위치를 sample/anchor level로 명확하게 제한**하는 쪽이 좋습니다.

반대로 dataset-level에서는:

> `grid/r0/c0 = UNRELIABLE`

보다는

> `grid/r0/c0`
> Top vulnerable region in 82% of samples
> Reliable-sensitive in 91% of samples
> Spatially dominant

처럼 표현되어야 M_shortcut의 특징이 즉시 드러납니다.

---

# 특히 추가를 권하는 파생 지표

이들은 새로운 모델 추론 없이 **현재 metrics + reliability + sample rankings로 계산할 수 있는 값들**입니다. 따라서 사용자께서 말씀하신 “기존 분석 파일에서 report를 생성한다”는 제약에도 잘 맞습니다. 기획서 역시 sample·region·dataset 수준 집계를 명시적으로 목표로 하고 있습니다. 

**Dominant-region share**가 가장 중요합니다.

[
\max_r \frac{#{i:\operatorname{topRegion}(i)=r}}{N}
]

예를 들어 M_shortcut이면 아마 `r0/c0 ≈ 매우 높은 비율`이 되고, M_normal이면 여러 위치에 퍼질 것입니다.

두 번째로 **Spatial entropy**를 권합니다.

[
H=-\sum_r p_r\log p_r
]

이를 `[0,1]`로 normalize하면,

* `≈ 0`: 거의 하나의 위치에 몰림
* `≈ 1`: 샘플별 top region이 넓게 분산

이 됩니다.

따라서 M_normal은

> **Sample-specific / distributed spatial sensitivity**

M_shortcut은

> **Highly concentrated fixed-location sensitivity**

처럼 한 문장으로 정리할 수 있습니다.

중요한 것은 이것을 **shortcut이라고 자동 판정하지 않는 것**입니다. 기획서도 결과만으로 지름길 학습이나 인과관계를 확정하지 않고 후속 검증 후보를 제공하는 것으로 범위를 제한하고 있습니다. 

---

# 제가 만든 3가지 HTML 시안

모두 **CSS까지 내부에 포함된 완전한 단일 HTML**이라 그대로 브라우저에서 열 수 있습니다.

### A. Interpretation First — 가장 추천

[HTML 시안 A 열기](./report_layout_A_interpretation_first.html)

제가 최종 리포트의 기반으로 가장 추천하는 형태입니다.

순서는:

**한 문장 결론 → baseline → 전체 민감도 → 공간 집중 패턴 → region evidence → 취약 sample → reliability 상세 → provenance**

입니다.

특히 현재의

> “HIGH region이 없습니다.”

를 없애고,

> **“민감도는 존재하지만 한 고정 위치에 집중되지 않습니다.”**

처럼 데이터의 실제 의미를 먼저 보여 줍니다.

Region 표에서도 `UNRELIABLE` 배지 하나 대신:

> HIGH 34% · LOW 12% · UNRELIABLE 54%

처럼 **구성비 자체**를 보여주도록 만들었습니다.

이것만으로도 해석이 상당히 달라집니다.

---

### B. Question Driven — 비전문 사용자에게 가장 쉬움

[HTML 시안 B 열기](./report_layout_B_question_driven.html)

이쪽은 벤치마크 결과표라기보다 **감사 보고서(audit report)**에 가깝습니다.

리포트 전체를 다음 질문으로 구성했습니다.

1. 이 모델은 원래 얼마나 잘 작동하는가?
2. 영역을 가리면 얼마나 흔들리는가?
3. 모든 샘플이 반복적으로 의존하는 고정 위치가 있는가?
4. 그 효과가 대조군보다 강하고 반복 가능한가?
5. 그렇다면 어떤 샘플을 먼저 조사해야 하는가?

개인적으로 **처음 보는 사용자의 이해도는 B가 가장 높을 것**이라고 생각합니다.

대신 논문 부록이나 연구자가 반복해서 보는 정량 분석 보고서로서는 A보다 조금 장황합니다.

---

### C. Behavioral Fingerprint — M_normal vs M_shortcut 차이가 가장 잘 보임

[HTML 시안 C 열기](./report_layout_C_behavioral_fingerprint.html)

사용자께서 설명하신 **M_normal과 M_shortcut의 차이를 직관적으로 보여주는 데 가장 특화된 시안**입니다.

핵심은 모델을

> **Sensitivity magnitude × Spatial concentration**

의 두 축으로 놓는 것입니다.

그러면 이론적으로:

**M_normal**

* Clean accuracy: 낮거나 보통
* Mean degradation: 중간
* top region: 여러 위치에 분산
* dominant-region share: 낮음
* spatial entropy: 높음

→ **분산된 sample-specific sensitivity**

반면 **M_shortcut**

* Clean accuracy: 100%
* Mean degradation: 0.6842
* Z vs control: +7133
* dominant region: `r0/c0`
* top-region frequency: 거의 `r0/c0`로 집중
* entropy: 매우 낮음

→ **강하고 고정 위치에 집중된 sensitivity**

가 됩니다.

사용자께서 원하시는

> “리포트만 보고도 normal과 shortcut의 성격 차이를 알아채고 싶다”

라는 목표에는 이 표현이 매우 잘 맞습니다.

---

# 최종적으로는 A + C 조합을 추천합니다

세 시안 중 하나만 골라야 한다면 **A**입니다.

다만 최종 리포트는 사실상 다음처럼 만드는 것이 가장 좋다고 봅니다.

```text
Header / Run metadata
        ↓
[1] Executive Interpretation
    "민감도는 존재하나 특정 고정 위치에 집중되지 않음"

        ↓
[2] Behavioral Fingerprint       ← C에서 가져옴
    Clean performance
    Mean degradation
    Control separation
    Dominant-region share
    Spatial entropy

        ↓
[3] Dataset Spatial Pattern
    4×4 top-region frequency map
    4×4 reliable-sensitive rate map

        ↓
[4] Region Summary
    mean degradation
    top-region share
    HIGH / MOD / LOW / UNREL distribution
    sign consistency
    N

        ↓
[5] Vulnerable Samples
    image + heatmap
    top region
    vulnerability
    anchor-level reliability
    reliability reason

        ↓
[6] Stability / Controls
    fill strategy correlation
    control comparison
    seed variation
    area uniformity

        ↓
[7] Detailed Tables / Flagged Anchors

        ↓
[8] Provenance / Raw data
```

기획서가 원래 요구하는 것도 **직접 관측 → 제한적 해석 → 후속 검증**의 구분이고 이를 UI에도 반영하겠다고 명시하고 있습니다.  위 구조가 그 원칙과도 가장 자연스럽게 맞습니다.

## 한 가지는 강하게 바꾸는 것이 좋습니다

현재의 **“Region-level worst-case reliability badge”는 제거하는 편을 권합니다.**

현재 리포트 스스로도 sample 카드의 큰 배지가 하나의 UNRELIABLE anchor 때문에 전체 UNRELIABLE이 되는 worst-case rollup임을 설명합니다.  이 방식은 품질보증 시스템의 “하나라도 실패하면 FAIL” 표시에는 적합하지만, **데이터셋에서 모델의 행동 패턴을 이해하려는 분석 보고서에는 지나치게 정보 손실이 큽니다.**

대신:

* `Anchor reliability = HIGH / MODERATE / LOW / UNRELIABLE`
* `Region support = HIGH 34% / LOW 12% / UNREL 54%`
* `Dataset spatial pattern = concentrated / distributed`

처럼 **세 용어 자체를 분리**하는 것이 가장 깔끔합니다.

이렇게 바꾸면 M_normal에서 “모두 UNRELIABLE이라 아무것도 모르겠다”가 아니라,

> **“개별 샘플에서는 확실한 취약 지점들이 관찰되지만, 그 위치가 데이터셋 전체에서 일관되게 반복되지는 않는다.”**

라는 실제로 원하는 해석이 리포트 자체에서 자연스럽게 나오게 됩니다. 그리고 M_shortcut에서는 반대로 `r0/c0` 하나가 빈도·효과·신뢰성·대조군 분리 모두에서 튀어나오므로, 프로그램 목적을 전혀 모르는 사용자에게도 차이가 상당히 명확해질 것입니다.
