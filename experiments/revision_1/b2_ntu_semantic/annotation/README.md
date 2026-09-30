# B2 annotation guide

One sheet per annotator:
- `annotator_A.csv`, `annotator_B.csv`: one row per NTU RGB+D 60 action;
- `annotators.csv`: one row per annotator.

Fill in five columns per row: `head`, `torso`, `arms`, `hands`, `legs`.
This takes about 1 hour. The full protocol is in `../protocol.md`; a Korean
version of this guide follows below.

## Before you start (blind rules)

- **Do not look at any SSAT output for NTU before your sheet is committed.**
  This covers `experiments/real_dataset_case_study/results/ntu60_*`, the
  HTML reports, the paper's NTU figures or heatmaps, `../summary/`, and
  the other annotator's sheet.
- **Use only the action name** (`action_name`, the official NTU RGB+D 60
  name). Do not watch the videos.
- **Work independently.** Do not discuss ratings with the other annotator
  until both sheets are committed.
- **Record your exposure.** In `annotators.csv`, record whether you have
  seen SSAT's NTU results before (for example, as an author of the paper).

## What to rate

For each action, ask: **to recognize this action from the person's
appearance and motion, how much do I need to see this body part?**

| Rating | Meaning |
|---|---|
| **2** core | Hard to recognize the action without seeing this part. |
| **1** supporting | The part takes part or moves, but the action is still recognizable without it. |
| **0** not involved | The part is not characteristic of this action. |

Body parts (left and right together):

| Column | Covers | NTU joints in SSAT's region box |
|---|---|---|
| `head` | head, face, neck | neck, head, spine-shoulder |
| `torso` | chest, belly, back, hips (the trunk) | spine base / mid, neck, spine-shoulder, both shoulders, both hips |
| `arms` | upper arm and forearm movement (shoulder, elbow, wrist) | shoulder, elbow, wrist, hand, hand tip, thumb |
| `hands` | hands and fingers, and objects held in the hand | wrist, hand, hand tip, thumb |
| `legs` | thigh, knee, lower leg, foot | hip, knee, ankle, foot |

- **Arms vs hands.** SSAT's arm box also contains the hand. Rate each
  column by its own role: `arms` for arm movement or posture, `hands` for
  what the hands and fingers do.
- **Left and right.** Rate a column by the more involved side. For
  example, if either hand is core, `hands` = 2.
- **Held objects** (cup, phone, paper, shoe, hat, glasses) count toward
  `hands`. Rate the body part they touch (for example `head` for putting on
  a hat) by that part's own role.
- **Two-person actions** (`two_person` = 1, A050-A060). Rate the body parts
  of the person performing the named action: the one punching, kicking,
  pushing, pointing, or giving. For symmetric actions (hugging,
  handshaking, walking towards / apart) rate either person.
- **Multiple 2s are allowed**, and so is a row of all 0s. Every cell must
  hold 0, 1, or 2.
- `note` (optional, any language): record uncertainty or reasoning.

## Filling in the sheet

1. Open `annotator_<your id>.csv` in a spreadsheet program (Excel,
   LibreOffice) or a text editor.
2. Do not change the header, the row order, or the first four columns.
3. Save as CSV (UTF-8, comma-separated).
4. Fill in your row of `annotators.csv`: start and finish time, and prior
   exposure.
5. Check the sheet inside the container:

   ```bash
   python experiments/revision_1/b2_ntu_semantic/annotations.py check --annotators A
   ```

   It must report `60/60 rows complete` and no errors.
6. Commit your sheet and `annotators.csv` before anyone runs
   `ssat_part_scores.py`. The commit time is the record that the rating
   was blind. The analysis scripts refuse to run until every sheet is
   complete and committed unchanged.

---

# B2 어노테이션 안내 (한국어)

작성할 파일은 다음과 같습니다.
- `annotator_A.csv`, `annotator_B.csv`: annotator마다 한 파일이며, NTU RGB+D 60의 action 하나가 한 행입니다.
- `annotators.csv`: annotator 정보입니다.

행마다 `head`, `torso`, `arms`, `hands`, `legs` 다섯 칸을 채웁니다. 약 1시간 걸립니다.

## 시작 전 (blind 규칙)

- **본인 시트를 커밋하기 전에는 NTU에 대한 SSAT 결과를 보지 않습니다.** `experiments/real_dataset_case_study/results/ntu60_*`, HTML report, 논문의 NTU 그림과 heatmap, `../summary/`, 다른 annotator의 시트가 모두 해당됩니다.
- **action 이름(`action_name`, NTU RGB+D 60 공식 이름)만 보고 판단합니다.** 영상은 보지 않습니다.
- **각자 독립적으로 작성합니다.** 두 시트가 모두 커밋될 때까지 서로 평점을 상의하지 않습니다.
- **노출 여부를 기록합니다.** SSAT의 NTU 결과를 이전에 본 적이 있는지(예: 논문 저자) `annotators.csv`에 적습니다.

## 채점 기준

각 action에 대해 **"사람의 모습과 움직임만으로 이 동작을 알아보려면 이 신체 부위를 얼마나 봐야 하는가?"**를 판단합니다.

| 점수 | 의미 |
|---|---|
| **2** 핵심 | 이 부위를 보지 않으면 동작을 알아보기 어렵다. |
| **1** 보조 | 부위가 관여하거나 움직이지만, 보지 않아도 동작은 알아볼 수 있다. |
| **0** 무관 | 이 동작의 특징과 관계없다. |

신체 부위 정의(좌우를 합쳐서 봅니다):

| 열 | 범위 | SSAT region box에 들어가는 NTU joint |
|---|---|---|
| `head` | 머리, 얼굴, 목 | neck, head, spine-shoulder |
| `torso` | 가슴, 배, 등, 골반(몸통) | spine base/mid, neck, spine-shoulder, 양 어깨, 양 골반 |
| `arms` | 위팔과 아래팔의 움직임(어깨, 팔꿈치, 손목) | shoulder, elbow, wrist, hand, hand tip, thumb |
| `hands` | 손과 손가락, 손에 든 물체 | wrist, hand, hand tip, thumb |
| `legs` | 허벅지, 무릎, 종아리, 발 | hip, knee, ankle, foot |

- **arms와 hands 구분:** SSAT의 arm box에는 손도 들어 있습니다. 그래도 각 열은 그 부위의 역할로 채점합니다. `arms`는 팔의 움직임이나 자세, `hands`는 손과 손가락이 하는 일입니다.
- **좌우:** 더 많이 관여하는 쪽 기준으로 채점합니다. 예를 들어 어느 한 손이라도 핵심이면 `hands` = 2입니다.
- **손에 든 물체**(컵, 휴대폰, 종이, 신발, 모자, 안경)는 `hands`로 봅니다. 물체가 닿는 부위(예: 모자를 쓸 때의 `head`)는 그 부위 자체의 역할로 채점합니다.
- **2인 동작**(`two_person` = 1, A050–A060): 이름에 나온 동작을 하는 사람(때리는 사람, 차는 사람, 미는 사람, 가리키는 사람, 건네는 사람)의 부위로 채점합니다. 대칭적인 동작(포옹, 악수, 서로 다가가기/멀어지기)은 어느 쪽이든 괜찮습니다.
- **2를 여러 개 줘도 되고, 모두 0이어도 됩니다.** 모든 칸은 0, 1, 2 중 하나여야 합니다.
- `note`(선택, 한국어 가능): 불확실한 점이나 판단 근거를 적습니다.

## 작성 절차

1. `annotator_<본인 id>.csv`를 스프레드시트(Excel, LibreOffice)나 텍스트 편집기로 엽니다.
2. header, 행 순서, 앞의 네 열은 바꾸지 않습니다.
3. CSV(UTF-8, 쉼표 구분)로 저장합니다.
4. `annotators.csv`의 본인 행에 시작·종료 시각과 사전 노출 여부를 채웁니다.
5. container 안에서 검사합니다. `60/60 rows complete`가 나오고 오류가 없어야 합니다.

   ```bash
   python experiments/revision_1/b2_ntu_semantic/annotations.py check --annotators A
   ```

6. 누구든 `ssat_part_scores.py`를 실행하기 전에 본인 시트와 `annotators.csv`를 커밋합니다. 커밋 시각이 blind의 근거가 됩니다. 분석 스크립트는 모든 시트가 완성되어 변경 없이 커밋되기 전에는 실행되지 않습니다.
