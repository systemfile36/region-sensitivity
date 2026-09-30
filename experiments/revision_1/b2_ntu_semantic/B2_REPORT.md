# B2 report: NTU RGB+D semantic validation (Route A)

## Setup

- **Pre-registration:** `protocol.md` and `protocol.json` (commit `8c29dd7`,
  before any annotation).
- **Annotation:** one annotator (A) rated the 5 body-part groups of all 60
  NTU-60 classes as 0 / 1 / 2 from the action names only. The sheet was
  committed (`3076550`) before any SSAT score was computed; the scripts
  enforce this order.
- **SSAT scores:** the stored case-study runs (TSM-R50; 1,200 x-sub test
  videos, 20 per class; margin drop over mean fill, blur, and Gaussian
  noise). There is no new inference.
- **Reproducibility:** all summaries were generated from clean trees: the
  pre-registered outputs at `3076550`, the post hoc breakdown at the next
  commit.
- **Deviations:** D-011 (single annotator, empty annotator metadata, one
  post hoc analysis).

Human relevance is an external semantic reference, not causal ground truth.
Agreement does not show that SSAT is correct, and disagreement does not
show that it is wrong (experiment plan section 9.7).

## 1. Annotation (`consensus.csv`)

- **Coverage.** With one annotator, the consensus is annotator A's
  ratings. Every class has at least one relevant and at least one
  non-relevant group, so all 60 classes are eligible for the primary
  metric. 46 classes have at least one primary (rating 2) group.
- **Relevance per group:**

  | Group | Relevant (>= 1) | Primary (2) |
  |---|---|---|
  | arms | 48 classes | 23 |
  | hands | 48 classes | 23 |
  | torso | 32 classes | 10 |
  | head | 21 classes | 6 |
  | legs | 16 classes | 8 |

- **Agreement.** Annotator agreement (weighted kappa) cannot be computed
  with one annotator.
- **Blind status.** `annotators.csv` was not filled in, so the report does
  not record whether annotator A had seen SSAT's NTU results before (D-011).

## 2. Primary result (`alignment_summary.json`, `alignment_by_class.csv`, `fig_b2_alignment.pdf`)

| Metric (exact run, all samples, lift score) | Value |
|---|---|
| Mean class AUROC, relevant vs non-relevant groups | **0.69** (95 % CI 0.60-0.77, class bootstrap) |
| Permutation p (annotation rows reassigned to classes, 10,000 times) | < 0.001 (no permutation reached 0.69; null mean 0.49, max 0.62) |
| Classes with AUROC > 0.5 / = 0.5 / < 0.5 | 39 / 8 / 13 (25 at 1.0) |
| Top-1 hit rate (highest-lift group annotated primary) | 39 % (18 of 46) vs 30 % chance, p = 0.033 |
| Mean per-class Spearman (ratings vs lift) | 0.32 |

- **SSAT ranks relevant groups above non-relevant ones** more often than
  annotation-shuffled classes would: AUROC 0.69 against a null
  distribution centred at 0.49.
- **The agreement is moderate, not complete.** In 13 of 60 classes, the
  non-relevant groups score higher.
- **The top-1 agreement is weaker.** The single highest group matches a
  primary group in 39 % of classes, against 30 % by chance.

## 3. Robustness (`robustness.csv`)

| Variant | Mean AUROC (95 % CI) | Top-1 hit vs chance (p) | Mean Spearman |
|---|---|---|---|
| Primary (exact, all samples, lift) | 0.69 (0.60-0.77) | 39 % vs 30 % (0.033) | 0.32 |
| Crop-free run | 0.67 (0.59-0.74) | 39 % vs 30 % (0.027) | 0.26 |
| Clean-correct samples only | 0.68 (0.60-0.76) | 41 % vs 30 % (0.022) | 0.31 |
| Raw `S` instead of lift | 0.69 (0.62-0.75) | 57 % vs 30 % (< 0.001) | 0.32 |
| Area-adjusted lift | 0.70 (0.62-0.78) | 46 % vs 30 % (0.006) | 0.34 |
| Single-person classes (A001-A049) | 0.70 (0.60-0.80) | 38 % vs 30 % (0.064) | 0.34 |

All permutation p-values for the mean AUROC are < 0.001.

- **The class AUROC is stable at 0.67-0.70** across both preprocessing
  runs, sample filtering, the area adjustment, and removing the
  two-person classes.
- **The top-1 hit rate depends on the score.** It is higher with raw `S`
  (57 %) than with lift (39 %). For single-person classes only, it is not
  significant (p = 0.064).

## 4. Where the agreement comes from (post hoc, `group_breakdown.csv`; D-011)

For each group, across the 60 classes, the table shows how well its lift
separates the classes that rate it relevant from those that do not.

| Group | Cross-class AUROC (exact / crop-free) | Mean lift, relevant vs not | Classes where it is SSAT's top group | Classes where it is primary |
|---|---|---|---|---|
| head | 0.72 / 0.66 | +0.47 vs -0.25 | 16 | 6 |
| torso | 0.71 / 0.75 | +0.28 vs -0.32 | 10 | 10 |
| arms | 0.83 / 0.79 | +0.21 vs -0.83 | 8 | 23 |
| hands | 0.73 / 0.68 | +0.14 vs -0.57 | 10 | 23 |
| legs | 0.73 / 0.68 | +0.67 vs -0.24 | 16 | 8 |

- **Each group separates relevant from non-relevant classes** (AUROC
  0.66-0.83). The part-level signal is therefore not carried by one group.
- **Lift over-selects head and legs as the top group.** They are the top
  group in 16 classes each, but primary in only 6 and 8.
  - Arms and hands are relevant in 48 of 60 classes. Lift is a z-score
    across classes, so a group that matters for most classes rarely stands
    out in one class. The classes where head or legs are unusually
    sensitive stand out instead.
  - This is why the top-1 hit rate is higher with raw `S` (section 3),
    while the AUROC, which uses all groups, is not affected.
- **Examples:**
  - Leg-dominant classes mostly agree: kicking something, staggering,
    sitting down, standing up, falling, and walking towards each other
    have AUROC 1.0.
  - Fine hand-object classes mostly disagree: reading and writing have
    AUROC 0.17 (SSAT's top groups head and legs); clapping has 0.5.

## 5. Representative cases (pre-registered: top 3 and bottom 3 by AUROC, ties by label id; `fig_b2_examples.pdf`)

- **Top:**
  - A002 eat meal/snack, A003 brushing teeth, A004 brushing hair
    (AUROC 1.0). 25 classes reach 1.0; the label-id tie rule picks these
    three.
  - In all three, head has the highest lift, and head, arms, and hands
    are rated relevant.
  - For A002 and A003 the top group is not primary: head is rated 1 and
    hands 2.
- **Bottom** (AUROC 0.0; A049 "use a fan" also has 0.0 and is excluded by
  the tie rule):
  - A014 wear jacket: head is highest and rated 0, while the relevant
    torso, arms, and hands have negative lift.
  - A016 wear a shoe: legs are rated 2, but head is highest and every
    relevant group has negative lift.
  - A037 wipe face: head is rated 2 but has the most negative lift of any
    group (-2.6), and torso is highest. Occluding the head box
    changes this class's prediction less than in other classes. Why was
    not examined; the hand covering the face is one possibility.

## Interpretation

- **Supported.**
  - SSAT's class-level body-part profile agrees with relevance ratings made
    from action names alone, beyond chance. The mean class AUROC is 0.69,
    and every group separates relevant from non-relevant classes.
  - The result is stable across preprocessing, sample filtering, area
    adjustment, and single-person classes.
  - This adds an external quantitative reference to the earlier
    qualitative NTU interpretation.
- **Not supported.**
  - The agreement is partial: 13 of 60 classes go the other way, and the
    single top part matches the annotated core part in only 39 % of
    classes.
  - Agreement is weak for fine hand-object actions such as reading and
    writing, and strong for leg-dominant ones.
  - One annotator, with no recorded blind status, cannot establish
    inter-rater reliability.
  - Human relevance is not causal ground truth, so none of this shows that
    SSAT's attributions are correct.

## Action under the original plan (section 15): add a limited quantitative check and limitation text

**NTU case study, draft:**

> To quantify the body-part interpretation, an annotator rated each NTU-60
> class's relevance for five body-part groups (head, torso, arms, hands,
> legs; 0/1/2) from the action name alone, before SSAT scores were
> computed. SSAT's class-level body-part sensitivity (z-scored across
> classes) ranked relevant groups above non-relevant ones with a mean class
> AUROC of 0.69 (95% CI 0.60-0.77; permutation p < 0.001), consistently for
> crop-free preprocessing, correctly classified samples only, and
> area-adjusted scores (0.67-0.70). Agreement was partial: it was high for
> leg-dominant actions and low for fine hand-object actions, and the
> highest-scoring part matched a core part in 39% of classes (chance 30%).

**Limitations, draft:**

> Human relevance ratings are a semantic reference, not causal ground
> truth, and were provided by a single annotator. NTU has no matched
> controls, so body-part scores still include region-size effects that the
> cross-class normalization and area adjustment remove only in part.

**Paper locations:**
- NTU case-study paragraph and figure (add the AUROC result and optionally
  `fig_b2_alignment.pdf`);
- Limitations;
- Response to Reviewer (quantitative semantic validation).

If a second annotator completes `annotator_B.csv`, rerunning
`ssat_part_scores.py`, `evaluate_alignment.py`, `plot_figures.py`, and
`group_breakdown.py` with `--annotators A B` adds the agreement table and
per-annotator robustness rows.

## Deviations

D-011:
- a single annotator (the pre-registered fallback);
- `annotators.csv` empty at analysis time;
- `group_breakdown.py` added post hoc and labelled as such.
