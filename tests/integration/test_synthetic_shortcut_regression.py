"""Regression test locking in the synthetic-shortcut experiment's Q1-Q5 verdicts.

This imports the judging logic straight out of
``experiments/synthetic_shortcut/evaluate.py`` instead of reimplementing it,
so this test breaks the moment that script's rank/multiplier/margin logic
changes -- which is the intended safety net, since that script is what
originally produced the pre-registered Q1-Q5 results. It replays that
judgment against a small, checked-in copy of the crop-free run's metrics
stores and accuracy numbers (``tests/fixtures/synthetic_shortcut_regression/``)
rather than the full (gitignored, multi-hundred-MB) experiment output tree.

``experiments/synthetic_shortcut/`` is not an importable package (no
``__init__.py``), and ``evaluate.py`` itself uses bare imports
(``from common import ...``, ``from run_audit import ...``) that only resolve
when its own directory is first on ``sys.path`` -- mirroring how
``tests/conftest.py`` already puts ``tests/fixtures/`` on ``sys.path`` for
similar bare imports. Both ``common`` and ``run_audit`` are fairly generic
module names; if ``tests/fixtures/`` ever grows same-named modules, whichever
directory sys.path favors will shadow the other silently, so keep an eye out
if this test starts importing unexpected code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "synthetic_shortcut"
_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_shortcut_regression"

if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from evaluate import _judge, _load_region_means  # noqa: E402
from run_audit import RUN_SPECS  # noqa: E402


def _region_means_by_run() -> dict[str, dict[str, float]]:
    return {
        spec.run_id: _load_region_means(_FIXTURE_DIR / "metrics" / spec.run_id)
        for spec in RUN_SPECS
    }


def test_synthetic_shortcut_q1_q5_regression() -> None:
    """Q1-Q5 must keep passing against the fixed crop-free fixture snapshot.

    Reference values at the time this fixture was captured: Q1 rank=1,
    Q2 multiplier~=175.65 (>=3.0), Q3 rank in M_normal=16 (!=1), Q4
    reproduced in 5/5 fill strategies (>=2), Q5 margin~=95.75 points
    (>=10.0) -- see thresholds.json for the authoritative pass/fail
    thresholds this test asserts against.
    """

    thresholds = json.loads((_EXPERIMENT_DIR / "thresholds.json").read_text(encoding="utf-8"))
    accuracy = json.loads((_FIXTURE_DIR / "accuracy.json").read_text(encoding="utf-8"))
    verdicts = _judge(thresholds, _region_means_by_run(), accuracy)

    q1 = verdicts["Q1_identifies_patch_region"]
    assert q1["pass"]
    assert q1["patch_region_rank"] == 1

    q2 = verdicts["Q2_separated_from_baseline"]
    assert q2["pass"]
    assert q2["multiplier"] >= thresholds["q2_multiplier"]

    q3 = verdicts["Q3_distinguishes_normal_model"]
    assert q3["pass"]
    assert q3["patch_region_rank_in_m_normal"] != 1

    q4 = verdicts["Q4_robust_to_fill_strategy"]
    assert q4["pass"]
    assert len(q4["reproduced_in"]) >= thresholds["q4_min_fill_strategies"]

    q5 = verdicts["Q5_predicts_generalization_gap"]
    assert q5["pass"]
    assert q5["margin_points"] >= thresholds["q5_min_margin_points"]
