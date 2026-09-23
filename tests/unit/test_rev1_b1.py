"""Unit tests for the B1 ranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.revision_1.b1_preprocessing.area_tables import SQUARE_EXACT_RATIO
from experiments.revision_1.b1_preprocessing.compare_protocols import (
    descending_ranks,
    ranking_summary,
    representative_cases,
    row_spearman,
    top_share,
)
from experiments.revision_1.common.agreement import spearman

CELLS = ["a", "b", "c", "d"]


def _matrix(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CELLS, index=[f"s{index}" for index in range(len(rows))])


def test_descending_ranks_break_ties_by_column_order() -> None:
    ranks = descending_ranks(_matrix([[0.1, 0.5, 0.5, -0.2], [3.0, 2.0, 1.0, 0.0]]))
    assert ranks.tolist() == [[3, 1, 2, 4], [1, 2, 3, 4]]


def test_row_spearman_matches_pairwise_spearman_and_flags_constant_rows() -> None:
    a = _matrix([[0.1, 0.5, 0.2, -0.2], [1.0, 1.0, 1.0, 1.0], [0.3, 0.3, 0.1, 0.2]])
    b = _matrix([[0.2, 0.4, 0.1, 0.0], [0.1, 0.2, 0.3, 0.4], [0.1, 0.3, 0.3, 0.2]])
    rho = row_spearman(a, b)
    assert rho["s0"] == pytest.approx(spearman(a.loc["s0"], b.loc["s0"]))
    assert rho["s2"] == pytest.approx(spearman(a.loc["s2"], b.loc["s2"]))
    assert np.isnan(rho["s1"])


def test_ranking_summary_top1_and_top3_overlap() -> None:
    a = _matrix([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    b = _matrix([[4.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]])
    summary, _, ranks_a, _ = ranking_summary(a, b)
    assert summary["top1_agreement"] == 0.5
    # s0 top-3 {a,b,c} vs {a,d,c} -> 2/3; s1 {d,c,b} vs {a,b,c} -> 2/3.
    assert summary["top3_overlap_mean"] == pytest.approx(2 / 3)
    assert top_share(ranks_a, CELLS).to_dict() == {"a": 0.5, "b": 0.0, "c": 0.0, "d": 0.5}


def test_representative_cases_pick_nearest_quantile_with_sample_id_ties() -> None:
    rho = pd.Series({"s3": 0.5, "s1": 0.5, "s2": 0.1, "s0": 0.9, "s4": np.nan})
    cases = pd.DataFrame(representative_cases(rho, ("x", "y"))).set_index("quantile")
    assert cases.loc[50, "sample_id"] == "s1"
    assert cases.loc[10, "sample_id"] == "s2"
    assert cases.loc[90, "sample_id"] == "s0"


def test_square_image_exact_ratio_is_the_published_bound() -> None:
    assert SQUARE_EXACT_RATIO == pytest.approx(4096 / 2304)
    assert round(SQUARE_EXACT_RATIO, 2) == 1.78
