"""Debugging visualizations over computed metrics (V1-V3 implemented in stages 7-8).

Each view lives in its own submodule and is imported directly (e.g.
``from ssat.metrics.viz.mask_check import save_mask_check_views``), not
re-exported here or from ``ssat.metrics`` — a broken view module must not
prevent the rest of the metrics engine, or the other debugging views, from
importing (design METRIC_ENGINE_DESIGN_v1.md §3.1 "하나가 깨져도 나머지 디버깅
도구는 계속 쓸 수 있어야 한다"). ``mask_check.py`` (V1) is fully independent
of ``heatmap.py``/``ranking.py`` (V2/V3); ``ranking.py`` (V3), by contrast,
deliberately imports ``heatmap.py`` (V2) since "상위/하위 N개를 히트맵과 함께
나열" makes that coupling part of V3's own definition, not an accident (see
``ranking.py``'s module docstring). Shared low-level plumbing (canonical-JSON
decoding, dump-relative image loading) lives in the private ``_shared.py``,
mirroring ``ssat.metrics._storage``'s precedent, so no view module needs to
import another purely for that.
"""
