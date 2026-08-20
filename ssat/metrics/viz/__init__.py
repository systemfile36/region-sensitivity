"""Debugging visualizations over computed metrics.

Each view lives in its own submodule and is imported directly (e.g.
``from ssat.metrics.viz.mask_check import save_mask_check_views``), not
re-exported here or from ``ssat.metrics`` — a broken view module must not
prevent the rest of the metrics engine, or the other debugging views, from
importing: even if one view breaks, the rest must stay usable.
``mask_check.py`` (V1) is fully independent of ``heatmap.py``/``ranking.py``
(V2/V3); ``ranking.py`` (V3), by contrast, deliberately imports
``heatmap.py`` (V2) since listing the top/bottom N samples alongside their
heatmaps makes that coupling part of V3's own definition, not an accident
(see ``ranking.py``'s module docstring). Shared low-level plumbing
(canonical-JSON decoding, dump-relative image loading) lives in the private
``_shared.py``, mirroring ``ssat.metrics._storage``'s precedent, so no view
module needs to import another purely for that.
"""
