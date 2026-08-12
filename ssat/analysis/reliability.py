"""A6 ReliabilityScorer.

Combines A2-A5 outputs into per-(AnchorKey, metric) flags
(``ssat.analysis.types.FlagValue``, always distinguishing "unavailable"
from "false"), an overall ``ReliabilityGrade``, and human-readable reasons.

Not yet implemented — this is a scaffolding placeholder. Implemented in
IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계7; design CONTROL_STABILITY_DESIGN_v1.md §A6.
"""

from __future__ import annotations
