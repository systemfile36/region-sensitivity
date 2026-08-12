"""A2 ControlComparator.

Compares each target ``AnchorKey`` against its matched controls per
``ConditionKey`` (``excess``, ``ratio``, ``z_vs_control``), distinguishing
"no control requested" from "control requested but unmatched" via
``FlagValue.UNAVAILABLE``.

Not yet implemented — this is a scaffolding placeholder. Implemented in
IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계3; design CONTROL_STABILITY_DESIGN_v1.md §A2.
"""

from __future__ import annotations
