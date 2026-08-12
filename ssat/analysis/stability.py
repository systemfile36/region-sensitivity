"""A3 StabilityAnalyzer.

Measures seed, jitter, and fill-strategy stability for a fixed ``AnchorKey``
across varying ``ConditionKey``s, plus dataset-level operator rank
correlation. The three axes are kept in separate functions but share this
one module because they are variations of the same question (design §3.1)
rather than distinct computations. Jitter stability is permanently
unavailable in v1 — the core has no jitter-mask-variation support
(IMPLE_PLAN_CONTROL_STABILITY_v1.md §1 항목2) — so its function will always
return ``FlagValue.UNAVAILABLE`` once implemented.

Not yet implemented — this is a scaffolding placeholder. Implemented in
IMPLE_PLAN_CONTROL_STABILITY_v1.md §5 단계4; design CONTROL_STABILITY_DESIGN_v1.md §A3.
"""

from __future__ import annotations
