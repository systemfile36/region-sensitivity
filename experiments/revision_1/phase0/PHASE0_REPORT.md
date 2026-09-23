# Phase 0 report: revision baseline freeze

Baseline code: tag `v1.0.0` (`9d192c8`). Phase 0 tooling: branch
`revision-1`. Every JSON in `summary/` carries a `*.provenance.json` with the
commit, inputs (manifest SHA-256s), and environment it was produced from.

## Outcome by task

| ID | Result |
|---|---|
| P0-1 baseline manifest | 8 baseline runs + 4 K=1 parity runs described (`baseline_manifest.json`). Every baseline store chains correctly (metrics computed from its dump, analysis from its metrics), all items `ok`, source annotation files unchanged. |
| P0-2 paper numbers | 32/32 checks pass against `docs/internal/paper/manuscript.tex` (`paper_numbers.json`): ImageNet grade shares, clean accuracy, mean region `margin_drop`, top-region share, NTU body-part ranking and class examples, synthetic Q1-Q5. |
| P0-3 K=1 vs K=3 parity | Identity passes in all four pairs (items, masks, seeds, clean logits). Logits are not bitwise equal everywhere; see finding 1 and `deviations.md` D-001. |
| P0-4 benchmark note | `docs/BENCHMARK_v1.md` now states the real-dataset benchmark used one control per target (config SHA-256 `0abaa1e6…`, commit `bb6b0af`). |
| P0-5 environment | `environment.json`: RTX 4090 (driver 580.173.02), CUDA 12.9, cuDNN 9.10.2, torch 2.8.0+cu129, timm 1.0.28, numpy 2.3.2, pandas 3.0.5, pyarrow 25.0.0, Python 3.11.13, image `sha256:d78849c9…`. |
| P0-6 common modules | `common/` plus `tests/unit/test_rev1_common.py` (15 tests) pass in the container. |
| P0-7 (added) recompute | NTU stores reproduce exactly; synthetic stores reproduce except bootstrap CI bounds (8/3,200 shortcut grades). The NTU and synthetic baselines now read the recomputed stores (`deviations.md` D-002). |

## Findings that affect later experiments

1. **Model outputs depend on batch composition (A2, A4).** cuDNN runs
   convolutions in TF32 by default, so the same item can get slightly
   different logits depending on the batch it lands in. For mnv2_050,
   a third of the K=1 controls differ from their identical K=3
   counterparts by 1e-3 to 2.4e-2 (0.02 % change top-1); mnv2_100 stays
   within 4e-6. A2's "bit-identical nested pool" therefore holds for items,
   masks, and seeds but not for logits; A2 compares K subsets within the
   K=20 run.
2. **The CI flag is Monte-Carlo sensitive (A1).** Reordering the bootstrap
   input alone moved CI bounds by up to 0.018 and flipped
   `ci_excludes_zero` for one synthetic region. A1 should treat
   `ci_excludes_zero` near zero as noisy and report the bootstrap seed.
3. **Paper claim scope (B1/A3 write-up).** The "roughly 14-17 %"
   top-region share holds for the exact runs (14.7-17.3 %); the crop-free
   runs give 12.3-15.1 % with the same four central cells leading.
4. **K=1 runs have no `exceeds_control` (context).** With one control the
   z-score is undefined, so the K=1 runs grade 0 % HIGH and ~41 % MODERATE;
   an earlier manuscript draft quoted that split (40.97/59.03). The
   submitted numbers come from the K=3 runs.

## Environment notes

- The container was recreated on 2026-09-23 from an image built on
  2026-08-04. `decord` (a declared dependency) was missing and was
  pip-installed into the running container; it has to be reinstalled if the
  container is recreated. `captum` (optional `reference` extra) is not
  installed and is needed only for C1.
- `ssat` is imported from the bind-mounted `/workspace` tree, but the
  installed distribution metadata is a stale `0.1.0`, so every dump
  manifest records `code_version: 0.1.0`. Provenance should cite the git
  SHA in `*.provenance.json`, not `code_version`.
- One pre-existing test failure at `v1.0.0`, unrelated to the revision:
  `tests/unit/test_stage10_artifacts.py` still expects `version == "0.1.0"`
  while `pyproject.toml` says `1.0.0`. The Captum tests cannot be collected
  without the `reference` extra.
- A non-container process held ~400 MiB of GPU memory at capture time;
  A4/C1 measurements must record `nvidia-smi` process lists as planned.
