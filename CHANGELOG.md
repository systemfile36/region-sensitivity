# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-10-06

Release for the first revision of the SoftwareX paper. The `ssat` package's
defaults, behavior, and outputs are unchanged; this release adds the
revision's validation experiments and documentation, and one performance fix.

### Added

- `experiments/revision_1/`: the revision's pre-registered experiments, each
  with a protocol committed before it ran, tracked summary tables, and a
  report, plus a deviations log. They cover
  reliability-threshold sensitivity (A1), the number of matched controls
  (A2), two more ImageNet architectures, ConvNeXt-T and DeiT-S (A3), audit
  cost scaling (A4), the ImageNet preprocessing confound (B1), NTU RGB+D
  body-part profiles against human relevance ratings (B2), and the Captum
  comparison's runtime/memory/storage measurement and second, ImageNet,
  setting (C1).
- A Captum reference workflow for ImageNet
  (`experiments/revision_1/c1_captum/imagenet_workflow/`), adapted from the
  synthetic-shortcut reference.
- `docs/VALIDATION_REVISION_1.md`, a public summary of the revision
  experiments.
- `scripts/paper_figures/rev1_fig5_design_sensitivity.py`, which draws the
  revised paper's design-sensitivity and cost figure from tracked summary
  tables only.
- Unit tests for the revision experiment code (`tests/unit/test_rev1_*.py`).
- `psutil` in the `reference` optional extra, for the Captum comparison's
  measurement scripts.

### Changed

- `PlanBuilder.materialize` recomputes a sample's work items once per sample instead
  of once per chunk. Runs with many items per sample (controls, several operators
  and seeds) spent most of their per-item time re-enumerating the plan; item IDs,
  chunk IDs, and dumps are unchanged. The synthetic Captum-comparison workflow
  went from 1,207 s to 389 s.
- README: links the revision validation and states two limitations it measured:
  few matched controls inflate the HIGH grade share, and the per-item CPU
  pipeline is slower than purpose-built GPU-batched code.
- `docs/CONFIG_REFERENCE.md`: guidance on choosing `controls[].n_samples` and on
  control area matching under crop-based preprocessing.
- `docs/REFERENCE_COMPARISON_CAPTUM_v1.md`: runtime, memory, and storage of the
  synthetic comparison, the ImageNet comparison, and the speed limitation.
- `docs/BENCHMARK_v1.md`: notes that its real-dataset numbers were measured with
  one control per target and before the planner fix, and points to the revision's
  scaling study.
- `docs/REAL_DATASET_CASE_STUDY_v1.md`: a section on the revision's extensions of
  the case study.

### Fixed

- Two tests asserted the version literal `0.1.0` and failed against the `1.0.0`
  metadata; they now check that `pyproject.toml`, `CITATION.cff`, and the installed
  package report the same version.
- The CI test job now installs the `reference` extra's packages (`captum`,
  `psutil`), so the Captum reference-workflow tests are collected and run instead
  of failing to import.
- `docs/INSTALLATION.md` says to re-run the editable install after a version
  change: a stale editable install reports an old version in `ssat --version` and
  in every run manifest's `code_version`.
- Public documents no longer point to internal documents that are not distributed
  (`docs/internal/`): the README, the reproducibility demo, the benchmark, and the
  dependency comments; the experiment READMEs now say which cited documents are
  internal.

## [1.0.0] - 2026-09-02

_The `v1.0.0` tag was amended in place before the SoftwareX submission, so the
fixes below were folded into this entry. It has since been cited, and later
changes are released as new versions._

### Added

- Video-classification support: deterministic `uniform` and `segment_center` frame
  sampling, a Kinetics-style CSV source provider, a native TSM-ResNet50 adapter, and
  crop-free timm preprocessing.
- End-to-end real-dataset case study workflow (ImageNet, NTU RGB+D) with
  reproducibility demo scripts and pretrained checkpoints reproducing the Q1-Q5
  synthetic-shortcut result.
- Captum reference workflow for cross-checking SSAT's core sensitivity computation
  against a Captum baseline.
- An effective-area sanity check in the core sensitivity computation.
- Skeleton-based heat-map visualization and video playback support in the HTML
  report layer.
- An automated Q1-Q5 synthetic-shortcut regression test and an advisory CI job for
  the reproducibility demo.

### Changed

- Vectorized the analysis module for improved performance.
- Refactored the project's module boundaries and internal structure.
- The package version is now derived from installed package metadata
  (`importlib.metadata`) instead of a hard-coded string; dependency version
  constraints were updated accordingly.
- Heatmap gallery cards now render only the heatmap overlay instead of a
  redundant original|overlay two-panel figure, since the original is already
  shown via the card's own thumbnail.
- Removed the title from the fill-strategy rank-correlation chart.
- Increased the real-dataset case study's per-target control sample count from
  1 to 3 for the ImageNet crop-free/exact configurations, and updated the case
  study documentation accordingly.

### Fixed

- Fixed an error where ImageNet annotation files were incorrectly parsed as JSON.
- Fixed a flagged-anchors list that could grow large enough on real-dataset-scale
  runs to bloat `report.html` past 400MB and become effectively unopenable; the
  inline list is now capped at 20 rows, with the full list still available via
  `data/flagged_items.csv`.
- Fixed overly long grid-based region ids (e.g. `grid_4x4::grid_4x4/r0/c0`)
  crowding out chart labels and report text by displaying a shortened region id
  (dropping the redundant `<region_id>::` prefix) everywhere a region key
  appears in the report, while keeping the full key in a `title` attribute and
  in CSV/JSON exports.
- Renamed `LICENSE` to `LICENSE.txt` and updated the README links that pointed
  to it, matching the SoftwareX Guide for Authors' expected repository file
  name.
- Fixed a grammar typo in the README's preflight-check section ("an reviewed
  run" -> "a reviewed run").
- Updated `CITATION.cff`'s `version` and `date-released`, which had been left
  at `0.1.0`/2026-08-20 since the initial packaging commit, to match this
  `1.0.0` release.

### Removed

- Retired stale internal planning/design documents and the Korean-language code
  comments that referenced them.

## [0.1.0] - 2026-08-20

### Added

- Minimum packaging/release/regression-testing foundation for the SoftwareX
  submission: LICENSE, CITATION.cff, CONTRIBUTING.md, PEP 621 dependencies, a
  clean-install CI job, and an automated Q1-Q5 synthetic-shortcut regression test.

### Changed

- Moved internal planning/design documents into `docs/internal/`; code comments no
  longer reference them.

[Unreleased]: https://github.com/systemfile36/region-sensitivity/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/systemfile36/region-sensitivity/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/systemfile36/region-sensitivity/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/systemfile36/region-sensitivity/releases/tag/v0.1.0
