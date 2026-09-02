# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-02

_The `v1.0.0` tag is being amended in place ahead of the SoftwareX submission: it has
not been released/cited externally yet, so the fixes below are folded into this
entry rather than published as a separate `1.0.1`._

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

[Unreleased]: https://github.com/systemfile36/region-sensitivity/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/systemfile36/region-sensitivity/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/systemfile36/region-sensitivity/releases/tag/v0.1.0
