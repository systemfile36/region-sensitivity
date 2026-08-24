# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-24

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

### Fixed

- Fixed an error where ImageNet annotation files were incorrectly parsed as JSON.

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
