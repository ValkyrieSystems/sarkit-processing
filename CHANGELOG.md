# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- TypeError calling `sicd_area_plane.compute_suitable_rc_plane` with `numpy>=2.5.0`


## [0.1.0] - 2026-08-03

### Added
- `sarkit_processing` and `skp` CLI entrypoints
- `sicd_chip` and `coords` CLI subcommands
- `sicd_area_plane` module
- `sicd_deskew` module
- `sicd_scene_to_image` module
- `atmosphere` module
- `remocomp` module
- `sicd_pixel_type` module

[unreleased]: https://github.com/ValkyrieSystems/sarkit-processing/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ValkyrieSystems/sarkit-processing/releases/tag/v0.1.0
