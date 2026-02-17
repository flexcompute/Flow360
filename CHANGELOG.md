All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### New Features
- added base model with imports/exports from/to JSON and YAML
- added Flow360Params which inherits from the base model
- exposed most of the classes and functions in `__init__.py`, eg `flow360.MyCases()`
- added params for surface and volume meshing
- listing cases/meshes follows pagination from webUI for improved performance
- added example files (links to meshes, case.json etc) inside flow360 package so the example scripts can be standalone
- added more examples, including surface meshing and volume meshing
- added support for units, eg: `flow360.Freestream.from_speed(speed=(10, "m/s"))`, `flow360.SlidingInterface(omega=(1, "rad/s"), ...)`
- added custom types validation, eg. coordinates, axes.
- added custom exceptions and logger (with support to log to file)
- added mock webAPI for local unit testing

### Updates
- case lists and mesh lists return Case objects (instead of CaseMeta)
- all server-side data is a lazy load
- split code to `Case` (cloud resource) and `CaseDraft` (before submission)
- split code to `VolumeMesh` (cloud resource) and `VolumeMeshDraft` (before submission)
- added constructor from filename: `Flow360Params("path/to/file.json")`
- status is enum with `is_final()` method
- dropped support for Python 3.9; minimum supported Python version is now 3.10

### Bug Fixes
- added unittests, code coverage 67%


## [v0.1.8] - 2023-3-21

### New Features
- ...

### Updates
- ...

### Bug Fixes
- supporting muRef in freestream by
- fixed maxPhysicalSteps

## [v0.1.6] - 2023-1-24

### New Features
- Initial release

### Updates
- ...

### Bug Fixes
- ...

## [v0.1.1] - 2023-8-24

### Added
- Initial release




[Unreleased]: https://github.com/flexcompute/Flow360/compare/v0.1.8...develop
[v0.1.8]: https://github.com/flexcompute/Flow360/compare/v0.1.6...v0.1.8
[v0.1.6]: https://github.com/flexcompute/Flow360/compare/v0.1.1...v0.1.6
[v0.1.1]: https://github.com/flexcompute/Flow360/releases/tag/v0.1.1
