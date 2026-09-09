"""flow360-schema: Pure Pydantic schemas for Flow360.

Single Source of Truth for Flow360 simulation parameters.

- ``framework/``: Base model, validation, and mixins
- ``models/``: Schema model definitions (reference geometry, simulation params, etc.)
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from flow360_schema.exceptions import Flow360DeprecationError, Flow360Error, Flow360ValueError
from flow360_schema.framework.unit_system import (
    CGS_unit_system,
    CGSUnitSystem,
    ImperialUnitSystem,
    SI_unit_system,
    SIUnitSystem,
    UnitSystem,
    UnitSystemType,
    create_flow360_unit_system,
    imperial_unit_system,
)
from flow360_schema.framework.validation.context import StrictUnitContext, unit_system_manager

SCHEMA_DIST_NAME = "flow360-schema"


def _read_pyproject_version_at(pyproject_path: Path) -> str | None:
    """Return the version from a pyproject.toml only if it identifies schema.

    Gates the read on `name = "flow360-schema"` so a sibling client pyproject
    encountered during the parent-walk fallback cannot satisfy the lookup
    with the wrong version. See _resolve_version().
    """
    if not pyproject_path.is_file():
        return None
    content = pyproject_path.read_text(encoding="utf-8")
    name_match = re.search(r'^name\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    if name_match is None or name_match.group(1) != SCHEMA_DIST_NAME:
        return None
    version_match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    if version_match is None:
        return None
    return version_match.group(1)


def _read_local_pyproject_version(here: Path | None = None) -> str | None:
    """Locate schema's pyproject.toml across the layouts that ship it.

    - Source checkout: ``flex/share/flow360-schema/pyproject.toml`` is two
      parents above ``src/flow360_schema/__init__.py``.
    - Inlined install (PyPI / public mirror): ``utilities/inline_flow360.py``
      writes a minimal stub at ``flow360_schema/pyproject.toml`` next to the
      package.

    Both candidates are gated on ``name = "flow360-schema"`` so a stray
    client pyproject can never satisfy the lookup.
    """
    here = here if here is not None else Path(__file__).resolve().parent
    for candidate in (here / "pyproject.toml", here.parents[1] / "pyproject.toml"):
        found = _read_pyproject_version_at(candidate)
        if found is not None:
            return found
    return None


def _resolve_version() -> str:
    try:
        return version(SCHEMA_DIST_NAME)
    except PackageNotFoundError:
        local_version = _read_local_pyproject_version()
        if local_version is not None:
            return local_version
        raise RuntimeError("Unable to determine package version from metadata or pyproject.toml") from None


__version__ = _resolve_version()

__all__ = [
    "__version__",
    "Flow360DeprecationError",
    "Flow360Error",
    "Flow360ValueError",
    "StrictUnitContext",
    "unit_system_manager",
    "CGSUnitSystem",
    "CGS_unit_system",
    "ImperialUnitSystem",
    "SIUnitSystem",
    "SI_unit_system",
    "UnitSystem",
    "UnitSystemType",
    "create_flow360_unit_system",
    "imperial_unit_system",
]
