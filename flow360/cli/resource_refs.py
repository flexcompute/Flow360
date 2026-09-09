"""
Shared CLI parsing for typed Flow360 resource references.
"""

from __future__ import annotations

from dataclasses import dataclass

RESOURCE_PREFIX_MAP = {
    "prj": "Project",
    "geo": "Geometry",
    "sm": "SurfaceMesh",
    "vm": "VolumeMesh",
    "case": "Case",
    "dft": "Draft",
    "folder": "Folder",
}
ROOT_FOLDER_PREFIX = "ROOT.FLOW360"


class ResourceRefError(ValueError):
    """Raised when a CLI resource reference is malformed or unsupported."""


@dataclass(frozen=True)
class ResourceRef:
    """Normalized typed resource reference parsed from a Flow360 id."""

    id: str
    resource_type: str


def parse_resource_ref(resource_id: str) -> ResourceRef:
    """Parse a Flow360 resource id by its stable type prefix."""
    normalized_id = resource_id.strip()
    if not normalized_id:
        raise ResourceRefError("Resource ID cannot be empty.")

    if normalized_id == ROOT_FOLDER_PREFIX or normalized_id.startswith(f"{ROOT_FOLDER_PREFIX}."):
        return ResourceRef(id=normalized_id, resource_type="Folder")

    prefix, separator, suffix = normalized_id.partition("-")
    if not separator or not suffix:
        raise ResourceRefError(
            f"Resource ID '{resource_id}' does not have the expected '<prefix>-...' shape."
        )

    resource_type = RESOURCE_PREFIX_MAP.get(prefix)
    if resource_type is None:
        expected_prefixes = ", ".join(f"{value}-" for value in sorted(RESOURCE_PREFIX_MAP))
        raise ResourceRefError(
            f"Unsupported resource ID prefix in '{normalized_id}'. "
            f"Expected one of: {expected_prefixes}."
        )

    return ResourceRef(id=normalized_id, resource_type=resource_type)


def require_resource_type(resource_id: str, expected_type: str) -> ResourceRef:
    """Parse and validate that a resource id matches the expected Flow360 type."""
    resource_ref = parse_resource_ref(resource_id)
    if resource_ref.resource_type != expected_type:
        raise ResourceRefError(
            f"Expected a {expected_type} ID, got {resource_ref.id} ({resource_ref.resource_type})."
        )
    return resource_ref
