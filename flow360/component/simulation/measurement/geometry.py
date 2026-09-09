"""Geometry-root data access shared by measurement utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.entity.entity_registry import EntityRegistryView
from flow360_schema.framework.entity.entity_selector import EntitySelector
from flow360_schema.models.entities.surface_entities import Surface
from unyt import unyt_quantity

from flow360.cloud.file_cache import get_shared_cloud_file_cache
from flow360.component.simulation.framework.entity_expansion_utils import (
    normalize_selection,
)
from flow360.exceptions import Flow360RuntimeError, Flow360ValueError

from .tessellation_loader import TessellationFileLoader

if TYPE_CHECKING:
    from flow360.component.geometry import Geometry
    from flow360.component.simulation.draft_context.context import DraftContext


# Every form a caller may use to name surfaces. `normalize_selection` collapses all of
# them to `list[Surface]` so downstream code has a single shape to handle.
SurfaceSelection = (
    Surface | EntitySelector | EntityList | EntityRegistryView | list[Surface | EntitySelector]
)


def require_geometry_root(draft: DraftContext, *, operation: str) -> Geometry:
    """Return the draft's Geometry root or raise a direct measurement error."""
    geometry_root = draft._geometry_root  # pylint: disable=protected-access
    if geometry_root is None:
        raise Flow360RuntimeError(
            f"{operation} requires a draft created from a Geometry resource. "
            "Drafts from SurfaceMesh or VolumeMesh do not have tessellation data."
        )
    return geometry_root


def require_project_length_unit(geometry_root: Geometry, *, operation: str) -> unyt_quantity:
    """Return the geometry's project length unit or raise a direct measurement error.

    A measurement result without a length unit is not usable downstream, so a missing
    unit is a hard error rather than a silently dimensionless return value. Call this
    before any tessellation download so the failure is immediate.
    """
    length_unit = geometry_root._project_length_unit  # pylint: disable=protected-access
    if length_unit is None:
        raise Flow360RuntimeError(
            f"{operation} requires the Geometry resource to carry a project length unit, but "
            "`private_attribute_asset_cache.project_length_unit` is missing from its simulation "
            "settings. Re-create the draft from a Geometry resource that was uploaded with a "
            "length unit."
        )
    return length_unit


def load_transformed_surface_vertices(
    draft: DraftContext,
    geometry_root: Geometry,
    surface_face_ids: list[tuple[Surface, list[str]]],
) -> np.ndarray:
    """Load and transform tessellated vertices while preserving surface ownership."""
    loader = _create_tessellation_loader(draft, geometry_root)
    transforms = _resolve_surface_transforms(draft)

    transformed_vertices = []
    for surface, face_ids in surface_face_ids:
        vertices = loader.load_vertices(face_ids)
        transformed_vertices.append(_apply_transform(transforms.get(surface.name), vertices))
    return np.concatenate(transformed_vertices, axis=0)


def load_transformed_surface_triangles(
    draft: DraftContext,
    geometry_root: Geometry,
    surface_face_ids: list[tuple[Surface, list[str]]],
) -> np.ndarray:
    """Load and transform tessellated triangles while preserving surface ownership."""
    loader = _create_tessellation_loader(draft, geometry_root)
    transforms = _resolve_surface_transforms(draft)

    transformed_triangles = []
    for surface, face_ids in surface_face_ids:
        triangles_by_face = loader.load_triangles(face_ids)
        triangles = np.concatenate(list(triangles_by_face.values()), axis=0)
        transformed_triangles.append(_apply_transform(transforms.get(surface.name), triangles))
    return np.concatenate(transformed_triangles, axis=0)


def _create_tessellation_loader(
    draft: DraftContext, geometry_root: Geometry
) -> TessellationFileLoader:
    """Create an on-demand tessellation loader for the draft's active geometries."""
    # pylint: disable=protected-access
    geometry_resources = {
        geometry.id: geometry._webapi for geometry in [geometry_root, *draft.imported_geometries]
    }
    return TessellationFileLoader(
        geometry_resources=geometry_resources,
        cloud_cache=get_shared_cloud_file_cache(),
    )


def _resolve_surface_transforms(draft: DraftContext) -> dict[str, np.ndarray]:
    """Map surface name to the transform of the body group owning that surface's faces.

    A coordinate system attaches to a body group, never to an individual surface, so a
    surface inherits whatever transform its owning body group carries. Many surfaces
    share one body group, so each matrix is composed once per body group and shared by
    every surface that body group owns -- composing one walks the whole parent chain and
    scans the registered coordinate systems, which is far too costly to repeat per
    surface.

    Returns an empty mapping when the draft assigns no coordinate system at all. That
    shortcut matters: the body-to-face mapping is unavailable on geometries uploaded
    before 25.5, and measuring those must keep working as long as they use no transform.
    """
    coordinate_systems = draft.coordinate_systems
    # pylint: disable=protected-access
    if not coordinate_systems._entity_key_to_coordinate_system_id:
        return {}

    face_group_to_body_group = draft._entity_info.get_face_group_to_body_group_id_map()

    matrix_by_body_group = {}
    for body_group_id in set(face_group_to_body_group.values()):
        matrix = coordinate_systems._get_matrix_for_entity_key(
            entity_type="GeometryBodyGroup", entity_id=body_group_id
        )
        if matrix is not None:
            matrix_by_body_group[body_group_id] = np.asarray(matrix, dtype=np.float64)

    return {
        surface_name: matrix_by_body_group[body_group_id]
        for surface_name, body_group_id in face_group_to_body_group.items()
        if body_group_id in matrix_by_body_group
    }


def _apply_transform(matrix: Optional[np.ndarray], coordinates: np.ndarray) -> np.ndarray:
    """Apply a composed coordinate-system transform to tessellation coordinates."""
    if matrix is None:
        return coordinates
    return coordinates @ matrix[:, :3].T + matrix[:, 3]


def resolve_surface_entities(
    draft: DraftContext,
    surfaces: SurfaceSelection,
    *,
    operation: str,
) -> list[Surface]:
    """Resolve a measurement surface input into deduplicated source Surface entities."""
    resolved = normalize_selection(
        draft._entity_registry,  # pylint: disable=protected-access
        surfaces,
        operation=operation,
        expected_entity_type=Surface,
    )

    unsupported = [
        getattr(surface, "name", type(surface).__name__)
        for surface in resolved
        if not isinstance(surface, Surface)
    ]
    if unsupported:
        raise Flow360ValueError(
            f"{operation} supports only source Surface entities. "
            "MirroredSurface and other draft-only entities are not supported: "
            f"{', '.join(unsupported)}."
        )
    return resolved


def collect_surface_face_ids(
    surfaces: list[Surface], *, operation: str
) -> list[tuple[Surface, list[str]]]:
    """Pair each selected source surface with its tessellation face identifiers."""
    missing_face_ids = [
        surface.name for surface in surfaces if not surface.private_attribute_sub_components
    ]
    if missing_face_ids:
        raise Flow360ValueError(
            f"{operation} requires each selected Surface to have tessellation face identifiers. "
            f"Missing on: {', '.join(missing_face_ids)}."
        )

    if not surfaces:
        raise Flow360ValueError(f"{operation} requires at least one Surface.")
    return [(surface, list(surface.private_attribute_sub_components)) for surface in surfaces]
