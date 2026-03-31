"""Re-import relay: entity_materializer moved to flow360_schema.

This module retains ENTITY_TYPE_MAP and _build_entity_instance because they
depend on concrete entity classes that live in the client package.
materialize_entities_and_selectors_in_place is re-exported with a default
entity_builder so that existing callers need no changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pydantic as pd
from flow360_schema.framework.entity.entity_materializer import (
    materialize_entities_and_selectors_in_place as _materialize_entities_and_selectors_in_place,
)
from flow360_schema.framework.entity.entity_utils import DEFAULT_NOT_MERGED_TYPES
from flow360_schema.framework.validation.context import DeserializationContext

from flow360.component.simulation.draft_context.mirror import MirrorPlane
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    ImportedSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    SeedpointVolume,
    SnappyBody,
    Sphere,
    Surface,
    WindTunnelGhostSurface,
)

if TYPE_CHECKING:
    from flow360.component.simulation.framework.entity_registry import EntityRegistry

ENTITY_TYPE_MAP = {
    "Surface": Surface,
    "Edge": Edge,
    "GenericVolume": GenericVolume,
    "GeometryBodyGroup": GeometryBodyGroup,
    "CustomVolume": CustomVolume,
    "AxisymmetricBody": AxisymmetricBody,
    "Box": Box,
    "Cylinder": Cylinder,
    "Sphere": Sphere,
    "ImportedSurface": ImportedSurface,
    "GhostSurface": GhostSurface,
    "GhostSphere": GhostSphere,
    "GhostCircularPlane": GhostCircularPlane,
    "Point": Point,
    "PointArray": PointArray,
    "PointArray2D": PointArray2D,
    "Slice": Slice,
    "SeedpointVolume": SeedpointVolume,
    "SnappyBody": SnappyBody,
    "WindTunnelGhostSurface": WindTunnelGhostSurface,
    "MirroredSurface": MirroredSurface,
    "MirroredGeometryBodyGroup": MirroredGeometryBodyGroup,
    "MirrorPlane": MirrorPlane,
}


def _build_entity_instance(entity_dict: dict):
    """Construct a concrete entity instance from a dictionary via TypeAdapter."""
    type_name = entity_dict.get("private_attribute_entity_type_name")
    cls = ENTITY_TYPE_MAP.get(type_name)
    if cls is None:
        raise ValueError(f"[Internal] Unknown entity type: {type_name}")
    with DeserializationContext():
        return pd.TypeAdapter(cls).validate_python(entity_dict)


def materialize_entities_and_selectors_in_place(
    params_as_dict: dict,
    *,
    not_merged_types: set[str] = DEFAULT_NOT_MERGED_TYPES,
    entity_registry: Optional[EntityRegistry] = None,
) -> dict:
    """Wrapper that injects the default entity builder for the client package."""
    return _materialize_entities_and_selectors_in_place(
        params_as_dict,
        entity_builder=_build_entity_instance,
        not_merged_types=not_merged_types,
        entity_registry=entity_registry,
    )
