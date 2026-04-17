"""Re-import relay: entity info models now live in flow360_schema.models.entity_info."""

# pylint: disable=unused-import
# ruff: noqa: F401
from flow360_schema.models.entity_info import (
    BodyComponentInfo,
    DraftEntityTypes,
    EntityInfoModel,
    EntityInfoUnion,
    GeometryEntityInfo,
    SurfaceMeshEntityInfo,
    VolumeMeshEntityInfo,
    merge_geometry_entity_info,
    parse_entity_info_model,
)
