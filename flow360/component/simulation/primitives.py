"""Relay primitive entity types and ReferenceGeometry from schema."""

# pylint: disable=unused-import
from flow360_schema.models.entities.base import (
    BOUNDARY_FULL_NAME_WHEN_NOT_FOUND,
    OrthogonalAxes,
    SurfacePrivateAttributes,
    _auto_symmetric_plane_exists_from_bbox,
    _check_axis_is_orthogonal,
    _get_generated_boundary_names,
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360_schema.models.entities.geometry_entities import (
    Edge,
    GeometryBodyGroup,
    SnappyBody,
)
from flow360_schema.models.entities.surface_entities import (
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    GhostSurfacePair,
    ImportedSurface,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
    SurfacePair,
    SurfacePairBase,
    WindTunnelGhostSurface,
    _MirroredEntityBase,
    compute_bbox_tolerance,
)
from flow360_schema.models.entities.volume_entities import (
    AxisymmetricBody,
    Box,
    BoxCache,
    CustomVolume,
    Cylinder,
    GenericVolume,
    SeedpointVolume,
    Sphere,
)
from flow360_schema.models.reference_geometry import ReferenceGeometry
