"""
Primitive type definitions for simulation entities.

Re-import relay: entity classes are defined in concrete modules under
flow360_schema.models.entities.
Only ReferenceGeometry and VolumeEntityTypes remain client-owned.
"""

# pylint: disable=unused-import
from typing import Optional, Union

import pydantic as pd
from flow360_schema.framework.physical_dimensions import Area, Length
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

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.user_code.core.types import ValueOrExpression

VolumeEntityTypes = Union[GenericVolume, Cylinder, Sphere, Box, str]


class ReferenceGeometry(Flow360BaseModel):
    """
    :class:`ReferenceGeometry` class contains all geometrical related reference values.

    Example
    -------
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=(1, 1, 1) * u.m,
    ...     area=1.5 * u.m**2
    ... )
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=1 * u.m,
    ...     area=1.5 * u.m**2
    ... )  # Equivalent to above

    ====
    """

    # pylint: disable=no-member
    moment_center: Optional[Length.Vector3] = pd.Field(
        None, description="The x, y, z coordinate of moment center."
    )
    moment_length: Optional[Union[Length.PositiveFloat64, Length.PositiveVector3]] = pd.Field(
        None, description="The x, y, z component-wise moment reference lengths."
    )
    area: Optional[ValueOrExpression[Area.PositiveFloat64]] = pd.Field(
        None, description="The reference area of the geometry."
    )
    private_attribute_area_settings: Optional[dict] = pd.Field(None)

    @classmethod
    def fill_defaults(cls, ref, params):  # type: ignore[override]
        """Return a new ReferenceGeometry with defaults filled using SimulationParams.

        Defaults when missing or when ref is None:
        - area: 1 * (base_length)**2
        - moment_center: (0,0,0) * base_length
        - moment_length: (1,1,1) * base_length
        """
        # Determine base length unit from params
        base_length_unit = params.base_length  # LengthType quantity

        # Start from provided or empty
        if ref is None:
            ref = cls()

        # Compose output using provided values when available
        area = ref.area
        if area is None:
            area = 1.0 * (base_length_unit**2)

        moment_center = ref.moment_center
        if moment_center is None:
            moment_center = (0, 0, 0) * base_length_unit

        moment_length = ref.moment_length
        if moment_length is None:
            moment_length = (1.0, 1.0, 1.0) * base_length_unit

        return cls(area=area, moment_center=moment_center, moment_length=moment_length)
