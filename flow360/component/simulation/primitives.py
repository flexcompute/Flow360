"""Relay primitive entity types and ReferenceGeometry from schema."""

# pylint: disable=unused-import
from fnmatch import fnmatchcase

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
from flow360_schema.models.entities.geometry_entities import Edge, GeometryBodyGroup

try:
    from flow360_schema.models.entities.geometry_entities import SnappyBody
except ImportError:
    SnappyBody = None
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


class _SnappyBodySelection:
    """Collection of snappy bodies addressable by region pattern."""

    def __init__(self, bodies):
        self._bodies = bodies

    def __len__(self):
        """Return the number of selected snappy bodies."""

        return len(self._bodies)

    def __iter__(self):
        """Iterate over selected snappy bodies."""

        return iter(self._bodies)

    @property
    def name(self):
        """Return a display name for the selected snappy bodies."""

        return ", ".join(body.name for body in self._bodies)

    def __getitem__(self, pattern):
        surfaces = []
        for body in self._bodies:
            surfaces.extend(body._matching_surfaces(pattern))  # pylint: disable=protected-access
        if not surfaces:
            raise ValueError(
                f"No entity found in registry for parent entities: {self.name} "
                f"with given name/naming pattern: '{pattern}'."
            )
        return surfaces


if SnappyBody is None:

    class SnappyBody:  # pylint: disable=function-redefined,too-few-public-methods
        """Compatibility wrapper for schemas that removed SnappyBody."""

        def __init__(self, name, surfaces):
            self.name = name
            self._surfaces = list(surfaces)

        def _matching_surfaces(self, pattern):
            return [surface for surface in self._surfaces if fnmatchcase(surface.name, pattern)]

        def __getitem__(self, pattern):
            surfaces = self._matching_surfaces(pattern)
            if not surfaces:
                raise KeyError(pattern)
            return surfaces


class SnappyBodyRegistry:
    """Minimal snappy body registry for newer flow360-schema versions."""

    def __init__(self, bodies):
        self._bodies = list(bodies)

    def __iter__(self):
        """Iterate over registered snappy bodies."""

        return iter(self._bodies)

    def __getitem__(self, pattern):
        matches = [body for body in self._bodies if fnmatchcase(body.name, pattern)]
        if not matches:
            raise ValueError(
                f"No entity found in registry with given name/naming pattern: '{pattern}'."
            )
        if len(matches) == 1:
            return matches[0]
        return _SnappyBodySelection(matches)
