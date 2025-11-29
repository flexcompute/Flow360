from typing import ClassVar, Literal
from flow360.component.simulation.framework.entity_base import EntityBase
import pydantic as pd
from flow360.component.simulation.framework.entity_base import generate_uuid
from flow360.component.simulation.unit_system import LengthType
from flow360.component.types import Axis


class MirrorPlane(EntityBase):
    """
    :class:`MirrorPlane` class for defining a mirror plane for mirroring entities.

     Example
     -------

     >>> fl.MirrorPlane(
     ...     name="MirrorPlane",
     ...     normal=(0, 1, 0),
     ...     center=(0, 0, 0)*fl.u.m
     ... )
    """

    name: str = pd.Field()
    normal: Axis = pd.Field(description="Normal direction of the plane.")
    center: LengthType.Point = pd.Field(description="Center point of the plane.")

    private_attribute_entity_type_name: Literal["MirrorPlane"] = pd.Field(
        "MirrorPlane", frozen=True
    )
    entity_bucket: ClassVar[str] = "MirrorPlaneType"
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)


class MirroredGeometryBodyGroup(EntityBase):
    """
    :class:`MirroredGeometryBodyGroup` class for representing a mirrored geometry body group.
    """

    name: str = pd.Field()
    geometry_body_group_id: str = pd.Field(description="ID of the geometry body group to mirror.")
    mirror_plane_id: str = pd.Field(
        description="ID of the mirror plane to mirror the geometry body group."
    )

    private_attribute_entity_type_name: Literal["MirroredGeometryBodyGroup"] = pd.Field(
        "MirroredGeometryBodyGroup", frozen=True
    )
    entity_bucket: ClassVar[str] = "MirroredGeometryBodyGroupType"
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)


class MirroredSurface(EntityBase):
    """
    :class:`MirroredSurface` class for representing a mirrored surface.
    """

    # TODO: What if the surface grouping groups faces across multiple geometry body groups?
    # TODO: Then mirroring a geometry body group does not generate a corresponding new mirrored surface?

    name: str = pd.Field()
    surface_id: str = pd.Field(description="ID of the surface to mirror.")
    mirror_plane_id: str = pd.Field(description="ID of the mirror plane to mirror the surface.")

    private_attribute_entity_type_name: Literal["MirroredSurface"] = pd.Field(
        "MirroredSurface", frozen=True
    )
    entity_bucket: ClassVar[str] = "MirroredSurfaceType"
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
