"""
Primitive type definitions for simulation entities.
"""

from abc import ABCMeta
from typing import Final, Literal, Optional, Tuple, Union, final

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.unit_system import AreaType, LengthType
from flow360.component.types import Axis


class ReferenceGeometry(Flow360BaseModel):
    """
    Contains all geometrical related refrence values
    Note:
    - mesh_unit is removed from here and will be a property
    TODO:
    - Support expression for time-dependent axis etc?
    - What about force axis?
    """

    # pylint: disable=no-member
    moment_center: Optional[LengthType.Point] = pd.Field(None)
    moment_length: Optional[Union[LengthType.Positive, LengthType.Moment]] = pd.Field(None)
    area: Optional[AreaType.Positive] = pd.Field(None)


class Transformation(Flow360BaseModel):
    """Used in preprocess()/translator to meshing param for volume meshing interface"""

    axis_of_rotation: Optional[Axis] = pd.Field()
    angle_of_rotation: Optional[float] = pd.Field()


class _VolumeEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericVolumeZoneType"] = "GenericVolumeZoneType"


class _SurfaceEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericSurfaceZoneType"] = "GenericSurfaceZoneType"


class _EdgeEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericEdgeType"] = "GenericEdgeType"


@final
class Edge(_EdgeEntityBase):
    """
    Edge with edge name defined in the geometry file
    """

    # pylint: disable=invalid-name
    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericEdgeType"] = "GenericEdgeType"


@final
class GenericVolume(_VolumeEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    # pylint: disable=invalid-name
    _auto_constructed: Final[bool] = True


@final
class GenericSurface(_SurfaceEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    # pylint: disable=invalid-name
    _auto_constructed: Final[bool] = True


@final
class Box(_VolumeEntityBase):
    """
    Represents a box in three-dimensional space.

    Attributes:
        center (LengthType.Point): The coordinates of the center of the box.
        size (LengthType.Point): The dimensions of the box (length, width, height).
        axes (Tuple[Axis, Axis]]): The axes of the box.
    """

    # pylint: disable=no-member
    center: LengthType.Point = pd.Field()
    size: LengthType.Point = pd.Field()
    axes: Tuple[Axis, Axis] = pd.Field()


@final
class Cylinder(_VolumeEntityBase):
    """
    Represents a cylinder in three-dimensional space.

    Attributes:
        axis (Axis): The axis of the cylinder.
        center (LengthType.Point): The center point of the cylinder.
        height (LengthType.Postive): The height of the cylinder.
        inner_radius (LengthType.Positive): The inner radius of the cylinder.
        outer_radius (LengthType.Positive): The outer radius of the cylinder.
    """

    axis: Axis = pd.Field()
    # pylint: disable=no-member
    center: LengthType.Point = pd.Field()
    height: LengthType.Positive = pd.Field()
    inner_radius: Optional[LengthType.Positive] = pd.Field(None)
    # pylint: disable=fixme
    # TODO validation outer > inner
    outer_radius: LengthType.Positive = pd.Field()


@final
class Surface(_SurfaceEntityBase):
    """
    Represents a boudary surface in three-dimensional space.
    """

    # pylint: disable=fixme
    # TODO: Should inherit from `ReferenceGeometry` but we do not support this from solver side.


class SurfacePair(Flow360BaseModel):
    """
    Represents a pair of surfaces.

    Attributes:
        pair (Tuple[Surface, Surface]): A tuple containing two Surface objects representing the pair.
    """

    pair: Tuple[Surface, Surface]

    @pd.field_validator("pair", mode="after")
    @classmethod
    def check_unique(cls, v):
        """Check if pairing with self."""
        if v[0].name == v[1].name:
            raise ValueError("A surface cannot be paired with itself.")
        return v

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input(cls, input_data: Union[dict, list, tuple]):
        if isinstance(input_data, (list, tuple)):
            return {"pair": input_data}
        if isinstance(input_data, dict):
            return {"pair": input_data["pair"]}
        raise ValueError("Invalid input data.")

    def __hash__(self):
        return hash(tuple(sorted([self.pair[0].name, self.pair[1].name])))

    def __eq__(self, other):
        if isinstance(other, SurfacePair):
            return tuple(sorted([self.pair[0].name, self.pair[1].name])) == tuple(
                sorted([other.pair[0].name, other.pair[1].name])
            )
        return False

    def __str__(self):
        return ",".join(sorted([self.pair[0].name, self.pair[1].name]))


VolumeEntityTypes = Union[GenericVolume, Cylinder, Box, str]
