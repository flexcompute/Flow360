from abc import ABCMeta
from typing import Final, Literal, Optional, Tuple, Union, final

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.unit_system import AreaType, LengthType


class ReferenceGeometry(Flow360BaseModel):
    """
    Contains all geometrical related refrence values
    Note:
    - mesh_unit is removed from here and will be a property
    TODO:
    - Support expression for time-dependent axis etc?
    - What about force axis?
    """

    moment_center: Optional[LengthType.Point] = pd.Field()
    moment_length: Optional[Union[LengthType.Positive, LengthType.Moment]] = pd.Field()
    area: Optional[AreaType.Positive] = pd.Field()


class Transformation(Flow360BaseModel):
    """Used in preprocess()/translator to meshing param for volume meshing interface"""

    axis_of_rotation: Optional[Tuple[float, float, float]] = pd.Field()
    angle_of_rotation: Optional[float] = pd.Field()


class _VolumeEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericVolumeZoneType"] = "GenericVolumeZoneType"


class _SurfaceEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericSurfaceZoneType"] = "GenericSurfaceZoneType"


@final
class Face(Flow360BaseModel):
    """
    Collection of the patches that share meshing parameters
    """

    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericFaceType"] = "GenericFaceType"


@final
class Edge(Flow360BaseModel):
    """
    Edge with edge name defined in the geometry file
    """

    ### Warning: Please do not change this as it affects registry bucketing.
    _entity_type: Literal["GenericEdgeType"] = "GenericEdgeType"


@final
class GenericVolume(_VolumeEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    _auto_constructed: Final[bool] = True


@final
class GenericSurface(_SurfaceEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    _auto_constructed: Final[bool] = True


@final
class Box(_VolumeEntityBase):
    center: Tuple[float, float, float] = pd.Field()
    size: Tuple[float, float, float] = pd.Field()
    axes: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = pd.Field()


@final
class Cylinder(_VolumeEntityBase):
    axis: Tuple[float, float, float] = pd.Field()
    center: Tuple[float, float, float] = pd.Field()
    height: float = pd.Field()
    inner_radius: pd.PositiveFloat = pd.Field()
    outer_radius: pd.PositiveFloat = pd.Field()


@final
class Surface(_SurfaceEntityBase):
    # Should inherit from `ReferenceGeometry` but we do not support this from solver side.
    pass


class SurfacePair(Flow360BaseModel):
    pair: Tuple[Surface, Surface]

    @pd.field_validator("pair", mode="after")
    def check_unique(cls, v):
        if v[0].name == v[1].name:
            raise ValueError(f"A surface cannot be paired with itself.")
        return v

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input(cls, input: Union[dict, list, tuple]):
        if isinstance(input, list) or isinstance(input, tuple):
            return dict(pair=input)
        elif isinstance(input, dict):
            return dict(pair=input["pair"])

    def __hash__(self):
        return hash(tuple(sorted([self.pair[0].name, self.pair[1].name])))

    def __eq__(self, other):
        if isinstance(other, SurfacePair):
            return tuple(sorted([self.pair[0].name, self.pair[1].name])) == tuple(
                sorted([other.pair[0].name, other.pair[1].name])
            )
        return False

    def __str__(self):
        return "-".join(sorted([self.pair[0].name, self.pair[1].name]))


VolumeEntityTypes = Union[GenericVolume, Cylinder, Box, str]
