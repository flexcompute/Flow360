from abc import ABCMeta
from typing import Literal, Optional, Tuple

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase


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


class GenericVolume(_VolumeEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    _auto_constructed = True

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        raise TypeError(f"Subclassing of {cls.__name__} is not allowed")


class GenericSurface(_SurfaceEntityBase):
    """Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata."""

    _auto_constructed = True

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        raise TypeError(f"Subclassing of {cls.__name__} is not allowed")


class Box(_VolumeEntityBase):
    center: Tuple[float, float, float] = pd.Field()
    size: Tuple[float, float, float] = pd.Field()
    axes: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = pd.Field()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        raise TypeError(f"Subclassing of {cls.__name__} is not allowed")


class Cylinder(_VolumeEntityBase):
    axis: Tuple[float, float, float] = pd.Field()
    center: Tuple[float, float, float] = pd.Field()
    height: float = pd.Field()
    inner_radius: pd.PositiveFloat = pd.Field()
    outer_radius: pd.PositiveFloat = pd.Field()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        raise TypeError(f"Subclassing of {cls.__name__} is not allowed")
