from typing import List, Literal, Tuple, final

import pydantic as pd

from flow360.component.flow360_params.flow360_fields import (
    IsoSurfaceFieldNames,
    IsoSurfaceFieldNamesFull,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.unit_system import LengthType
from flow360.component.types import Axis


class _OutputItemBase(Flow360BaseModel):
    name: str = pd.Field()

    def __hash__(self):
        return hash(self.name + "-" + self.__class__.__name__)

    def __eq__(self, other):
        if isinstance(other, _OutputItemBase):
            return (self.name + "-" + self.__class__.__name__) == (
                other.name + "-" + other.__class__.__name__
            )
        return False

    def __str__(self):
        return f"{self.__class__.__name__} with name: {self.name}"


class Slice(_OutputItemBase):
    slice_normal: Axis = pd.Field()
    slice_origin: LengthType.Point = pd.Field()


class Isosurface(_OutputItemBase):
    field: Literal[IsoSurfaceFieldNames, IsoSurfaceFieldNamesFull] = pd.Field()
    # TODO: Maybe we need some unit helper function to help user figure out what is the value to use here?
    iso_value: float = pd.Field(description="Expect non-dimensional value.")


class SurfaceList(_OutputItemBase):
    entities: EntityList[Surface] = pd.Field(alias="surfaces")


class Probe(_OutputItemBase):
    locations: List[LengthType.Point] = pd.Field()
