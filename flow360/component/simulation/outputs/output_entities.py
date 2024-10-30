"""Output for simulation."""

from abc import ABCMeta
from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.unit_system import LengthType
from flow360.component.types import Axis
from flow360.component.v1.flow360_fields import IsoSurfaceFieldNames


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


class _SliceEntityBase(EntityBase, metaclass=ABCMeta):
    private_attribute_registry_bucket_name: Literal["SliceEntityType"] = "SliceEntityType"


class _PointEntityBase(EntityBase, metaclass=ABCMeta):
    private_attribute_registry_bucket_name: Literal["PointEntityType"] = "PointEntityType"


class Slice(_SliceEntityBase):
    """Slice output item."""

    private_attribute_entity_type_name: Literal["Slice"] = pd.Field("Slice", frozen=True)
    normal: Axis = pd.Field()
    # pylint: disable=no-member
    origin: LengthType.Point = pd.Field()


class Isosurface(_OutputItemBase):
    """Isosurface output item."""

    field: Literal[IsoSurfaceFieldNames] = pd.Field()
    # pylint: disable=fixme
    # TODO: Maybe we need some unit helper function to help user figure out what is the value to use here?
    iso_value: float = pd.Field(description="Expect non-dimensional value.")


class Point(_PointEntityBase):
    """A single point for probe output"""

    private_attribute_entity_type_name: Literal["Point"] = pd.Field("Point", frozen=True)
    # pylint: disable=no-member
    location: LengthType.Point = pd.Field()


class PointArray(_PointEntityBase):
    """A single point for probe output"""

    private_attribute_entity_type_name: Literal["PointArray"] = pd.Field("PointArray", frozen=True)
    # pylint: disable=no-member
    start: LengthType.Point = pd.Field()
    end: LengthType.Point = pd.Field()
    number_of_points: int = pd.Field(gt=2)
