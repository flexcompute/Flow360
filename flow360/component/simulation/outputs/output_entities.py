"""Output for simulation."""

from abc import ABCMeta
from typing import List, Literal

import pydantic as pd

from flow360.component.flow360_params.flow360_fields import (
    IsoSurfaceFieldNames,
    IsoSurfaceFieldNamesFull,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
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


class _SliceEntityBase(EntityBase, metaclass=ABCMeta):
    private_attribute_registry_bucket_name: Literal["SliceEntityType"] = "SliceEntityType"


class _ProbeGroupEntityBase(EntityBase, metaclass=ABCMeta):
    private_attribute_registry_bucket_name: Literal["ProbeEntityType"] = "ProbeEntityType"


class Slice(_SliceEntityBase):
    """Slice output item."""

    private_attribute_entity_type_name: Literal["Slice"] = pd.Field("Slice", frozen=True)
    normal: Axis = pd.Field()
    # pylint: disable=no-member
    origin: LengthType.Point = pd.Field()


class Isosurface(_OutputItemBase):
    """Isosurface output item."""

    field: Literal[IsoSurfaceFieldNames, IsoSurfaceFieldNamesFull] = pd.Field()
    # pylint: disable=fixme
    # TODO: Maybe we need some unit helper function to help user figure out what is the value to use here?
    iso_value: float = pd.Field(description="Expect non-dimensional value.")


class ProbeGroup(_ProbeGroupEntityBase):
    """A group of coordinates that are used to probe the solution."""

    private_attribute_entity_type_name: Literal["ProbeGroup"] = pd.Field("ProbeGroup", frozen=True)
    # pylint: disable=no-member
    locations: List[LengthType.Point] = pd.Field()

    def from_csv_file(self):
        """Load group of probe coordinates from csv file."""
