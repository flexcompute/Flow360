"""Output for simulation."""

from abc import ABCMeta
from typing import Literal, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, generate_uuid
from flow360.component.simulation.unit_system import LengthType
from flow360.component.types import Axis

# pylint: disable=duplicate-code
# inlined from v1 module to avoid circular import
IsoSurfaceFieldNames = Literal[
    "p", "rho", "Mach", "qcriterion", "s", "T", "Cp", "mut", "nuHat", "Cpt"
]


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
    """:class:`Slice` class for defining a slice for :class:`~flow360.SliceOutput`."""

    private_attribute_entity_type_name: Literal["Slice"] = pd.Field("Slice", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    normal: Axis = pd.Field(description="Normal direction of the slice.")
    # pylint: disable=no-member
    origin: LengthType.Point = pd.Field(description="A single point on the slice.")


class Isosurface(_OutputItemBase):
    """:class:`Isosurface` class for defining an isosurface for :class:`~flow360.IsosurfaceOutput`."""

    field: Union[IsoSurfaceFieldNames, str] = pd.Field(
        description="Isosurface field variable. One of :code:`p`, :code:`rho`, "
        ":code:`Mach`, :code:`qcriterion`, :code:`s`, :code:`T`, :code:`Cp`, :code:`mut`,"
        " :code:`nuHat` or one of scalar field defined in :class:`UserDefinedField`."
    )
    # pylint: disable=fixme
    # TODO: Maybe we need some unit helper function to help user figure out what is the value to use here?
    iso_value: float = pd.Field(description="Expect non-dimensional value.")


class Point(_PointEntityBase):
    """
    :class:`Point` class for defining a single point for
    :class:`~flow360.ProbeOutput`/:class:`~flow360.SurfaceProbeOutput`.
    """

    private_attribute_entity_type_name: Literal["Point"] = pd.Field("Point", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    # pylint: disable=no-member
    location: LengthType.Point = pd.Field(description="The coordinate of the point.")


class PointArray(_PointEntityBase):
    """
    :class:`PointArray` class for defining a line for
    :class:`~flow360.ProbeOutput`/:class:`~flow360.SurfaceProbeOutput`.
    """

    private_attribute_entity_type_name: Literal["PointArray"] = pd.Field("PointArray", frozen=True)
    # pylint: disable=no-member
    start: LengthType.Point = pd.Field(description="The starting point of the line.")
    end: LengthType.Point = pd.Field(description="The end point of the line.")
    number_of_points: int = pd.Field(gt=2, description="Number of points along the line.")
