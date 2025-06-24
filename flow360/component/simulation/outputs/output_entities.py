"""Output for simulation."""

from abc import ABCMeta
from typing import Literal, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, generate_uuid
from flow360.component.simulation.outputs.output_fields import IsoSurfaceFieldNames
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


class _PointEntityBase(EntityBase, metaclass=ABCMeta):
    private_attribute_registry_bucket_name: Literal["PointEntityType"] = "PointEntityType"


class Slice(_SliceEntityBase):
    """

    :class:`Slice` class for defining a slice for :class:`~flow360.SliceOutput`.

    Example
    -------

    Define a :class:`Slice` along (0,1,0) direction with the origin of (0,2,0) fl.u.m.

    >>> fl.Slice(
    ...     name="Slice",
    ...     normal=(0, 1, 0),
    ...     origin=(0, 2, 0)*fl.u.m
    ... )

    ====
    """

    private_attribute_entity_type_name: Literal["Slice"] = pd.Field("Slice", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    normal: Axis = pd.Field(description="Normal direction of the slice.")
    # pylint: disable=no-member
    origin: LengthType.Point = pd.Field(description="A single point on the slice.")


class Isosurface(_OutputItemBase):
    """

    :class:`Isosurface` class for defining an isosurface for :class:`~flow360.IsosurfaceOutput`.

    Example
    -------

    Define a :class:`Isosurface` of temperature equal to 1.5 non-dimensional temperature.

    >>> fl.Isosurface(
    ...     name="Isosurface_T_1.5",
    ...     iso_value=1.5,
    ...     field="T",
    ... )

    ====
    """

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
    :class:`Point` class for defining a single point used in various outputs.

    Example
    -------

    >>> fl.Point(
    ...      name="Point",
    ...      location=(1.0, 2.0, 3.0) * fl.u.m,
    ...  )

    ====
    """

    private_attribute_entity_type_name: Literal["Point"] = pd.Field("Point", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    # pylint: disable=no-member
    location: LengthType.Point = pd.Field(description="The coordinate of the point.")


class PointArray(_PointEntityBase):
    """
    :class:`PointArray` class for defining multiple equally spaced monitor points along a line used in various outputs.

    Example
    -------
    Define :class:`PointArray` with 6 equally spaced points along a line starting from
    (0,0,0) * fl.u.m to (1,2,3) * fl.u.m.
    Both the starting and end points are included in the :class:`PointArray`.

    >>> fl.PointArray(
    ...     name="Line_1",
    ...     start=(0.0, 0.0, 0.0) * fl.u.m,
    ...     end=(1.0, 2.0, 3.0) * fl.u.m,
    ...     number_of_points=6,
    ... )

    ====
    """

    private_attribute_entity_type_name: Literal["PointArray"] = pd.Field("PointArray", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    # pylint: disable=no-member
    start: LengthType.Point = pd.Field(description="The starting point of the line.")
    end: LengthType.Point = pd.Field(description="The end point of the line.")
    number_of_points: int = pd.Field(ge=2, description="Number of points along the line.")


class PointArray2D(_PointEntityBase):
    """
    :class:`PointArray2D` class for defining multiple equally spaced points along the u and
    v axes of a parallelogram.


    Example
    -------
    Define :class:`PointArray2D` with points equally distributed on a parallelogram with
    origin (1.0, 0.0, 0.0) * fl.u.m. There are 7 equally spaced points along the parallelogram's u-axis
    of (0.5, 1.0, 0.2) * fl.u.m and 10 equally spaced points along the its v-axis of
    (0.1, 0, 1) * fl.u.m.

    Both the starting and end points are included in the :class:`PointArray`.

    >>> fl.PointArray2D(
    ...     name="Parallelogram_1",
    ...     origin=(1.0, 0.0, 0.0) * fl.u.m,
    ...     u_axis_vector=(0.5, 1.0, 0.2) * fl.u.m,
    ...     v_axis_vector=(0.1, 0, 1) * fl.u.m,
    ...     u_number_of_points=7,
    ...     v_number_of_points=10
    ... )

    ====
    """

    private_attribute_entity_type_name: Literal["PointArray2D"] = pd.Field(
        "PointArray2D", frozen=True
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    # pylint: disable=no-member
    origin: LengthType.Point = pd.Field(description="The corner of the parallelogram.")
    u_axis_vector: LengthType.Axis = pd.Field(description="The scaled u-axis of the parallelogram.")
    v_axis_vector: LengthType.Axis = pd.Field(description="The scaled v-axis of the parallelogram.")
    u_number_of_points: int = pd.Field(ge=2, description="The number of points along the u axis.")
    v_number_of_points: int = pd.Field(ge=2, description="The number of points along the v axis.")
