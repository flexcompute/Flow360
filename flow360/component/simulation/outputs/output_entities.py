"""Output for simulation."""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd
import unyt as u

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, generate_uuid
from flow360.component.simulation.outputs.output_fields import IsoSurfaceFieldNames
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UnytQuantity,
    UserVariable,
    ValueOrExpression,
    get_input_value_dimensions,
    get_input_value_length,
    solver_variable_to_user_variable,
)
from flow360.component.simulation.user_code.core.utils import is_runtime_expression
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
    ...     wallDistanceClipThreshold=0.005 * fl.u.m, (optional)
    ... )

    ====
    """

    field: Union[IsoSurfaceFieldNames, str, UserVariable] = pd.Field(
        description="Isosurface field variable. One of :code:`p`, :code:`rho`, "
        ":code:`Mach`, :code:`qcriterion`, :code:`s`, :code:`T`, :code:`Cp`, :code:`mut`,"
        " :code:`nuHat` or one of scalar field defined in :class:`UserDefinedField`."
    )
    # pylint: disable=fixme
    iso_value: ValueOrExpression[Union[UnytQuantity, float]] = pd.Field(
        description="Expect non-dimensional value.",
    )

    # pylint: disable=no-member
    wall_distance_clip_threshold: Optional[LengthType.Positive] = pd.Field(
        default=None,
        description="Optional parameter to remove the isosurface within a specified distance from walls.",
    )

    @pd.field_validator("field", mode="before")
    @classmethod
    def _preprocess_expression_and_solver_variable(cls, value):
        if isinstance(value, Expression):
            raise ValueError(
                f"Expression ({value}) cannot be directly used as isosurface field, "
                "please define a UserVariable first."
            )
        return solver_variable_to_user_variable(value)

    @pd.field_validator("iso_value", mode="before")
    @classmethod
    def _preprocess_field_with_unit_system(cls, value, info: pd.ValidationInfo):
        if (
            not isinstance(value, dict)
            or "units" not in value
            or value["units"]
            not in (
                "SI_unit_system",
                "Imperial_unit_system",
                "CGS_unit_system",
            )
        ):
            return value
        if info.data.get("field") is None:
            # `field` validation failed.
            raise ValueError(
                "The isosurface field is invalid and therefore unit inference is not possible."
            )
        units = value["units"]
        field = info.data["field"]
        field_dimensions = get_input_value_dimensions(value=field)
        if units == "SI_unit_system":
            value["units"] = u.unit_systems.mks_unit_system[field_dimensions]
        if units == "Imperial_unit_system":
            value["units"] = u.unit_systems.imperial_unit_system[field_dimensions]
        if units == "CGS_unit_system":
            value["units"] = u.unit_systems.cgs_unit_system[field_dimensions]
        return value

    @pd.field_validator("field", mode="after")
    @classmethod
    def check_expression_length(cls, v):
        """Ensure the isofield is a scalar."""
        if isinstance(v, UserVariable) and len(v) != 0:
            raise ValueError(f"The isosurface field ({v}) must be defined with a scalar variable.")
        return v

    @pd.field_validator("field", mode="after")
    @classmethod
    def check_runtime_expression(cls, v):
        """Ensure the isofield is a runtime expression but not a constant value."""
        if isinstance(v, UserVariable):
            if not isinstance(v.value, Expression):
                raise ValueError(f"The isosurface field ({v}) cannot be a constant value.")
            try:
                result = v.value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            except Exception as err:
                raise ValueError(f"expression evaluation failed for the isofield: {err}") from err
            if not is_runtime_expression(result):
                raise ValueError(f"The isosurface field ({v}) cannot be a constant value.")
        return v

    @pd.field_validator("iso_value", mode="after")
    @classmethod
    def check_single_iso_value(cls, v):
        """Ensure the iso_value is a single value."""
        if get_input_value_length(v) == 0:
            return v
        raise ValueError(f"The iso_value ({v}) must be a scalar.")

    @pd.field_validator("iso_value", mode="after")
    @classmethod
    def check_iso_value_dimensions(cls, v, info: pd.ValidationInfo):
        """Ensure the iso_value has the same dimensions as the field."""

        field = info.data.get("field", None)
        if not isinstance(field, UserVariable):
            return v
        value_dimensions = get_input_value_dimensions(value=v)
        if value_dimensions is None:
            return v
        field_dimensions = get_input_value_dimensions(value=field)
        if field_dimensions != value_dimensions:
            raise ValueError(
                f"The iso_value ({v}, dimensions:{value_dimensions}) should have the same dimensions as "
                f"the isosurface field ({field}, dimensions: {field_dimensions})."
            )
        return v

    @pd.field_validator("iso_value", mode="after")
    @classmethod
    def check_iso_value_for_string_field(cls, v, info: pd.ValidationInfo):
        """Ensure the iso_value is float when string field is used."""

        field = info.data.get("field", None)
        if isinstance(field, str) and not isinstance(v, float):
            raise ValueError(
                f"The isosurface field ({field}) specified by string "
                "can only be used with a nondimensional iso_value."
            )
        return v


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
