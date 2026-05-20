"""Output entity definitions: Point, PointArray, PointArray2D, Slice, Isosurface."""
# mypy: disable-error-code="import-not-found"

from __future__ import annotations

from typing import Any, Literal

import pydantic as pd
import unyt as u

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_operation import (
    _transform_direction,
    _transform_point,
)
from flow360_schema.framework.entity.entity_utils import generate_uuid
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.expression import (
    Expression,
    UnytQuantity,
    UserVariable,
    ValueOrExpression,
    get_input_value_dimensions,
    get_input_value_length,
    solver_variable_to_user_variable,
)
from flow360_schema.framework.expression.utils import is_runtime_expression
from flow360_schema.framework.physical_dimensions import Length

NDArray = Any

# IsoSurfaceFieldNames: pure Literal type, no external dependency.
IsoSurfaceFieldNames = Literal[
    "Mach",
    "qcriterion",
    "s",
    "T",
    "Cp",
    "Cpt",
    "mut",
    "nuHat",
    "vorticityMagnitude",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
    "velocity_magnitude",
    "velocity_x",
    "velocity_y",
    "velocity_z",
]


def _should_skip_unit_system_inference(value: Any) -> bool:
    """[Frontend] Check whether unit inference should be skipped for this value."""
    return (
        not isinstance(value, dict)
        or "units" not in value
        or value["units"]
        not in (
            "SI_unit_system",
            "Imperial_unit_system",
            "CGS_unit_system",
        )
    )


def _infer_units_by_unit_system(value: dict[str, Any], unit_system: str, value_dimensions: Any) -> dict[str, Any]:
    """[Frontend] Infer the units based on the unit system."""
    unit_systems = {
        "SI_unit_system": u.unit_systems.mks_unit_system,
        "Imperial_unit_system": u.unit_systems.imperial_unit_system,
        "CGS_unit_system": u.unit_systems.cgs_unit_system,
    }
    value["units"] = str(unit_systems[unit_system][value_dimensions])
    return value


class _OutputItemBase(Flow360BaseModel):
    name: str = pd.Field()

    def __hash__(self) -> int:
        return hash(self.name + "-" + self.__class__.__name__)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _OutputItemBase):
            return (self.name + "-" + self.__class__.__name__) == (other.name + "-" + other.__class__.__name__)
        return False

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with name: {self.name}"


class Slice(EntityBase):
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
    origin: Length.Vector3 = pd.Field(description="A single point on the slice.")  # type: ignore[valid-type]

    def _apply_transformation(self, matrix: NDArray) -> Slice:
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        import numpy as np

        origin_array = np.asarray(self.origin.value)  # type: ignore[attr-defined]
        new_origin_array = _transform_point(origin_array, matrix)
        new_origin = type(self.origin)(new_origin_array, self.origin.units)  # type: ignore[attr-defined, misc]

        normal_array = np.asarray(self.normal)
        transformed_normal = _transform_direction(normal_array, matrix)
        new_normal = tuple(transformed_normal / np.linalg.norm(transformed_normal))

        return self.model_copy(update={"origin": new_origin, "normal": new_normal})


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

    field: IsoSurfaceFieldNames | str | UserVariable = pd.Field(
        description="Isosurface field variable. One of :code:`p`, :code:`rho`, "
        ":code:`Mach`, :code:`qcriterion`, :code:`s`, :code:`T`, :code:`Cp`, :code:`mut`,"
        " :code:`nuHat` or one of scalar field defined in :class:`UserDefinedField`."
    )
    iso_value: ValueOrExpression[UnytQuantity | float] = pd.Field(
        description="Expect non-dimensional value.",
    )

    wall_distance_clip_threshold: Length.PositiveFloat64 | None = pd.Field(  # type: ignore[valid-type]
        default=None,
        description="Optional parameter to remove the isosurface within a specified distance from walls.",
    )

    @pd.field_validator("field", mode="before")
    @classmethod
    def _preprocess_expression_and_solver_variable(cls, value: Any) -> Any:
        if isinstance(value, Expression):
            raise ValueError(
                f"Expression ({value}) cannot be directly used as isosurface field, "
                "please define a UserVariable first."
            )
        return solver_variable_to_user_variable(value)

    @pd.field_validator("iso_value", mode="before")
    @classmethod
    def _preprocess_field_with_unit_system(cls, value: Any, info: pd.ValidationInfo) -> Any:
        if _should_skip_unit_system_inference(value):
            return value
        if info.data.get("field") is None:
            # `field` validation failed.
            raise ValueError("The isosurface field is invalid and therefore unit inference is not possible.")
        units = value["units"]
        field = info.data["field"]
        field_dimensions = get_input_value_dimensions(value=field)
        value = _infer_units_by_unit_system(value=value, value_dimensions=field_dimensions, unit_system=units)
        return value

    @pd.field_validator("field", mode="after")
    @classmethod
    def check_expression_length(
        cls,
        v: IsoSurfaceFieldNames | str | UserVariable,
    ) -> IsoSurfaceFieldNames | str | UserVariable:
        """Ensure the isofield is a scalar."""
        if isinstance(v, UserVariable) and len(v) != 0:
            raise ValueError(f"The isosurface field ({v}) must be defined with a scalar variable.")
        return v

    @pd.field_validator("field", mode="after")
    @classmethod
    def check_runtime_expression(
        cls,
        v: IsoSurfaceFieldNames | str | UserVariable,
    ) -> IsoSurfaceFieldNames | str | UserVariable:
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
    def check_single_iso_value(
        cls,
        v: ValueOrExpression[UnytQuantity | float],
    ) -> ValueOrExpression[UnytQuantity | float]:
        """Ensure the iso_value is a single value."""
        if get_input_value_length(v) == 0:
            return v
        raise ValueError(f"The iso_value ({v}) must be a scalar.")

    @pd.field_validator("iso_value", mode="after")
    @classmethod
    def check_iso_value_dimensions(
        cls,
        v: ValueOrExpression[UnytQuantity | float],
        info: pd.ValidationInfo,
    ) -> ValueOrExpression[UnytQuantity | float]:
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
    def check_iso_value_for_string_field(
        cls,
        v: ValueOrExpression[UnytQuantity | float],
        info: pd.ValidationInfo,
    ) -> ValueOrExpression[UnytQuantity | float]:
        """Ensure the iso_value is float when string field is used."""
        field = info.data.get("field", None)
        if isinstance(field, str) and not isinstance(v, float):
            raise ValueError(
                f"The isosurface field ({field}) specified by string "
                "can only be used with a nondimensional iso_value."
            )
        return v


class Point(EntityBase):
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
    location: Length.Vector3 = pd.Field(description="The coordinate of the point.")  # type: ignore[valid-type]

    def _apply_transformation(self, matrix: NDArray) -> Point:
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        import numpy as np

        location_array = np.asarray(self.location.value)  # type: ignore[attr-defined]
        new_location_array = _transform_point(location_array, matrix)
        new_location = type(self.location)(new_location_array, self.location.units)  # type: ignore[attr-defined, misc]
        return self.model_copy(update={"location": new_location})


class PointArray(EntityBase):
    """
    :class:`PointArray` class for defining multiple equally spaced monitor points along a line.

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
    start: Length.Vector3 = pd.Field(description="The starting point of the line.")  # type: ignore[valid-type]
    end: Length.Vector3 = pd.Field(description="The end point of the line.")  # type: ignore[valid-type]
    number_of_points: int = pd.Field(ge=2, description="Number of points along the line.")

    def _apply_transformation(self, matrix: NDArray) -> PointArray:
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        import numpy as np

        start_array = np.asarray(self.start.value)  # type: ignore[attr-defined]
        end_array = np.asarray(self.end.value)  # type: ignore[attr-defined]

        new_start_array = _transform_point(start_array, matrix)
        new_end_array = _transform_point(end_array, matrix)

        new_start = type(self.start)(new_start_array, self.start.units)  # type: ignore[attr-defined, misc]
        new_end = type(self.end)(new_end_array, self.end.units)  # type: ignore[attr-defined, misc]

        return self.model_copy(update={"start": new_start, "end": new_end})


class PointArray2D(EntityBase):
    """
    :class:`PointArray2D` class for defining multiple equally spaced points along the u and
    v axes of a parallelogram.

    Example
    -------
    Define :class:`PointArray2D` with points equally distributed on a parallelogram with
    origin (1.0, 0.0, 0.0) * fl.u.m. There are 7 equally spaced points along the parallelogram's
    u-axis of (0.5, 1.0, 0.2) * fl.u.m and 10 equally spaced points along its v-axis of
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

    private_attribute_entity_type_name: Literal["PointArray2D"] = pd.Field("PointArray2D", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    origin: Length.Vector3 = pd.Field(description="The corner of the parallelogram.")  # type: ignore[valid-type]
    u_axis_vector: Length.NonNullVector3 = pd.Field(description="The scaled u-axis of the parallelogram.")  # type: ignore[valid-type]
    v_axis_vector: Length.NonNullVector3 = pd.Field(description="The scaled v-axis of the parallelogram.")  # type: ignore[valid-type]
    u_number_of_points: int = pd.Field(ge=2, description="The number of points along the u axis.")
    v_number_of_points: int = pd.Field(ge=2, description="The number of points along the v axis.")

    def _apply_transformation(self, matrix: NDArray) -> PointArray2D:
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        import numpy as np

        origin_array = np.asarray(self.origin.value)  # type: ignore[attr-defined]
        new_origin_array = _transform_point(origin_array, matrix)
        new_origin = type(self.origin)(new_origin_array, self.origin.units)  # type: ignore[attr-defined, misc]

        u_axis_array = np.asarray(self.u_axis_vector.value)  # type: ignore[attr-defined]
        v_axis_array = np.asarray(self.v_axis_vector.value)  # type: ignore[attr-defined]

        new_u_axis_array = _transform_direction(u_axis_array, matrix)
        new_v_axis_array = _transform_direction(v_axis_array, matrix)

        new_u_axis = type(self.u_axis_vector)(new_u_axis_array, self.u_axis_vector.units)  # type: ignore[attr-defined, misc]
        new_v_axis = type(self.v_axis_vector)(new_v_axis_array, self.v_axis_vector.units)  # type: ignore[attr-defined, misc]

        return self.model_copy(update={"origin": new_origin, "u_axis_vector": new_u_axis, "v_axis_vector": new_v_axis})
