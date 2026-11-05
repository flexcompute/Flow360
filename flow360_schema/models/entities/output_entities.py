"""Output entity definitions: Point, PointArray, PointArray2D, Slice, Isosurface."""
# mypy: disable-error-code="import-not-found"

from __future__ import annotations

import csv
import warnings
from typing import Any, Literal

import pydantic as pd
import unyt as u

from flow360_schema.exceptions import Flow360FileError, Flow360ValueError
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_operation import (
    TransformableEntity,
    _transform_direction,
    _transform_point,
    _transform_points,
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
from flow360_schema.models.entities.volume_entities import Box

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


class Slice(TransformableEntity, EntityBase):
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
    ...     # optional: drop the isosurface within 0.005 m of any wall
    ...     wall_distance_clip_threshold=0.005 * fl.u.m,
    ...     # optional: keep only the isosurface inside this box
    ...     clipping_box=fl.Box(
    ...         name="clip_box", center=(0, 0, 0) * fl.u.m, size=(1, 1, 1) * fl.u.m
    ...     ),
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
    clipping_box: Box | None = pd.Field(
        default=None,
        description="Optional :class:`~flow360.Box` that limits the isosurface to the region inside the box.",
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


class Point(TransformableEntity, EntityBase):
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

    @property
    def _monitor_key_names(self) -> list[str]:
        """The `monitorLocations` keys this entity writes, in the order it writes them.

        The solver receives one flat dict of monitors per probe output, so an entity
        standing for several points expands into several keys. Two entities that expand
        onto the same key would silently share one monitor, which is why the probe-output
        validator compares these instead of raw entity names, and why the translator
        derives its keys from here rather than spelling the convention out a second time.
        """
        return [self.name]

    def _apply_transformation(self, matrix: NDArray) -> Point:
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        import numpy as np

        location_array = np.asarray(self.location.value)  # type: ignore[attr-defined]
        new_location_array = _transform_point(location_array, matrix)
        new_location = type(self.location)(new_location_array, self.location.units)  # type: ignore[attr-defined, misc]
        return self.model_copy(update={"location": new_location})


def _coordinate_column_indices(
    file_path: str, header: list[str], coordinate_columns: tuple[str, str, str], delimiter: str
) -> tuple[int, ...]:
    """Locate the three named coordinate columns in a CSV header row."""
    if len(set(coordinate_columns)) != len(coordinate_columns):
        raise Flow360ValueError(
            f"`coordinate_columns` {list(coordinate_columns)} names the same column more than once, "
            f"so two axes would read one column. Name three distinct columns."
        )

    # Only a repeat among the requested names is ambiguous. An export's other duplicate
    # columns -- the empty trailing pair a spreadsheet likes to add, most of all -- are
    # never read, so rejecting the file over them would refuse a file that reads fine.
    repeated = sorted({column for column in coordinate_columns if header.count(column) > 1})
    if repeated:
        raise Flow360FileError(
            f"{file_path}: header repeats the column name(s) {repeated}, so a coordinate column "
            f"cannot be identified unambiguously. Rename the repeats and try again."
        )

    missing = [column for column in coordinate_columns if column not in header]
    if missing:
        raise Flow360FileError(
            f"{file_path}: coordinate column(s) {missing} not found; the header is {header} "
            f"(delimiter={delimiter!r}). If the file uses a different separator, pass `delimiter`."
        )

    return tuple(header.index(column) for column in coordinate_columns)


class PointGroup(TransformableEntity, EntityBase):
    """
    :class:`PointGroup` class for defining many monitor points under a single name.

    Example
    -------

    Import a set of probe locations from a CSV export. This is what the class is for:
    a file of experimental probe coordinates becomes one entity instead of one entity
    per row.

    >>> group = fl.PointGroup.from_csv(
    ...     "probes.csv",
    ...     name="Rake A",
    ...     unit=fl.u.mm,
    ...     coordinate_columns=("x_coord", "y_coord", "z_coord"),
    ... )
    >>> fl.ProbeOutput(name="probes", entities=[group], output_fields=["Cp"])

    Or list the coordinates directly:

    >>> fl.PointGroup(
    ...     name="Rake A",
    ...     locations=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]] * fl.u.m,
    ... )

    ====
    """

    private_attribute_entity_type_name: Literal["PointGroup"] = pd.Field("PointGroup", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    locations: Length.Vector3Array = pd.Field(description="The coordinates of the points in the group.")  # type: ignore[valid-type]

    @pd.field_validator("locations", mode="after")
    @classmethod
    def _reject_duplicate_locations(cls, value: Any) -> Any:
        """Reject repeated coordinates.

        Probing one coordinate twice produces two identical monitors, so a repeat is
        always a mistake in the input rather than something to honour. The rule lives
        here and not on the `Vector3Array` type because it is about what this class
        models -- a list of velocities, or a path that revisits a point, would be
        entitled to repeat.
        """
        import numpy as np

        rows = np.asarray(value.value)
        _, inverse, counts = np.unique(rows, axis=0, return_inverse=True, return_counts=True)
        repeated = np.nonzero(counts > 1)[0]
        if repeated.size == 0:
            return value

        shown = [
            f"rows {np.nonzero(inverse == group)[0].tolist()} all equal "
            f"{tuple(rows[np.nonzero(inverse == group)[0][0]].tolist())}"
            for group in repeated[:3]
        ]
        remainder = "" if repeated.size <= 3 else f", and {repeated.size - 3} further repeated coordinate(s)"
        raise ValueError(
            f"`locations` must not repeat a coordinate: {'; '.join(shown)}{remainder}. "
            "Probing the same coordinate twice yields nothing; remove the repeats "
            "(`PointGroup.from_csv` drops them for you)."
        )

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        *,
        name: str,
        unit: Any,
        coordinate_columns: tuple[str, str, str] | None = None,
        delimiter: str = ",",
    ) -> PointGroup:
        """Build a :class:`PointGroup` from the coordinates in a CSV file.

        The shape of the file is stated by the caller, never guessed. Pass
        ``coordinate_columns`` and the first row is read as a header from which exactly
        those three columns are taken; omit it and the file is read as headerless rows
        of exactly three columns. Columns that are not selected are ignored, so an
        export carrying a ``name`` or ``id`` column needs no preparation.

        Exact duplicate coordinates are dropped in first-occurrence order and reported
        as a ``UserWarning``, since a repeated probe location yields nothing.

        Parameters
        ----------
        file_path : str
            Path to the CSV file. Read as ``utf-8-sig``, so a spreadsheet export that
            starts with a byte-order mark is read the same as plain UTF-8.
        name : str
            Name of the resulting group. Monitor keys are ``{name}_{index}``.
        unit : unyt.Unit
            Unit of every coordinate in the file, for example ``fl.u.mm``. The file
            carries no unit of its own, so this is required.
        coordinate_columns : tuple[str, str, str], optional
            Header names of the x, y and z columns, which must be three distinct names.
            Columns the header repeats but these do not name are never read. Omit for a
            headerless file.
        delimiter : str, default ","
            Field separator, for exports that use ``;`` or a tab.

        Returns
        -------
        :class:`PointGroup`
            A group holding one point per surviving row.

        Example
        -------
        >>> fl.PointGroup.from_csv(
        ...     "probes.csv",
        ...     name="Rake A",
        ...     unit=fl.u.mm,
        ...     coordinate_columns=("x_coord", "y_coord", "z_coord"),
        ... )

        A headerless file of exactly three columns needs nothing but the unit:

        >>> fl.PointGroup.from_csv("bare.csv", name="Rake A", unit=fl.u.mm)
        """
        with open(file_path, newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter)
            # `line_num` is the file line the record ends on, captured per row so every
            # message points at the line the user sees in their editor. Rows the reader
            # yields as `[]` are blank lines and carry nothing; a row of bare delimiters
            # is `["", "", ""]`, which is three missing coordinates and survives to fail.
            numbered_rows = [(reader.line_num, row) for row in reader if row]

        if not numbered_rows:
            raise Flow360FileError(f"{file_path}: file has no rows.")

        # With named columns the first row is the header, which fixes both the coordinate
        # positions and the width every later row must match. Without them the file is
        # headerless and every row is exactly the three coordinates.
        first_row = numbered_rows[0][1]
        column_indices = (
            _coordinate_column_indices(file_path, first_row, coordinate_columns, delimiter)
            if coordinate_columns is not None
            else (0, 1, 2)
        )
        expected_column_count = len(first_row) if coordinate_columns is not None else 3
        data_rows = numbered_rows[1:] if coordinate_columns is not None else numbered_rows

        if not data_rows:
            raise Flow360FileError(f"{file_path}: file has a header row but no data rows.")

        coordinates: list[tuple[float, ...]] = []
        seen: set[tuple[float, ...]] = set()
        duplicate_lines: list[int] = []
        for line_number, row in data_rows:
            if len(row) != expected_column_count:
                raise Flow360FileError(
                    f"{file_path} line {line_number}: expected {expected_column_count} column(s), "
                    f"got {len(row)} (delimiter={delimiter!r}). If the file uses a different "
                    f"separator, pass `delimiter`."
                )
            try:
                coordinate = tuple(float(row[index]) for index in column_indices)
            except ValueError as error:
                raise Flow360FileError(f"{file_path} line {line_number}: {error}.") from error

            if coordinate in seen:
                duplicate_lines.append(line_number)
                continue
            seen.add(coordinate)
            coordinates.append(coordinate)

        if duplicate_lines:
            listed = ", ".join(str(line) for line in duplicate_lines[:10])
            elided = "" if len(duplicate_lines) <= 10 else ", ..."
            warnings.warn(
                f"`from_csv` dropped {len(duplicate_lines)} duplicate row(s) of {file_path} "
                f"at line(s) {listed}{elided}; {len(coordinates)} point(s) kept.",
                UserWarning,
                stacklevel=2,
            )

        # `private_attribute_entity_type_name` has a default; mypy cannot see defaults
        # declared through `pd.Field` without the pydantic plugin.
        return cls(name=name, locations=u.unyt_array(coordinates, unit))  # type: ignore[call-arg]

    @property
    def _monitor_key_names(self) -> list[str]:
        """The `monitorLocations` keys this entity writes. See :attr:`Point._monitor_key_names`."""
        return [f"{self.name}_{index}" for index in range(len(self.locations))]

    def _apply_transformation(self, matrix: NDArray) -> PointGroup:
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        import numpy as np

        locations_array = np.asarray(self.locations.value)  # type: ignore[attr-defined]
        new_locations_array = _transform_points(locations_array, matrix)
        new_locations = type(self.locations)(new_locations_array, self.locations.units)  # type: ignore[attr-defined, misc]
        return self.model_copy(update={"locations": new_locations})


class PointArray(TransformableEntity, EntityBase):
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

    @property
    def _monitor_key_names(self) -> list[str]:
        """The `monitorLocations` keys this entity writes. See :attr:`Point._monitor_key_names`."""
        return [f"{self.name}_{index}" for index in range(self.number_of_points)]

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


class PointArray2D(TransformableEntity, EntityBase):
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
