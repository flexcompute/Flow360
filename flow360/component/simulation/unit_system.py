"""
Unit system definitions and utilities
"""

# pylint: disable=too-many-lines, duplicate-code
from __future__ import annotations

from abc import ABCMeta
from enum import Enum
from numbers import Number
from operator import add, sub
from threading import Lock
from typing import Annotated, Any, Collection, List, Literal, Union

import annotated_types
import numpy as np
import pydantic as pd
import unyt as u
import unyt.dimensions as udim
from pydantic import PlainSerializer
from pydantic_core import InitErrorDetails, core_schema
from sympy import Symbol

# because unit_system.py is the only interface to our unit functions, you can import unit_quantity directly
# "from unit_system import unyt_quantity" instead of knowing existence of unyt package.
from unyt import unyt_quantity  # pylint: disable=unused-import

from flow360.log import log
from flow360.utils import classproperty

udim.viscosity = udim.pressure * udim.time
udim.kinematic_viscosity = udim.length * udim.length / udim.time
udim.angular_velocity = udim.angle / udim.time
udim.heat_flux = udim.mass / udim.time**3
udim.moment = udim.force * udim.length
udim.heat_source = udim.mass / udim.time**3 / udim.length
udim.specific_heat_capacity = udim.length**2 / udim.temperature / udim.time**2
udim.thermal_conductivity = udim.mass / udim.time**3 * udim.length / udim.temperature
udim.inverse_area = 1 / udim.area
udim.inverse_length = 1 / udim.length
udim.mass_flow_rate = udim.mass / udim.time
udim.specific_energy = udim.length**2 * udim.time ** (-2)
udim.frequency = udim.time ** (-1)
udim.delta_temperature = Symbol("(delta temperature)", positive=True)

# u.Unit("delta_degF") is parsed by unyt as 'ΔdegF and cannot find the unit. Had to use expr instead.
u.unit_systems.imperial_unit_system["temperature"] = u.Unit("degF").expr
u.unit_systems.imperial_unit_system["delta_temperature"] = u.Unit("delta_degF").expr
u.unit_systems.mks_unit_system["delta_temperature"] = u.Unit("K").expr
u.unit_systems.cgs_unit_system["delta_temperature"] = u.Unit("K").expr


class UnitSystemManager:
    """
    :class: Class to manage global unit system context and switch currently used unit systems
    """

    def __init__(self):
        """
        Initialize the UnitSystemManager.
        """
        self._current = None

    @property
    def current(self) -> UnitSystem:
        """
        Get the current UnitSystem.
        :return: UnitSystem
        """

        return self._current

    def set_current(self, unit_system: UnitSystem):
        """
        Set the current UnitSystem.
        :param unit_system:
        :return:
        """
        self._current = unit_system


unit_system_manager = UnitSystemManager()


def _encode_ndarray(x):
    """
    encoder for ndarray
    """
    if x.shape == ():
        return float(x)
    return tuple(x.tolist())


def _dimensioned_type_serializer(x):
    """
    encoder for dimensioned type (unyt_quantity, unyt_array, DimensionedType)
    """
    # adding .expr helps to avoid degF/C becoming serialized as °F/C
    return {"value": _encode_ndarray(x.value), "units": str(x.units.expr)}


def _check_if_input_is_nested_collection(value, nest_level):
    def get_nesting_level(value):
        if isinstance(value, np.ndarray):
            return value.ndim
        if isinstance(value, _Flow360BaseUnit):
            return value.value.ndim
        if isinstance(value, (list, tuple)):
            return 1 + max(get_nesting_level(item) for item in value)
        return 0

    return get_nesting_level(value) == nest_level


def _check_if_input_has_delta_unit(quant):
    """
    Parse the input unit and see if it can be considered a delta unit. This only handles temperatures now.
    """
    is_input_delta_unit = (
        str(quant.units) == str(u.Unit("delta_degC"))  # delta unit
        or str(quant.units) == str(u.Unit("delta_degF"))  # delta unit
        or str(quant.units) == "K"  # absolute temperature so it can be considered delta
        or str(quant.units) == "R"  # absolute temperature so it can be considered delta
        # Flow360 temperature scaled by absolute temperature, making it also absolute temperature
        or str(quant.units) == "flow360_delta_temperature_unit"
        or str(quant.units) == "flow360_temperature_unit"
    )
    return is_input_delta_unit


# pylint: disable=no-member
def _has_dimensions(quant, dim, expect_delta_unit: bool):
    """
    Checks the argument has the right dimensionality.
    """

    try:
        # Delta unit check only needed for temperature
        # Note: direct unit comparison won't work. Unyt consider u.Unit("degC") == u.Unit("K") as True

        is_input_delta_unit = _check_if_input_has_delta_unit(quant=quant)
        arg_dim = quant.units.dimensions

    except AttributeError:
        arg_dim = u.dimensionless
    return arg_dim == dim and (is_input_delta_unit if expect_delta_unit else True)


def _unit_object_parser(value, unyt_types: List[type]):
    """
    Parses {'value': value, 'units': units}, into unyt_type object : unyt.unyt_quantity, unyt.unyt_array
    """
    if isinstance(value, dict) is False or "units" not in value:
        return value
    if "value" not in value:
        raise TypeError(
            f"Dimensioned type instance {value} expects a 'value' field which was not given"
        )
    for unyt_type in unyt_types:
        try:
            return unyt_type(value["value"], value["units"], dtype=np.float64)
        except u.exceptions.UnitParseError:
            pass
        except RuntimeError:
            pass
        except KeyError:
            pass
    return value


def _is_unit_validator(value):
    """
    Parses str (eg: "m", "cm"), into unyt.Unit object
    """
    if isinstance(value, str):
        try:
            value = u.Unit(value)
        except u.exceptions.UnitParseError as err:
            raise TypeError(str(err)) from err
    return value


# pylint: disable=too-many-return-statements
def _unit_inference_validator(value, dim_name, is_array=False, is_matrix=False):
    """
    Uses current unit system to infer units for value

    Parameters
    ----------
    value :
        value to infer units for
    expected_base_type : type
        Expected base type to infer units for, eg Number
    dim_name : str
        dimension name, eg, "length"

    Returns
    -------
    unyt_quantity or value
    """

    if unit_system_manager.current:
        unit = unit_system_manager.current[dim_name]
        if is_matrix:
            if all(all(isinstance(item, Number) for item in row) for row in value):
                float64_tuple = tuple(tuple(np.float64(row)) for row in value)
                if isinstance(unit, _Flow360BaseUnit):
                    return float64_tuple * unit
                return float64_tuple * unit.units
        if is_array:
            if all(isinstance(item, Number) for item in value):
                float64_tuple = tuple(np.float64(item) for item in value)
                if isinstance(unit, _Flow360BaseUnit):
                    return float64_tuple * unit
                return float64_tuple * unit.units
        if isinstance(value, Number):
            if isinstance(unit, _Flow360BaseUnit):
                return np.float64(value) * unit
            return np.float64(value) * unit.units
    return value


def _unit_array_validator(value, dim, expect_delta_unit: bool):
    """
    Checks if units are provided for one component instead of entire object

    Parameters
    ----------
    value :
        value to check units for
    dim : unyt.dimensions
        dimension name, eg, unyt.dimensions.length

    Returns
    -------
    unyt_quantity or value
    """

    if not _has_dimensions(value, dim, expect_delta_unit):
        if any(_has_dimensions(item, dim, expect_delta_unit) for item in np.nditer(value)):
            raise TypeError(
                f"arg '{value}' has unit provided per component, "
                "instead provide dimension for entire array."
            )
    return value


def _has_dimensions_validator(value, dim, expect_delta_unit: bool):
    """
    Checks if value has expected dimension and raises TypeError
    """
    if not _has_dimensions(value, dim, expect_delta_unit):
        if expect_delta_unit:
            raise TypeError(f"arg '{value}' does not match unit representing difference in {dim}.")
        raise TypeError(f"arg '{value}' does not match {dim} dimension.")
    return value


def _nan_inf_vector_validator(value):
    if not isinstance(value, np.ndarray):
        return value
    if np.ndim(value.value) > 0 and (any(np.isnan(value.value)) or any(np.isinf(value.value))):
        raise ValueError("NaN/Inf/None found in input array. Please ensure your input is complete.")
    return value


def _enforce_float64(unyt_obj):
    """
    This make sure all the values are float64 to minimize floating point errors
    """
    if isinstance(unyt_obj, (u.Unit, _Flow360BaseUnit)):
        return unyt_obj

    # Determine if the object is a scalar or an array and cast to float64
    if isinstance(unyt_obj, u.unyt_array):
        # For unyt_array, ensure all elements are float64
        new_values = np.asarray(unyt_obj, dtype=np.float64)
        return new_values * unyt_obj.units

    if isinstance(unyt_obj, u.unyt_quantity):
        # For unyt_quantity, ensure the value is float64
        new_value = np.float64(unyt_obj)
        return u.unyt_quantity(new_value, unyt_obj.units)

    raise TypeError(f"arg '{unyt_obj}' is not a valid unyt object")


class _DimensionedType(metaclass=ABCMeta):
    """
    :class: Base class for dimensioned values
    """

    dim = None
    dim_name = None
    has_defaults = True

    # For temperature, the conversion is different if it is a delta or absolute
    expect_delta_unit = False

    @classmethod
    # pylint: disable=unused-argument
    def validate(cls, value, *args, **kwargs):
        """
        Validator for value
        """

        try:
            value = _unit_object_parser(value, [u.unyt_quantity, _Flow360BaseUnit.factory])
            value = _is_unit_validator(value)
            if cls.has_defaults:
                value = _unit_inference_validator(value, cls.dim_name)
            value = _has_dimensions_validator(value, cls.dim, cls.expect_delta_unit)
            value = _enforce_float64(value)
        except TypeError as err:
            details = InitErrorDetails(type="value_error", ctx={"error": str(err)})
            raise pd.ValidationError.from_exception_data("validation error", [details])

        if isinstance(value, u.Unit):
            return np.float64(1.0) * value

        return value

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs) -> pd.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_json_schema__(cls, schema: pd.CoreSchema, handler: pd.GetJsonSchemaHandler):
        schema = {"properties": {"value": {"type": "number"}, "units": {"type": "string"}}}

        if cls.dim_name is not None:
            schema["properties"]["units"]["dimension"] = cls.dim_name

            # Local import to prevent exposing mappings to the user
            # pylint: disable=import-outside-toplevel
            from flow360.component.simulation.exposed_units import (
                extra_units,
                ordered_complete_units,
            )

            if cls.dim_name in ordered_complete_units:
                units = [str(unit) for unit in ordered_complete_units[cls.dim_name]]
            else:
                units = [
                    str(_SI_system[cls.dim_name]),
                    str(_CGS_system[cls.dim_name]),
                    str(_imperial_system[cls.dim_name]),
                ]
                units += [str(unit) for unit in extra_units[cls.dim_name]]
                units = list(dict.fromkeys(units))
            schema["properties"]["units"]["enum"] = units

            schema = handler.resolve_ref_schema(schema)

        return schema

    # pylint: disable=too-few-public-methods
    class _Constrained:
        """
        :class: _Constrained
        Note that these constrains work only for values, disregards units.
        We cannot constrain that mass > 2kg, we can only constrain that mass.value > 2
        """

        @classmethod
        def get_class_object(cls, dim_type, **kwargs):
            """Get a dynamically created metaclass representing the constraint"""

            class _ConType(pd.BaseModel):
                kwargs.pop("allow_inf_nan", None)
                value: Annotated[
                    float,
                    annotated_types.Interval(
                        **{k: v for k, v in kwargs.items() if k != "allow_inf_nan"}
                    ),
                ]

            def validate(con_cls, value, *args, **kwargs):
                """Additional validator for value"""
                try:
                    dimensioned_value = dim_type.validate(value, **kwargs)

                    # Workaround to run annotated validation for numeric value of field
                    _ = _ConType(value=dimensioned_value.value)

                    return dimensioned_value
                except TypeError as err:
                    details = InitErrorDetails(type="value_error", ctx={"error": err})
                    raise pd.ValidationError.from_exception_data("validation error", [details])

            def __get_pydantic_json_schema__(
                con_cls, schema: pd.CoreSchema, handler: pd.GetJsonSchemaHandler
            ):
                schema = dim_type.__get_pydantic_json_schema__(schema, handler)
                constraints = con_cls.con_type.model_fields["value"].metadata[0]
                if constraints.ge is not None:
                    schema["properties"]["value"]["minimum"] = constraints.ge
                if constraints.le is not None:
                    schema["properties"]["value"]["maximum"] = constraints.le
                if constraints.gt is not None:
                    schema["properties"]["value"]["exclusiveMinimum"] = constraints.gt
                if constraints.lt is not None:
                    schema["properties"]["value"]["exclusiveMaximum"] = constraints.lt

                return schema

            def __get_pydantic_core_schema__(con_cls, *args, **kwargs) -> pd.CoreSchema:
                return core_schema.no_info_plain_validator_function(
                    lambda *val_args: validate(con_cls, *val_args)
                )

            cls_obj = type("_Constrained", (), {})
            cls_obj.con_type = _ConType
            cls_obj.__get_pydantic_core_schema__ = lambda *args: __get_pydantic_core_schema__(
                cls_obj, *args
            )
            cls_obj.__get_pydantic_json_schema__ = (
                lambda schema, handler: __get_pydantic_json_schema__(cls_obj, schema, handler)
            )
            return Annotated[cls_obj, pd.PlainSerializer(_dimensioned_type_serializer)]

    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments
    @classmethod
    def Constrained(cls, gt=None, ge=None, lt=None, le=None, allow_inf_nan=False):
        """
        Utility method to generate a dimensioned type with constraints based on the pydantic confloat
        """
        return cls._Constrained.get_class_object(
            cls, gt=gt, ge=ge, lt=lt, le=le, allow_inf_nan=allow_inf_nan
        )

    # pylint: disable=invalid-name
    @classproperty
    def NonNegative(self):
        """
        Shorthand for a ge=0 constrained value
        """
        return self._Constrained.get_class_object(self, ge=0)

    # pylint: disable=invalid-name
    @classproperty
    def Positive(self):
        """
        Shorthand for a gt=0 constrained value
        """
        return self._Constrained.get_class_object(self, gt=0)

    # pylint: disable=invalid-name
    @classproperty
    def NonPositive(self):
        """
        Shorthand for a le=0 constrained value
        """
        return self._Constrained.get_class_object(self, le=0)

    # pylint: disable=invalid-name
    @classproperty
    def Negative(self):
        """
        Shorthand for a lt=0 constrained value
        """
        return self._Constrained.get_class_object(self, lt=0)

    # pylint: disable=too-few-public-methods
    class _VectorType:
        @classmethod
        def get_class_object(
            cls,
            dim_type,
            allow_zero_component=True,
            allow_zero_norm=True,
            allow_negative_value=True,
            length=3,
        ):
            """Get a dynamically created metaclass representing the vector"""

            def __get_pydantic_json_schema__(
                schema: pd.CoreSchema, handler: pd.GetJsonSchemaHandler
            ):
                schema = dim_type.__get_pydantic_json_schema__(schema, handler)
                schema["properties"]["value"]["type"] = "array"
                schema["properties"]["value"]["items"] = {"type": "number"}
                if length is not None:
                    schema["properties"]["value"]["minItems"] = length
                    schema["properties"]["value"]["maxItems"] = length
                if length == 3:
                    schema["properties"]["value"]["strictType"] = {"type": "vector3"}

                return schema

            def validate(vec_cls, value, *args, **kwargs):
                """additional validator for value"""
                try:
                    value = _unit_object_parser(value, [u.unyt_array, _Flow360BaseUnit.factory])
                    value = _is_unit_validator(value)

                    is_collection = _check_if_input_is_nested_collection(value=value, nest_level=1)

                    if length is None:
                        if not is_collection:
                            raise TypeError(
                                f"arg '{value}' needs to be a collection of values of any length"
                            )
                    else:
                        if not is_collection or len(value) != length:
                            raise TypeError(
                                f"arg '{value}' needs to be a collection of {length} values"
                            )
                    if not vec_cls.allow_zero_component and any(item == 0 for item in value):
                        raise ValueError(f"arg '{value}' cannot have zero component")
                    if not vec_cls.allow_zero_norm and all(item == 0 for item in value):
                        raise ValueError(f"arg '{value}' cannot have zero norm")
                    if not vec_cls.allow_negative_value and any(item < 0 for item in value):
                        raise ValueError(f"arg '{value}' cannot have negative value")

                    if vec_cls.type.has_defaults:
                        value = _unit_inference_validator(
                            value, vec_cls.type.dim_name, is_array=True
                        )
                    value = _unit_array_validator(
                        value, vec_cls.type.dim, vec_cls.type.expect_delta_unit
                    )

                    if kwargs.get("allow_inf_nan", False) is False:
                        value = _nan_inf_vector_validator(value)

                    value = _has_dimensions_validator(
                        value,
                        vec_cls.type.dim,
                        vec_cls.type.expect_delta_unit,
                    )

                    return value
                except TypeError as err:
                    details = InitErrorDetails(type="value_error", ctx={"error": err})
                    raise pd.ValidationError.from_exception_data("validation error", [details])

            def __get_pydantic_core_schema__(vec_cls, *args, **kwargs) -> pd.CoreSchema:
                return core_schema.no_info_plain_validator_function(
                    lambda *val_args: validate(vec_cls, *val_args)
                )

            cls_obj = type("_VectorType", (), {})
            cls_obj.type = dim_type
            cls_obj.allow_zero_norm = allow_zero_norm
            cls_obj.allow_zero_component = allow_zero_component
            cls_obj.allow_negative_value = allow_negative_value
            cls_obj.__get_pydantic_core_schema__ = lambda *args: __get_pydantic_core_schema__(
                cls_obj, *args
            )
            cls_obj.__get_pydantic_json_schema__ = __get_pydantic_json_schema__

            return Annotated[cls_obj, pd.PlainSerializer(_dimensioned_type_serializer)]

    # pylint: disable=too-few-public-methods
    class _MatrixType:
        @classmethod
        def get_class_object(
            cls,
            dim_type,
            shape=(None, None),
        ):
            """Get a dynamically created metaclass representing the tensor"""

            def __get_pydantic_json_schema__(
                schema: pd.CoreSchema, handler: pd.GetJsonSchemaHandler
            ):
                schema = dim_type.__get_pydantic_json_schema__(schema, handler)
                schema["properties"]["value"] = {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                }
                if shape[0] is not None:
                    schema["properties"]["minItems"] = shape[0]
                    schema["properties"]["maxItems"] = shape[0]
                if shape[1] is not None:
                    schema["properties"]["value"]["items"]["minItems"] = shape[1]
                    schema["properties"]["value"]["items"]["maxItems"] = shape[1]

                return schema

            def validate(matrix_cls, value, *args, **kwargs):
                """additional validator for value"""
                try:
                    value = _unit_object_parser(value, [u.unyt_array, _Flow360BaseUnit.factory])
                    value = _is_unit_validator(value)

                    is_nested_collection = _check_if_input_is_nested_collection(
                        value=value, nest_level=2
                    )
                    if not is_nested_collection:
                        raise TypeError(
                            f"arg '{value}' needs to be a 2-dimensional collection of values."
                        )

                    if shape[0] and len(value) != shape[0]:
                        raise TypeError(
                            f"arg '{value}' needs to be a 2-dimensional collection of values "
                            + f"with the 1st dimension as {shape[0]}."
                        )

                    if shape[1] and any(
                        len(item) != shape[1]
                        for item in (
                            value if not isinstance(value, _Flow360BaseUnit) else value.val
                        )
                    ):
                        raise TypeError(
                            f"arg '{value}' needs to be a 2-dimensional collection of values "
                            + f"with the 2nd dimension as {shape[1]}."
                        )

                    if matrix_cls.type.has_defaults:
                        value = _unit_inference_validator(
                            value, matrix_cls.type.dim_name, is_matrix=True
                        )
                    value = _unit_array_validator(
                        value, matrix_cls.type.dim, matrix_cls.type.expect_delta_unit
                    )

                    value = _has_dimensions_validator(
                        value,
                        matrix_cls.type.dim,
                        matrix_cls.type.expect_delta_unit,
                    )

                    return value
                except TypeError as err:
                    details = InitErrorDetails(type="value_error", ctx={"error": err})
                    raise pd.ValidationError.from_exception_data("validation error", [details])

            def __get_pydantic_core_schema__(matrix_cls, *args, **kwargs) -> pd.CoreSchema:
                return core_schema.no_info_plain_validator_function(
                    lambda *val_args: validate(matrix_cls, *val_args)
                )

            cls_obj = type("_MatrixType", (), {})
            cls_obj.type = dim_type
            cls_obj.__get_pydantic_core_schema__ = lambda *args: __get_pydantic_core_schema__(
                cls_obj, *args
            )
            cls_obj.__get_pydantic_json_schema__ = __get_pydantic_json_schema__

            return Annotated[cls_obj, pd.PlainSerializer(_dimensioned_type_serializer)]

    # pylint: disable=invalid-name
    @classproperty
    def Array(self):
        """
        Array value which accepts any length
        """
        return self._VectorType.get_class_object(self, length=None)

    # pylint: disable=invalid-name
    @classproperty
    def NonNegativeArray(self):
        """
        Array value which accepts nonnegative with any length
        """
        return self._VectorType.get_class_object(self, length=None, allow_negative_value=False)

    # pylint: disable=invalid-name
    @classproperty
    def Point(self):
        """
        Vector value which accepts zero components
        """
        return self._VectorType.get_class_object(self)

    # pylint: disable=invalid-name
    @classproperty
    def Vector(self):
        """
        Vector value which accepts zero components
        """
        return self._VectorType.get_class_object(self)

    # pylint: disable=invalid-name
    @classproperty
    def PositiveVector(self):
        """
        Vector value which only accepts positive components
        """
        return self._VectorType.get_class_object(
            self, allow_zero_component=False, allow_negative_value=False
        )

    # pylint: disable=invalid-name
    @classproperty
    def Direction(self):
        """
        Vector value which does not accept zero components
        """
        return self._VectorType.get_class_object(self, allow_zero_norm=False)

    # pylint: disable=invalid-name
    @classproperty
    def Axis(self):
        """
        Vector value which does not accept zero components
        """
        return self._VectorType.get_class_object(self, allow_zero_norm=False)

    # pylint: disable=invalid-name
    @classproperty
    def Moment(self):
        """
        Vector value which does not accept zero values in coordinates
        """
        return self._VectorType.get_class_object(
            self, allow_zero_norm=False, allow_zero_component=False
        )

    @classproperty
    def CoordinateGroupTranspose(self):
        """
        CoordinateGroup value which stores a group of 3D coordinates
        """
        return self._MatrixType.get_class_object(self, shape=(3, None))

    @classproperty
    def CoordinateGroup(self):
        """
        CoordinateGroup value which stores a group of 3D coordinates
        """
        return self._MatrixType.get_class_object(self, shape=(None, 3))


# pylint: disable=too-few-public-methods
class _LengthType(_DimensionedType):
    """:class: LengthType"""

    dim = udim.length
    dim_name = "length"


LengthType = Annotated[_LengthType, PlainSerializer(_dimensioned_type_serializer)]


# pylint: disable=too-few-public-methods
class _AngleType(_DimensionedType):
    """:class: AngleType"""

    dim = udim.angle
    dim_name = "angle"
    has_defaults = False


AngleType = Annotated[_AngleType, PlainSerializer(_dimensioned_type_serializer)]


# pylint: disable=too-few-public-methods
class _MassType(_DimensionedType):
    """:class: MassType"""

    dim = udim.mass
    dim_name = "mass"


MassType = Annotated[_MassType, PlainSerializer(_dimensioned_type_serializer)]


# pylint: disable=too-few-public-methods
class _TimeType(_DimensionedType):
    """:class: TimeType"""

    dim = udim.time
    dim_name = "time"


TimeType = Annotated[_TimeType, PlainSerializer(_dimensioned_type_serializer)]


class _AbsoluteTemperatureType(_DimensionedType):
    """
    :class: AbsoluteTemperatureType.
    This is the class for absolute temperature which is differentiated
    from DeltaTemperatureType where the change/offset of temperatures are handled.
    """

    dim = udim.temperature
    dim_name = "temperature"


def _check_temperature_is_physical(value):
    if str(value.units).startswith("flow360_") or value is None:
        # Scaled. No need to check
        return value
    if value.in_units("K").value < 0:
        raise ValueError(
            f"The specified temperature {value} is below absolute zero. Please input a physical temperature value."
        )
    return value


AbsoluteTemperatureType = Annotated[
    _AbsoluteTemperatureType,
    PlainSerializer(_dimensioned_type_serializer),
    pd.AfterValidator(_check_temperature_is_physical),
]


class _DeltaTemperatureType(_DimensionedType):
    """
    :class: DeltaTemperatureType.
    This is the class for absolute temperature which is differentiated
    from DeltaTemperatureType where the change/offset of temperatures are handled.
    """

    dim = udim.temperature
    dim_name = "delta_temperature"
    expect_delta_unit = True


DeltaTemperatureType = Annotated[
    _DeltaTemperatureType, PlainSerializer(_dimensioned_type_serializer)
]


class _VelocityType(_DimensionedType):
    """:class: VelocityType"""

    dim = udim.velocity
    dim_name = "velocity"


VelocityType = Annotated[_VelocityType, PlainSerializer(_dimensioned_type_serializer)]


class _AreaType(_DimensionedType):
    """:class: AreaType"""

    dim = udim.area
    dim_name = "area"


AreaType = Annotated[_AreaType, PlainSerializer(_dimensioned_type_serializer)]


class _ForceType(_DimensionedType):
    """:class: ForceType"""

    dim = udim.force
    dim_name = "force"


ForceType = Annotated[_ForceType, PlainSerializer(_dimensioned_type_serializer)]


class _PressureType(_DimensionedType):
    """:class: PressureType"""

    dim = udim.pressure
    dim_name = "pressure"


PressureType = Annotated[_PressureType, PlainSerializer(_dimensioned_type_serializer)]


class _DensityType(_DimensionedType):
    """:class: DensityType"""

    dim = udim.density
    dim_name = "density"


DensityType = Annotated[_DensityType, PlainSerializer(_dimensioned_type_serializer)]


class _ViscosityType(_DimensionedType):
    """:class: ViscosityType"""

    dim = udim.viscosity
    dim_name = "viscosity"


ViscosityType = Annotated[_ViscosityType, PlainSerializer(_dimensioned_type_serializer)]


class _KinematicViscosityType(_DimensionedType):
    """:class: KinematicViscosityType"""

    dim = udim.kinematic_viscosity
    dim_name = "kinematic_viscosity"


KinematicViscosityType = Annotated[
    _KinematicViscosityType, PlainSerializer(_dimensioned_type_serializer)
]


class _PowerType(_DimensionedType):
    """:class: PowerType"""

    dim = udim.power
    dim_name = "power"


PowerType = Annotated[_PowerType, PlainSerializer(_dimensioned_type_serializer)]


class _MomentType(_DimensionedType):
    """:class: MomentType"""

    dim = udim.moment
    dim_name = "moment"


MomentType = Annotated[_MomentType, PlainSerializer(_dimensioned_type_serializer)]


class _AngularVelocityType(_DimensionedType):
    """:class: AngularVelocityType"""

    dim = udim.angular_velocity
    dim_name = "angular_velocity"
    has_defaults = False


AngularVelocityType = Annotated[_AngularVelocityType, PlainSerializer(_dimensioned_type_serializer)]


class _HeatFluxType(_DimensionedType):
    """:class: HeatFluxType"""

    dim = udim.heat_flux
    dim_name = "heat_flux"


HeatFluxType = Annotated[_HeatFluxType, PlainSerializer(_dimensioned_type_serializer)]


class _HeatSourceType(_DimensionedType):
    """:class: HeatSourceType"""

    dim = udim.heat_source
    dim_name = "heat_source"


HeatSourceType = Annotated[_HeatSourceType, PlainSerializer(_dimensioned_type_serializer)]


class _SpecificHeatCapacityType(_DimensionedType):
    """:class: SpecificHeatCapacityType"""

    dim = udim.specific_heat_capacity
    dim_name = "specific_heat_capacity"


SpecificHeatCapacityType = Annotated[
    _SpecificHeatCapacityType, PlainSerializer(_dimensioned_type_serializer)
]


class _ThermalConductivityType(_DimensionedType):
    """:class: ThermalConductivityType"""

    dim = udim.thermal_conductivity
    dim_name = "thermal_conductivity"


ThermalConductivityType = Annotated[
    _ThermalConductivityType, PlainSerializer(_dimensioned_type_serializer)
]


class _InverseAreaType(_DimensionedType):
    """:class: InverseAreaType"""

    dim = udim.inverse_area
    dim_name = "inverse_area"


InverseAreaType = Annotated[_InverseAreaType, PlainSerializer(_dimensioned_type_serializer)]


class _InverseLengthType(_DimensionedType):
    """:class: InverseLengthType"""

    dim = udim.inverse_length
    dim_name = "inverse_length"


InverseLengthType = Annotated[_InverseLengthType, PlainSerializer(_dimensioned_type_serializer)]


class _MassFlowRateType(_DimensionedType):
    """:class: MassFlowRateType"""

    dim = udim.mass_flow_rate
    dim_name = "mass_flow_rate"


MassFlowRateType = Annotated[_MassFlowRateType, PlainSerializer(_dimensioned_type_serializer)]


class _SpecificEnergyType(_DimensionedType):
    """:class: SpecificEnergyType"""

    dim = udim.specific_energy
    dim_name = "specific_energy"


SpecificEnergyType = Annotated[_SpecificEnergyType, PlainSerializer(_dimensioned_type_serializer)]


class _FrequencyType(_DimensionedType):
    """:class: FrequencyType"""

    dim = udim.frequency
    dim_name = "frequency"


FrequencyType = Annotated[_FrequencyType, PlainSerializer(_dimensioned_type_serializer)]


DimensionedTypes = Union[
    LengthType,
    AngleType,
    MassType,
    TimeType,
    AbsoluteTemperatureType,
    VelocityType,
    AreaType,
    ForceType,
    PressureType,
    DensityType,
    ViscosityType,
    KinematicViscosityType,
    PowerType,
    MomentType,
    AngularVelocityType,
    HeatFluxType,
    HeatSourceType,
    SpecificHeatCapacityType,
    ThermalConductivityType,
    InverseAreaType,
    InverseLengthType,
    MassFlowRateType,
    SpecificEnergyType,
    FrequencyType,
]


def _iterable(obj):
    try:
        len(obj)
    except TypeError:
        return False
    return True


class _Flow360BaseUnit(_DimensionedType):
    dimension_type = None
    unit_name = None

    @classproperty
    def units(self):
        """
        Retrieve units of a flow360 unit system value
        """
        parent_self = self

        # pylint: disable=too-few-public-methods
        # pylint: disable=invalid-name
        class _Units:
            dimensions = self.dimension_type.dim

            def __str__(self):
                return f"{parent_self.unit_name}"

            def expr(self):
                """alias for __str__ so the serializer can work"""
                return str(self)

        return _Units()

    @property
    def value(self):
        """
        Retrieve value of a flow360 unit system value, use np.ndarray to keep interface consistent with unyt
        """
        return np.asarray(self.val)

    # pylint: disable=invalid-name
    @property
    def v(self):
        "alias for value"
        return self.value

    def __init__(self, val=None) -> None:
        self.val = val

    @classmethod
    def factory(cls, value, unit_name, dtype=np.float64):
        """Returns specialized class object based on unit name

        Parameters
        ----------
        value : Numeric or Collection
            Base value
        unit_name : str
            Unit name, e.g. flow360_length_unit

        Returns
        -------
        Specialized _Flow360BaseUnit
            Returns specialized _Flow360BaseUnit such as unit_name equals provided unit_name

        Raises
        ------
        ValueError
            If specialized class was not found based on provided unit_name
        """
        for sub_classes in _Flow360BaseUnit.__subclasses__():
            if sub_classes.unit_name == unit_name:
                return sub_classes(dtype(value))
        raise ValueError(f"No class found for unit_name: {unit_name}")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.val == other.val
        return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.val != other.val
        return True

    def __lt__(self, other):
        """
        This seems consistent with unyt in that for numbers only value is compared
        e.g.:
        >>> 1*u.mm >0.1
        array(True)
        """
        if isinstance(other, self.__class__):
            return self.val < other.val
        if isinstance(other, Number):
            return self.val < other
        raise ValueError(
            f"Invalid other value type for comparison, expected Number or Flow360BaseUnit but got {type(other)}"
        )

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.val > other.val
        if isinstance(other, Number):
            return self.val > other
        raise ValueError(
            f"Invalid other value type for comparison, expected Number or Flow360BaseUnit but got {type(other)}"
        )

    def __len__(self):
        if self.val and isinstance(self.val, Collection):
            return len(self.val)
        return 1

    @property
    def size(self):
        """implements numpy size interface"""
        return len(self)

    def _unit_iter(self, iter_obj):
        if not _iterable(iter_obj):
            yield self.__class__(iter_obj)
        else:
            for value in iter(iter_obj):
                yield self.__class__(value)

    def __iter__(self):
        try:
            return self._unit_iter(self.val)
        except TypeError as exc:
            raise TypeError(f"{self} is not iterable") from exc

    def __repr__(self):
        if self.val:
            return f"({self.val}, {self.units})"
        return f"({self.units})"

    def __str__(self):
        if self.val:
            return f"{self.val} {self.units}"
        return f"{self.units}"

    def _can_do_math_operations(self, other):
        if self.val is None:
            raise ValueError(
                "Cannot perform math operations on units only. Multiply unit by numerical value first."
            )
        if not isinstance(other, self.__class__):
            raise TypeError(f"Operation not defined on {self} and {other}")

    def __rsub__(self, other):
        self._can_do_math_operations(other)
        return self.__class__(other.val - self.val)

    def __sub__(self, other):
        self._can_do_math_operations(other)
        if isinstance(self.val, Collection):
            return self.__class__(list(map(sub, self.val, other.val)))
        return self.__class__(self.val - other.val)

    def __radd__(self, other):
        self._can_do_math_operations(other)
        return self.__add__(other)

    def __add__(self, other):
        self._can_do_math_operations(other)
        if isinstance(self.val, Collection):
            return self.__class__(list(map(add, self.val, other.val)))
        return self.__class__(self.val + other.val)

    def __rmul__(self, unit):
        return self.__mul__(unit)

    def __mul__(self, other):
        if isinstance(other, Number):
            if self.val:
                return self.__class__(self.val * other)
            return self.__class__(other)
        if isinstance(other, Collection) and (not self.val or self.val == 1):
            return self.__class__(other)
        raise TypeError(f"Operation not defined on {self} and {other}")

    def in_base(self, base, flow360_conv_system):
        """
        Convert unit to a specific base system
        """
        value = self.value * flow360_conv_system[self.dimension_type.dim_name]
        value.units.registry = flow360_conv_system.registry
        converted = value.in_base(unit_system=base)
        return converted


class Flow360LengthUnit(_Flow360BaseUnit):
    """:class: Flow360LengthUnit"""

    dimension_type = LengthType
    unit_name = "flow360_length_unit"


class Flow360AngleUnit(_Flow360BaseUnit):
    """:class: Flow360AngleUnit"""

    dimension_type = AngleType
    unit_name = "flow360_angle_unit"


class Flow360MassUnit(_Flow360BaseUnit):
    """:class: Flow360MassUnit"""

    dimension_type = MassType
    unit_name = "flow360_mass_unit"


class Flow360TimeUnit(_Flow360BaseUnit):
    """:class: Flow360TimeUnit"""

    dimension_type = TimeType
    unit_name = "flow360_time_unit"


class Flow360TemperatureUnit(_Flow360BaseUnit):
    """
    :class: Flow360TemperatureUnit.
    This is absolute temperature because temperature is scaled with Kelvin temperature.
    """

    dimension_type = AbsoluteTemperatureType
    unit_name = "flow360_temperature_unit"


class Flow360DeltaTemperatureUnit(_Flow360BaseUnit):
    """
    :class: Flow360DeltaTemperatureUnit.
    """

    dimension_type = DeltaTemperatureType
    unit_name = "flow360_delta_temperature_unit"


class Flow360VelocityUnit(_Flow360BaseUnit):
    """:class: Flow360VelocityUnit"""

    dimension_type = VelocityType
    unit_name = "flow360_velocity_unit"


class Flow360AreaUnit(_Flow360BaseUnit):
    """:class: Flow360AreaUnit"""

    dimension_type = AreaType
    unit_name = "flow360_area_unit"


class Flow360ForceUnit(_Flow360BaseUnit):
    """:class: Flow360ForceUnit"""

    dimension_type = ForceType
    unit_name = "flow360_force_unit"


class Flow360PressureUnit(_Flow360BaseUnit):
    """:class: Flow360PressureUnit"""

    dimension_type = PressureType
    unit_name = "flow360_pressure_unit"


class Flow360DensityUnit(_Flow360BaseUnit):
    """:class: Flow360DensityUnit"""

    dimension_type = DensityType
    unit_name = "flow360_density_unit"


class Flow360ViscosityUnit(_Flow360BaseUnit):
    """:class: Flow360ViscosityUnit"""

    dimension_type = ViscosityType
    unit_name = "flow360_viscosity_unit"


class Flow360KinematicViscosityUnit(_Flow360BaseUnit):
    """:class: Flow360KinematicViscosityUnit"""

    dimension_type = KinematicViscosityType
    unit_name = "flow360_kinematic_viscosity_unit"


class Flow360PowerUnit(_Flow360BaseUnit):
    """:class: Flow360PowerUnit"""

    dimension_type = PowerType
    unit_name = "flow360_power_unit"


class Flow360MomentUnit(_Flow360BaseUnit):
    """:class: Flow360MomentUnit"""

    dimension_type = MomentType
    unit_name = "flow360_moment_unit"


class Flow360AngularVelocityUnit(_Flow360BaseUnit):
    """:class: Flow360AngularVelocityUnit"""

    dimension_type = AngularVelocityType
    unit_name = "flow360_angular_velocity_unit"


class Flow360HeatFluxUnit(_Flow360BaseUnit):
    """:class: Flow360HeatFluxUnit"""

    dimension_type = HeatFluxType
    unit_name = "flow360_heat_flux_unit"


class Flow360HeatSourceUnit(_Flow360BaseUnit):
    """:class: Flow360HeatSourceUnit"""

    dimension_type = HeatSourceType
    unit_name = "flow360_heat_source_unit"


class Flow360SpecificHeatCapacityUnit(_Flow360BaseUnit):
    """:class: Flow360SpecificHeatCapacityUnit"""

    dimension_type = SpecificHeatCapacityType
    unit_name = "flow360_specific_heat_capacity_unit"


class Flow360ThermalConductivityUnit(_Flow360BaseUnit):
    """:class: Flow360ThermalConductivityUnit"""

    dimension_type = ThermalConductivityType
    unit_name = "flow360_thermal_conductivity_unit"


class Flow360InverseAreaUnit(_Flow360BaseUnit):
    """:class: Flow360InverseAreaUnit"""

    dimension_type = InverseAreaType
    unit_name = "flow360_inverse_area_unit"


class Flow360InverseLengthUnit(_Flow360BaseUnit):
    """:class: Flow360InverseLengthUnit"""

    dimension_type = InverseLengthType
    unit_name = "flow360_inverse_length_unit"


class Flow360MassFlowRateUnit(_Flow360BaseUnit):
    """:class: Flow360MassFlowRateUnit"""

    dimension_type = MassFlowRateType
    unit_name = "flow360_mass_flow_rate_unit"


class Flow360SpecificEnergyUnit(_Flow360BaseUnit):
    """:class: Flow360SpecificEnergyUnit"""

    dimension_type = SpecificEnergyType
    unit_name = "flow360_specific_energy_unit"


class Flow360FrequencyUnit(_Flow360BaseUnit):
    """:class: Flow360FrequencyUnit"""

    dimension_type = FrequencyType
    unit_name = "flow360_frequency_unit"


def is_flow360_unit(value):
    """
    Check if the provided value represents a dimensioned quantity with units
    that start with 'flow360'.

    Parameters:
    - value: The value to be checked for units.

    Returns:
    - bool: True if the value has units starting with 'flow360', False otherwise.

    Raises:
    - ValueError: If the provided value does not have the 'units' attribute.
    """

    if hasattr(value, "units"):
        return str(value.units).startswith("flow360")
    raise ValueError(f"Expected a dimensioned value, but {value} provided.")


_lock = Lock()


# pylint: disable=too-few-public-methods
class BaseSystemType(Enum):
    """
    :class: Type of the base unit system to use for unit inference (all units need to be specified if not provided)
    """

    SI = "SI"
    CGS = "CGS"
    IMPERIAL = "Imperial"
    FLOW360 = "Flow360"
    NONE = None


_dim_names = [
    "mass",
    "length",
    "angle",
    "time",
    "temperature",
    "velocity",
    "area",
    "force",
    "pressure",
    "density",
    "viscosity",
    "kinematic_viscosity",
    "power",
    "moment",
    "angular_velocity",
    "heat_flux",
    "heat_source",
    "specific_heat_capacity",
    "thermal_conductivity",
    "inverse_area",
    "inverse_length",
    "mass_flow_rate",
    "specific_energy",
    "frequency",
    "delta_temperature",
]


class UnitSystem(pd.BaseModel):
    """
    :class: Customizable unit system containing definitions for most atomic and complex dimensions.
    """

    mass: MassType = pd.Field()
    length: LengthType = pd.Field()
    angle: AngleType = pd.Field()
    time: TimeType = pd.Field()
    temperature: AbsoluteTemperatureType = pd.Field()
    velocity: VelocityType = pd.Field()
    area: AreaType = pd.Field()
    force: ForceType = pd.Field()
    pressure: PressureType = pd.Field()
    density: DensityType = pd.Field()
    viscosity: ViscosityType = pd.Field()
    kinematic_viscosity: KinematicViscosityType = pd.Field()
    power: PowerType = pd.Field()
    moment: MomentType = pd.Field()
    angular_velocity: AngularVelocityType = pd.Field()
    heat_flux: HeatFluxType = pd.Field()
    heat_source: HeatSourceType = pd.Field()
    specific_heat_capacity: SpecificHeatCapacityType = pd.Field()
    thermal_conductivity: ThermalConductivityType = pd.Field()
    inverse_area: InverseAreaType = pd.Field()
    inverse_length: InverseLengthType = pd.Field()
    mass_flow_rate: MassFlowRateType = pd.Field()
    specific_energy: SpecificEnergyType = pd.Field()
    frequency: FrequencyType = pd.Field()
    delta_temperature: DeltaTemperatureType = pd.Field()

    name: Literal["Custom"] = pd.Field("Custom")

    _verbose: bool = pd.PrivateAttr(True)

    @staticmethod
    def __get_unit(system, dim_name, unit):
        if unit is not None:
            return unit
        if system is not None:
            if system == BaseSystemType.SI:
                return _SI_system[dim_name]
            if system == BaseSystemType.CGS:
                return _CGS_system[dim_name]
            if system == BaseSystemType.IMPERIAL:
                return _imperial_system[dim_name]
            if system == BaseSystemType.FLOW360:
                return _flow360_system[dim_name]
        return None

    def __init__(self, verbose: bool = True, **kwargs):
        base_system = kwargs.get("base_system")
        base_system = BaseSystemType(base_system)
        units = {}

        for dim in _dim_names:
            unit = kwargs.get(dim)
            units[dim] = UnitSystem.__get_unit(base_system, dim, unit)

        missing = set(_dim_names) - set(units.keys())

        super().__init__(**units, base_system=base_system)

        if len(missing) > 0:
            raise ValueError(
                f"Tried defining incomplete unit system, missing definitions for {','.join(missing)}"
            )

        self._verbose = verbose

    def __eq__(self, other):
        equal = [getattr(self, name) == getattr(other, name) for name in _dim_names]
        return all(equal)

    @classmethod
    def from_dict(cls, **kwargs):
        """Construct a unit system from the provided dictionary"""

        class _TemporaryModel(pd.BaseModel):
            unit_system: UnitSystemType = pd.Field(discriminator="name")

        params = {"unit_system": kwargs}
        model = _TemporaryModel(**params)

        return model.unit_system

    def defaults(self):
        """
        Get the default units for each dimension in the unit system.

        Returns
        -------
        dict
            A dictionary containing the default units for each dimension. The keys are dimension names, and the values
            are strings representing the default unit expressions.

        Example
        -------
        >>> unit_system = UnitSystem(base_system=BaseSystemType.SI, length=u.m, mass=u.kg, time=u.s)
        >>> unit_system.defaults()
        {'mass': 'kg', 'length': 'm', 'time': 's', 'temperature': 'K', 'velocity': 'm/s',
        'area': 'm**2', 'force': 'N', 'pressure': 'Pa', 'density': 'kg/m**3',
        'viscosity': 'Pa*s', kinematic_viscosity': 'm**2/s', 'power': 'W', 'angular_velocity': 'rad/s',
        'heat_flux': 'kg/s**3', 'specific_heat_capacity': 'm**2/(s**2*K)', 'thermal_conductivity': 'kg*m/(s**3*K)',
        'inverse_area': '1/m**2', 'inverse_length': '1/m', 'heat_source': 'kg/(m*s**3)'}
        """

        defaults = {}
        for item in self._dim_names:
            defaults[item] = str(self[item].units)
        return defaults

    def __getitem__(self, item):
        """to support [] access"""
        return getattr(self, item)

    def system_repr(self):
        """(mass, length, time, temperature) string representation of the system"""
        units = [
            str(unit.units if unit.v == 1.0 else unit)
            for unit in [self.mass, self.length, self.time, self.temperature]
        ]
        str_repr = f"({', '.join(units)})"

        return str_repr

    def __enter__(self):
        _lock.acquire()
        if self._verbose:
            log.info(f"using: {self.system_repr()} unit system for unit inference.")
        unit_system_manager.set_current(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _lock.release()
        unit_system_manager.set_current(None)


_SI_system = u.unit_systems.mks_unit_system
_CGS_system = u.unit_systems.cgs_unit_system
_imperial_system = u.unit_systems.imperial_unit_system

flow360_length_unit = Flow360LengthUnit()
flow360_angle_unit = Flow360AngleUnit()
flow360_mass_unit = Flow360MassUnit()
flow360_time_unit = Flow360TimeUnit()
flow360_temperature_unit = Flow360TemperatureUnit()
flow360_velocity_unit = Flow360VelocityUnit()
flow360_area_unit = Flow360AreaUnit()
flow360_force_unit = Flow360ForceUnit()
flow360_pressure_unit = Flow360PressureUnit()
flow360_density_unit = Flow360DensityUnit()
flow360_viscosity_unit = Flow360ViscosityUnit()
flow360_kinematic_viscosity_unit = Flow360KinematicViscosityUnit()
flow360_power_unit = Flow360PowerUnit()
flow360_moment_unit = Flow360MomentUnit()
flow360_angular_velocity_unit = Flow360AngularVelocityUnit()
flow360_heat_flux_unit = Flow360HeatFluxUnit()
flow360_heat_source_unit = Flow360HeatSourceUnit()
flow360_specific_heat_capacity_unit = Flow360SpecificHeatCapacityUnit()
flow360_thermal_conductivity_unit = Flow360ThermalConductivityUnit()
flow360_inverse_area_unit = Flow360InverseAreaUnit()
flow360_inverse_length_unit = Flow360InverseLengthUnit()
flow360_mass_flow_rate_unit = Flow360MassFlowRateUnit()
flow360_specific_energy_unit = Flow360SpecificEnergyUnit()
flow360_delta_temperature_unit = Flow360DeltaTemperatureUnit()
flow360_frequency_unit = Flow360FrequencyUnit()

dimensions = [
    flow360_length_unit,
    flow360_angle_unit,
    flow360_mass_unit,
    flow360_time_unit,
    flow360_temperature_unit,
    flow360_velocity_unit,
    flow360_area_unit,
    flow360_force_unit,
    flow360_pressure_unit,
    flow360_density_unit,
    flow360_viscosity_unit,
    flow360_kinematic_viscosity_unit,
    flow360_power_unit,
    flow360_moment_unit,
    flow360_angular_velocity_unit,
    flow360_heat_flux_unit,
    flow360_specific_heat_capacity_unit,
    flow360_thermal_conductivity_unit,
    flow360_inverse_area_unit,
    flow360_inverse_length_unit,
    flow360_mass_flow_rate_unit,
    flow360_specific_energy_unit,
    flow360_delta_temperature_unit,
    flow360_frequency_unit,
    flow360_heat_source_unit,
]

_flow360_system = {u.dimension_type.dim_name: u for u in dimensions}


# pylint: disable=too-many-instance-attributes
class Flow360ConversionUnitSystem(pd.BaseModel):
    """
    Flow360ConversionUnitSystem class for setting conversion rates for converting from dimensioned values into flow360
    values
    """

    base_length: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360LengthUnit})
    base_angle: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360AngleUnit})
    base_mass: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360MassUnit})
    base_time: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360TimeUnit})
    base_temperature: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360TemperatureUnit}
    )
    base_velocity: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360VelocityUnit}
    )
    base_area: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360AreaUnit})
    base_force: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360ForceUnit})
    base_density: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360DensityUnit}
    )
    base_pressure: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360PressureUnit}
    )
    base_viscosity: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360ViscosityUnit}
    )
    base_kinematic_viscosity: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360KinematicViscosityUnit}
    )
    base_power: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360PowerUnit})
    base_moment: float = pd.Field(np.inf, json_schema_extra={"target_dimension": Flow360MomentUnit})
    base_angular_velocity: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360AngularVelocityUnit}
    )
    base_heat_flux: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360HeatFluxUnit}
    )
    base_heat_source: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360HeatSourceUnit}
    )
    base_specific_heat_capacity: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360SpecificHeatCapacityUnit}
    )
    base_thermal_conductivity: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360ThermalConductivityUnit}
    )
    base_inverse_area: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360InverseAreaUnit}
    )
    base_inverse_length: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360InverseLengthUnit}
    )
    base_mass_flow_rate: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360MassFlowRateUnit}
    )
    base_specific_energy: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360SpecificEnergyUnit}
    )
    base_frequency: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360FrequencyUnit}
    )
    base_delta_temperature: float = pd.Field(
        np.inf, json_schema_extra={"target_dimension": Flow360DeltaTemperatureUnit}
    )

    registry: Any = pd.Field(frozen=False)
    conversion_system: Any = pd.Field(frozen=False)

    model_config = pd.ConfigDict(extra="forbid", validate_assignment=True, frozen=False)

    def __init__(self):
        registry = u.UnitRegistry()

        for field in self.model_fields.values():
            if field.json_schema_extra is not None:
                target_dimension = field.json_schema_extra.get("target_dimension", None)
                if target_dimension is not None:
                    registry.add(
                        target_dimension.unit_name,
                        field.default,
                        target_dimension.dimension_type.dim,
                    )

        conversion_system = u.UnitSystem(
            "flow360_v2",
            "flow360_length_unit",
            "flow360_mass_unit",
            "flow360_time_unit",
            "flow360_temperature_unit",
            "flow360_angle_unit",
            registry=registry,
        )

        conversion_system["velocity"] = "flow360_velocity_unit"
        conversion_system["area"] = "flow360_area_unit"
        conversion_system["force"] = "flow360_force_unit"
        conversion_system["density"] = "flow360_density_unit"
        conversion_system["pressure"] = "flow360_pressure_unit"
        conversion_system["viscosity"] = "flow360_viscosity_unit"
        conversion_system["kinematic_viscosity"] = "flow360_kinematic_viscosity_unit"
        conversion_system["power"] = "flow360_power_unit"
        conversion_system["moment"] = "flow360_moment_unit"
        conversion_system["angular_velocity"] = "flow360_angular_velocity_unit"
        conversion_system["heat_flux"] = "flow360_heat_flux_unit"
        conversion_system["heat_source"] = "flow360_heat_source_unit"
        conversion_system["specific_heat_capacity"] = "flow360_specific_heat_capacity_unit"
        conversion_system["thermal_conductivity"] = "flow360_thermal_conductivity_unit"
        conversion_system["inverse_area"] = "flow360_inverse_area_unit"
        conversion_system["inverse_length"] = "flow360_inverse_length_unit"
        conversion_system["mass_flow_rate"] = "flow360_mass_flow_rate_unit"
        conversion_system["specific_energy"] = "flow360_specific_energy_unit"
        conversion_system["delta_temperature"] = "flow360_delta_temperature_unit"
        conversion_system["frequency"] = "flow360_frequency_unit"
        conversion_system["angle"] = "flow360_angle_unit"

        super().__init__(registry=registry, conversion_system=conversion_system)

    # pylint: disable=no-self-argument
    @pd.field_validator("*")
    def assign_conversion_rate(cls, value, info: pd.ValidationInfo):
        """
        Pydantic validator for assigning conversion rates to a specific unit in the registry.
        """
        field = cls.model_fields.get(info.field_name)
        if field.json_schema_extra is not None:
            target_dimension = field.json_schema_extra.get("target_dimension", None)
            if target_dimension is not None:
                registry = info.data["registry"]
                registry.modify(target_dimension.unit_name, value)

        return value


flow360_conversion_unit_system = Flow360ConversionUnitSystem()


class _PredefinedUnitSystem(UnitSystem):
    mass: MassType = pd.Field(exclude=True)
    length: LengthType = pd.Field(exclude=True)
    angle: AngleType = pd.Field(exclude=True)
    time: TimeType = pd.Field(exclude=True)
    temperature: AbsoluteTemperatureType = pd.Field(exclude=True)
    velocity: VelocityType = pd.Field(exclude=True)
    area: AreaType = pd.Field(exclude=True)
    force: ForceType = pd.Field(exclude=True)
    pressure: PressureType = pd.Field(exclude=True)
    density: DensityType = pd.Field(exclude=True)
    viscosity: ViscosityType = pd.Field(exclude=True)
    kinematic_viscosity: KinematicViscosityType = pd.Field(exclude=True)
    power: PowerType = pd.Field(exclude=True)
    moment: MomentType = pd.Field(exclude=True)
    angular_velocity: AngularVelocityType = pd.Field(exclude=True)
    heat_flux: HeatFluxType = pd.Field(exclude=True)
    heat_source: HeatSourceType = pd.Field(exclude=True)
    specific_heat_capacity: SpecificHeatCapacityType = pd.Field(exclude=True)
    thermal_conductivity: ThermalConductivityType = pd.Field(exclude=True)
    inverse_area: InverseAreaType = pd.Field(exclude=True)
    inverse_length: InverseLengthType = pd.Field(exclude=True)
    mass_flow_rate: MassFlowRateType = pd.Field(exclude=True)
    specific_energy: SpecificEnergyType = pd.Field(exclude=True)
    delta_temperature: DeltaTemperatureType = pd.Field(exclude=True)
    frequency: FrequencyType = pd.Field(exclude=True)

    # pylint: disable=missing-function-docstring
    def system_repr(self):
        return self.name


class SIUnitSystem(_PredefinedUnitSystem):
    """:class: `SIUnitSystem` predefined SI system wrapper"""

    name: Literal["SI"] = pd.Field("SI", frozen=True)

    def __init__(self, verbose: bool = True, **kwargs):
        super().__init__(base_system=BaseSystemType.SI, verbose=verbose, **kwargs)

    # pylint: disable=missing-function-docstring
    @classmethod
    def validate(cls, _):
        return SIUnitSystem()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


class CGSUnitSystem(_PredefinedUnitSystem):
    """:class: `CGSUnitSystem` predefined CGS system wrapper"""

    name: Literal["CGS"] = pd.Field("CGS", frozen=True)

    def __init__(self, **kwargs):
        super().__init__(base_system=BaseSystemType.CGS, **kwargs)

    # pylint: disable=missing-function-docstring
    @classmethod
    def validate(cls, _):
        return CGSUnitSystem()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


class ImperialUnitSystem(_PredefinedUnitSystem):
    """:class: `ImperialUnitSystem` predefined imperial system wrapper"""

    name: Literal["Imperial"] = pd.Field("Imperial", frozen=True)

    def __init__(self, **kwargs):
        super().__init__(base_system=BaseSystemType.IMPERIAL, **kwargs)

    # pylint: disable=missing-function-docstring
    @classmethod
    def validate(cls, _):
        return ImperialUnitSystem()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


class Flow360UnitSystem(_PredefinedUnitSystem):
    """:class: `Flow360UnitSystem` predefined flow360 system wrapper"""

    name: Literal["Flow360"] = pd.Field("Flow360", frozen=True)

    def __init__(self, verbose: bool = True):
        super().__init__(base_system=BaseSystemType.FLOW360, verbose=verbose)

    # pylint: disable=missing-function-docstring
    @classmethod
    def validate(cls, _):
        return Flow360UnitSystem()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate


UnitSystemType = Union[
    SIUnitSystem, CGSUnitSystem, ImperialUnitSystem, Flow360UnitSystem, UnitSystem
]

SI_unit_system = SIUnitSystem()
CGS_unit_system = CGSUnitSystem()
imperial_unit_system = ImperialUnitSystem()
flow360_unit_system = Flow360UnitSystem()
