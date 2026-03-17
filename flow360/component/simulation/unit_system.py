"""
Unit system definitions and utilities
"""

# pylint: disable=too-many-lines, duplicate-code
from __future__ import annotations

from abc import ABCMeta
from numbers import Number
from typing import Annotated, List, Union

import annotated_types
import numpy as np
import pydantic as pd
import unyt as u
import unyt.dimensions as udim

# Importing unit_system triggers udim.* dimension registrations and
# unit_systems configuration on the schema side. Must happen before
# _DimensionedType subclass bodies reference udim.viscosity etc.
from flow360_schema.framework.unit_system import (  # pylint: disable=unused-import
    _UNIT_SYSTEMS,
    CGS_unit_system,
    CGSUnitSystem,
    ImperialUnitSystem,
    SI_unit_system,
    SIUnitSystem,
    UnitSystem,
    UnitSystemConfig,
    create_flow360_unit_system,
    imperial_unit_system,
)
from flow360_schema.framework.unit_system.base_system_type import (  # pylint: disable=unused-import
    BaseSystemType,
)

# pylint: disable=wrong-import-order
from flow360_schema.framework.validation.context import (  # pylint: disable=unused-import
    unit_system_manager,
)
from pydantic import PlainSerializer
from pydantic_core import InitErrorDetails, core_schema

# because unit_system.py is the only interface to our unit functions, you can import unit_quantity directly
# "from unit_system import unyt_quantity" instead of knowing existence of unyt package.
from unyt import unyt_quantity  # pylint: disable=unused-import

# pylint: enable=wrong-import-order
from flow360.utils import classproperty


def _encode_ndarray(x):
    """
    encoder for ndarray

    For scalar values (ndim==0), convert to float.
    For arrays (ndim>0), preserve as tuple/list even if size==1,
    since Array types should remain as collections.
    """
    if x.ndim == 0:
        return float(x)
    # This is an array (e.g., LengthType.Array), preserve as collection
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
        if isinstance(value, (list, tuple)):
            return 1 + max(get_nesting_level(item) for item in value)
        return 0

    return get_nesting_level(value) == nest_level


def _check_if_input_has_delta_unit(quant):
    """
    Parse the input unit and see if it can be considered a delta unit. This only handles temperatures now.
    """
    unit_str = str(quant.units)
    is_input_delta_unit = (
        unit_str == str(u.Unit("delta_degC"))  # delta unit
        or unit_str == str(u.Unit("delta_degF"))  # delta unit
        or unit_str == "K"  # absolute temperature so it can be considered delta
        or unit_str == "R"  # absolute temperature so it can be considered delta
        # Flow360 temperature scaled by absolute temperature, making it also absolute temperature
        or unit_str == "flow360_delta_temperature_unit"
        or unit_str == "flow360_temperature_unit"
        # Check for units like "356.483333333333*K" (scaled K) which can occur during unit conversion
        or (unit_str.endswith("*K") or unit_str.endswith("*R"))
    )
    return is_input_delta_unit


# pylint: disable=no-member
def _has_dimensions(quant, dim, expect_delta_unit: bool):
    """
    Checks the argument has the right dimensions.
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


def _list_of_unyt_quantity_to_unyt_array(value):
    """
    Convert list of unyt_quantity (may come from `Expression`) to unyt_array
    Only handles situation where all components share exact same unit.
    We cab relax this to cover more expression results in the future when we decide how to convert.
    """

    if not isinstance(value, list):
        return value
    if not all(isinstance(item, unyt_quantity) for item in value):
        return value
    units = {item.units for item in value}
    if not len(units) == 1:
        return value
    shared_unit = units.pop()
    return [item.value for item in value] * shared_unit


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
                return float64_tuple * unit.units
        if is_array:
            if all(isinstance(item, Number) for item in value):
                float64_tuple = tuple(np.float64(item) for item in value)
                return float64_tuple * unit.units
        if isinstance(value, Number):
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
    if isinstance(unyt_obj, u.Unit):
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
            value = _unit_object_parser(value, [u.unyt_quantity])
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
                    str(u.unit_systems.mks_unit_system[cls.dim_name]),
                    str(u.unit_systems.cgs_unit_system[cls.dim_name]),
                    str(u.unit_systems.imperial_unit_system[cls.dim_name]),
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
            allow_decreasing=True,
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

            def validate(vec_cls, value, info, *args, **kwargs):
                """additional validator for value"""
                try:
                    value = _unit_object_parser(value, [u.unyt_array])
                    value = _list_of_unyt_quantity_to_unyt_array(value)
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
                    if not vec_cls.allow_decreasing and any(
                        x >= y for x, y in zip(value, value[1:])
                    ):
                        raise ValueError(f"arg '{value}' is not strictly increasing")

                    if vec_cls.type.has_defaults:
                        value = _unit_inference_validator(
                            value, vec_cls.type.dim_name, is_array=True
                        )
                    value = _unit_array_validator(
                        value, vec_cls.type.dim, vec_cls.type.expect_delta_unit
                    )

                    allow_inf_nan = kwargs.get("allow_inf_nan", False)

                    if info.context and "allow_inf_nan" in info.context:
                        allow_inf_nan = info.context.get("allow_inf_nan", False)

                    if allow_inf_nan is False:
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
                def validate_with_info(value, info):
                    return validate(vec_cls, value, info, *args, **kwargs)

                return core_schema.with_info_plain_validator_function(validate_with_info)

            cls_obj = type("_VectorType", (), {})
            cls_obj.type = dim_type
            cls_obj.allow_zero_norm = allow_zero_norm
            cls_obj.allow_zero_component = allow_zero_component
            cls_obj.allow_negative_value = allow_negative_value
            cls_obj.allow_decreasing = allow_decreasing
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
                    value = _unit_object_parser(value, [u.unyt_array])
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

                    if shape[1] and any(len(item) != shape[1] for item in value):
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
    def PositiveArray(self):
        """
        Array value which accepts positive with any length
        """
        return self._VectorType.get_class_object(
            self, length=None, allow_negative_value=False, allow_zero_component=False
        )

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

    @classproperty
    def Pair(self):
        """
        Array value which accepts length 2.
        """
        return self._VectorType.get_class_object(self, length=2)

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
    def Range(self):
        """
        Array value which accepts length 2 and is strictly increasing
        """
        return self._VectorType.get_class_object(self, allow_decreasing=False, length=2)

    @classproperty
    def PositiveRange(self):
        """
        Range which contains strictly positive values
        """
        return self._VectorType.get_class_object(
            self, allow_negative_value=False, allow_decreasing=False, length=2
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


class _AccelerationType(_DimensionedType):
    """:class: AccelerationType"""

    dim = udim.acceleration
    dim_name = "acceleration"


AccelerationType = Annotated[_AccelerationType, PlainSerializer(_dimensioned_type_serializer)]


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

UnitSystemType = UnitSystem
