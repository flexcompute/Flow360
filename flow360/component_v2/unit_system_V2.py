"""
Unit system definitions and utilities
"""

# pylint: disable=too-many-lines
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import Enum
from numbers import Number
from operator import add, sub
from threading import Lock
from typing import (
    Annotated,
    Any,
    Collection,
    List,
    Literal,
    Type,
    Union,
    get_type_hints,
)

import numpy as np
import pydantic as pd
import unyt as u
from pydantic_core import CoreSchema, core_schema
from unyt import unyt_quantity

from flow360.log import log
from flow360.utils import classproperty

u.dimensions.viscosity = u.dimensions.pressure * u.dimensions.time
u.dimensions.angular_velocity = u.dimensions.angle / u.dimensions.time
u.dimensions.heat_flux = u.dimensions.mass / u.dimensions.time**3
u.dimensions.moment = u.dimensions.force * u.dimensions.length
u.dimensions.heat_source = u.dimensions.mass / u.dimensions.time**3 / u.dimensions.length
u.dimensions.heat_capacity = (
    u.dimensions.mass / u.dimensions.time**2 / u.dimensions.length / u.dimensions.temperature
)
u.dimensions.thermal_conductivity = (
    u.dimensions.mass / u.dimensions.time**3 * u.dimensions.length / u.dimensions.temperature
)
u.dimensions.inverse_area = 1 / u.dimensions.area
u.dimensions.inverse_length = 1 / u.dimensions.length

# pylint: disable=no-member
u.unit_systems.mks_unit_system["viscosity"] = u.Pa * u.s
# pylint: disable=no-member
u.unit_systems.mks_unit_system["angular_velocity"] = u.rad / u.s
# pylint: disable=no-member
u.unit_systems.mks_unit_system["heat_flux"] = u.kg / u.s**3
# pylint: disable=no-member
u.unit_systems.mks_unit_system["moment"] = u.N * u.m
# pylint: disable=no-member
u.unit_systems.mks_unit_system["heat_source"] = u.kg / u.s**3 / u.m
# pylint: disable=no-member
u.unit_systems.mks_unit_system["heat_capacity"] = u.kg / u.s**2 / u.m / u.K
# pylint: disable=no-member
u.unit_systems.mks_unit_system["thermal_conductivity"] = u.kg / u.s**3 * u.m / u.K
# pylint: disable=no-member
u.unit_systems.mks_unit_system["inverse_area"] = u.m ** (-2)
# pylint: disable=no-member
u.unit_systems.mks_unit_system["inverse_length"] = u.m ** (-1)

# pylint: disable=no-member
u.unit_systems.cgs_unit_system["viscosity"] = u.dyn * u.s / u.cm**2
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["angular_velocity"] = u.rad / u.s
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["heat_flux"] = u.g / u.s**3
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["moment"] = u.dyn * u.m
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["heat_source"] = u.g / u.s**3 / u.cm
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["heat_capacity"] = u.g / u.s**2 / u.cm / u.K
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["thermal_conductivity"] = u.g / u.s**3 * u.cm / u.K
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["inverse_area"] = u.cm ** (-2)
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["inverse_length"] = u.cm ** (-1)

# pylint: disable=no-member
u.unit_systems.imperial_unit_system["viscosity"] = u.lbf * u.s / u.ft**2
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["angular_velocity"] = u.rad / u.s
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["heat_flux"] = u.lb / u.s**3
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["moment"] = u.lbf * u.ft
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["heat_source"] = u.lb / u.s**3 / u.ft
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["heat_capacity"] = u.lb / u.s**2 / u.ft / u.K
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["thermal_conductivity"] = u.lb / u.s**3 * u.ft / u.K
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["inverse_area"] = u.ft ** (-2)
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["inverse_length"] = u.ft ** (-1)


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

    def copy_current(self):
        """
        Get a copy of the current UnitSystem.
        :return: UnitSystem
        """
        if self._current:
            copy = self._current.model_copy(deep=True)
            return copy
        return None

    def set_current(self, unit_system: UnitSystem):
        """
        Set the current UnitSystem.
        :param unit_system:
        :return:
        """
        self._current = unit_system


unit_system_manager = UnitSystemManager()


# pylint: disable=no-member
def _has_dimensions(quant: unyt_quantity, dim):
    """
    Checks the argument has the right dimensionality.
    """

    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = u.dimensionless
    return arg_dim == dim


def _unit_object_parser(value, unyt_types: List[type]):  # unit_dict_parser?
    """
    Parses {'value': value, 'units': units}, into unyt_type object : unyt.unyt_quantity, unyt.unyt_array
    """
    if isinstance(value, dict) and "units" in value:
        if "value" in value:
            for unyt_type in unyt_types:
                try:
                    return unyt_type(value["value"], value["units"])
                except u.exceptions.UnitParseError:
                    pass
        else:
            raise TypeError(
                f"Dimensioned type instance {value} expects a 'value' field which was not given"
            )
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


def _unit_inference_validator(value, dim_name, is_array=False):
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
        if is_array:
            if all(isinstance(item, Number) for item in value):
                return value * unit
        if isinstance(value, Number):
            return value * unit
    return value


def _unit_array_validator(value, dim):
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

    if not _has_dimensions(value, dim):
        if any(_has_dimensions(item, dim) for item in value):
            raise TypeError(
                f"arg '{value}' has unit provided per component, "
                "instead provide dimension for entire array."
            )
    return value


def _has_dimensions_validator(value, dim):
    """
    Checks if value has expected dimension and raises TypeError
    """
    if not _has_dimensions(value, dim):
        raise TypeError(f"arg '{value}' does not match {dim}")
    return value


# pylint: disable=too-few-public-methods
class ValidatedType(metaclass=ABCMeta):
    """
    :class: Abstract class for dimensioned types with custom validation
    """

    @classmethod
    @abstractmethod
    def validate(cls, value):  # process?
        """validation and also process value into unyt_type object"""

    @classmethod
    @abstractmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: pd.GetJsonSchemaHandler
    ):
        """Change schema for front end (?)"""


class DimensionedType(ValidatedType):
    """
    :class: Base class for dimensioned values
    """

    dim = None
    dim_name = None

    @classmethod
    def validate(cls, value):
        """validation and also convert to unyt_type object"""

        # value is {'value': value, 'units': units}
        value = _unit_object_parser(value, [u.unyt_quantity, _Flow360BaseUnit.factory])
        # value is just "m" or "cm" etc.
        value = _is_unit_validator(value)
        # value is single scalar or scalar array
        value = _unit_inference_validator(value, cls.dim_name)
        # value is already a unyt_type object
        value = _has_dimensions_validator(value, cls.dim)

        if isinstance(value, u.Unit):
            return 1.0 * value

        return value

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: pd.GetJsonSchemaHandler
    ):
        field_schema = super().__get_pydantic_json_schema__(core_schema, handler)
        field_schema = handler.resolve_ref_schema(field_schema)

        field_schema["properties"] = {}
        field_schema["properties"]["value"] = {}
        field_schema["properties"]["units"] = {}
        field_schema["properties"]["value"]["type"] = "number"
        field_schema["properties"]["units"]["type"] = "string"
        if cls.dim_name is not None:
            field_schema["properties"]["units"]["dimension"] = cls.dim_name
            # Local import to prevent exposing mappings to the user
            # pylint: disable=import-outside-toplevel
            from flow360.component.flow360_params.exposed_units import extra_units

            units = [
                str(_SI_system[cls.dim_name]),
                str(_CGS_system[cls.dim_name]),
                str(_imperial_system[cls.dim_name]),
                str(_flow360_system[cls.dim_name]),
            ]
            units += [str(unit) for unit in extra_units[cls.dim_name]]
            units = list(dict.fromkeys(units))
            field_schema["properties"]["units"]["enum"] = units
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Type[Any], handler: pd.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        print(">>>", source.__base__)
        assert source.__base__ is DimensionedType
        return (
            core_schema.no_info_after_validator_function(
                cls.validate, core_schema.str_schema(), ref="Units"
            ),
        )

    class _Constrained:
        """
        :class: _Constrained
        Note that these constrains work only for values, disregards units.
        We cannot constrain that mass > 2kg, we can only constrain that mass.value > 2
        """

        @classmethod
        def get_class_object(cls, dim_type: ValidatedType, **kwargs):
            """Get a dynamically created metaclass representing the constraint"""

            class _ConType:
                type_ = Annotated[float, pd.Field(**kwargs)]

            def validate(con_cls, value):
                """Additional validator for value"""
                dimensioned_value = dim_type.validate(value)
                pd.validators.number_size_validator(dimensioned_value.value, con_cls.con_type)
                pd.validators.float_finite_validator(
                    dimensioned_value.value, con_cls.con_type, None
                )
                return dimensioned_value

            @classmethod
            def __get_pydantic_json_schema__(
                con_cls, core_schema: CoreSchema, handler: pd.GetJsonSchemaHandler
            ):
                field_schema = dim_type.__get_pydantic_json_schema__(core_schema, handler)
                field_schema = handler.resolve_ref_schema(field_schema)

                constraints = con_cls.con_type.type_
                if constraints.ge is not None:
                    field_schema["properties"]["value"]["minimum"] = constraints.ge
                if constraints.le is not None:
                    field_schema["properties"]["value"]["maximum"] = constraints.le
                if constraints.gt is not None:
                    field_schema["properties"]["value"]["exclusiveMinimum"] = constraints.gt
                if constraints.lt is not None:
                    field_schema["properties"]["value"]["exclusiveMaximum"] = constraints.lt

            cls_obj = type("_Constrained", (), {})
            cls_obj.con_type = _ConType
            cls_obj.validate = lambda value: validate(cls_obj, value)
            cls_obj.__get_pydantic_json_schema__ = (
                lambda core_schema, handler: __get_pydantic_json_schema__(
                    cls_obj, core_schema, handler
                )
            )

            return cls_obj

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
        return self._Constrained.get_class_object(self, ge=0, allow_inf_nan=False)

    # pylint: disable=invalid-name
    @classproperty
    def Positive(self):
        """
        Shorthand for a gt=0 constrained value
        """
        return self._Constrained.get_class_object(self, gt=0, allow_inf_nan=False)

    # pylint: disable=invalid-name
    @classproperty
    def NonPositive(self):
        """
        Shorthand for a le=0 constrained value
        """
        return self._Constrained.get_class_object(self, le=0, allow_inf_nan=False)

    # pylint: disable=invalid-name
    @classproperty
    def Negative(self):
        """
        Shorthand for a lt=0 constrained value
        """
        return self._Constrained.get_class_object(self, lt=0, allow_inf_nan=False)

    class _VectorType:
        @classmethod
        def get_class_object(
            cls, dim_type: ValidatedType, allow_zero_coord=True, allow_zero_norm=True, length=3
        ):
            """Get a dynamically created metaclass representing the vector"""

            @classmethod
            def __get_pydantic_json_schema__(
                cls, core_schema: CoreSchema, handler: pd.GetJsonSchemaHandler
            ):
                field_schema = dim_type.__get_pydantic_json_schema__(core_schema, handler)
                dim_type.__get_pydantic_json_schema__(field_schema, handler)

                field_schema = handler.resolve_ref_schema(field_schema)

                field_schema["properties"]["value"]["type"] = "array"
                field_schema["properties"]["value"]["items"] = {"type": "number"}
                if length is not None:
                    field_schema["properties"]["value"]["minItems"] = length
                    field_schema["properties"]["value"]["maxItems"] = length
                if length == 3:
                    field_schema["properties"]["value"]["strictType"] = {"type": "vector3"}

            def validate(vec_cls, value):
                """additional validator for value"""
                # value is {'value': value, 'units': units}
                value = _unit_object_parser(value, [u.unyt_array, _Flow360BaseUnit.factory])
                # value is just "m" or "cm" etc.
                value = _is_unit_validator(value)

                is_collection = isinstance(value, Collection) or (
                    isinstance(value, _Flow360BaseUnit) and isinstance(value.val, Collection)
                )

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
                if not vec_cls.allow_zero_coord and any(item == 0 for item in value):
                    raise ValueError(f"arg '{value}' cannot have zero coordinate values")
                if not vec_cls.allow_zero_norm and all(item == 0 for item in value):
                    raise ValueError(f"arg '{value}' cannot have zero norm")

                value = _unit_inference_validator(value, vec_cls.type.dim_name, is_array=True)
                value = _unit_array_validator(value, vec_cls.type.dim)
                value = _has_dimensions_validator(value, vec_cls.type.dim)

                return value

            cls_obj = type("_VectorType", (), {})
            cls_obj.type = dim_type
            cls_obj.allow_zero_norm = allow_zero_norm
            cls_obj.allow_zero_coord = allow_zero_coord
            cls_obj.validate = lambda value: validate(cls_obj, value)
            cls_obj.__get_pydantic_json_schema__ = __get_pydantic_json_schema__
            return cls_obj

    # pylint: disable=invalid-name
    @classproperty
    def Array(self):
        """
        Array value which accepts any length
        """
        return self._VectorType.get_class_object(self, length=None)

    # pylint: disable=invalid-name
    @classproperty
    def Point(self):
        """
        Vector value which accepts zero-vectors
        """
        return self._VectorType.get_class_object(self)

    # pylint: disable=invalid-name
    @classproperty
    def Vector(self):
        """
        Vector value which accepts zero-vectors
        """
        return self._VectorType.get_class_object(self)

    # pylint: disable=invalid-name
    @classproperty
    def Direction(self):
        """
        Vector value which does not accept zero-vectors
        """
        return self._VectorType.get_class_object(self, allow_zero_norm=False)

    # pylint: disable=invalid-name
    @classproperty
    def Axis(self):
        """
        Vector value which does not accept zero-vectors
        """
        return self._VectorType.get_class_object(self, allow_zero_norm=False)

    # pylint: disable=invalid-name
    @classproperty
    def Moment(self):
        """
        Vector value which does not accept zero values in coordinates
        """
        return self._VectorType.get_class_object(
            self, allow_zero_norm=False, allow_zero_coord=False
        )


class LengthType(DimensionedType):
    """:class: LengthType"""

    dim = u.dimensions.length
    dim_name = "length"


class MassType(DimensionedType):
    """:class: MassType"""

    dim = u.dimensions.mass
    dim_name = "mass"


class TimeType(DimensionedType):
    """:class: TimeType"""

    dim = u.dimensions.time
    dim_name = "time"


class TemperatureType(DimensionedType):
    """:class: TemperatureType"""

    dim = u.dimensions.temperature
    dim_name = "temperature"

    @classmethod
    def validate(cls, value):
        value = super(cls, cls).validate(value)

        if value is not None and isinstance(value, u.unyt_array) and value.to("K") <= 0:
            raise ValueError(
                f"Temperature cannot be lower or equal to absolute zero {value} == {value.to('K')}"
            )

        return value


class VelocityType(DimensionedType):
    """:class: VelocityType"""

    dim = u.dimensions.velocity
    dim_name = "velocity"


class AreaType(DimensionedType):
    """:class: AreaType"""

    dim = u.dimensions.area
    dim_name = "area"


class ForceType(DimensionedType):
    """:class: ForceType"""

    dim = u.dimensions.force
    dim_name = "force"


class PressureType(DimensionedType):
    """:class: PressureType"""

    dim = u.dimensions.pressure
    dim_name = "pressure"


class DensityType(DimensionedType):
    """:class: DensityType"""

    dim = u.dimensions.density
    dim_name = "density"


class ViscosityType(DimensionedType):
    """:class: ViscosityType"""

    dim = u.dimensions.viscosity
    dim_name = "viscosity"


class PowerType(DimensionedType):
    """:class: PowerType"""

    dim = u.dimensions.power
    dim_name = "power"


class MomentType(DimensionedType):
    """:class: MomentType"""

    dim = u.dimensions.moment
    dim_name = "moment"


class AngularVelocityType(DimensionedType):
    """:class: AngularVelocityType"""

    dim = u.dimensions.angular_velocity
    dim_name = "angular_velocity"


class HeatFluxType(DimensionedType):
    """:class: HeatFluxType"""

    dim = u.dimensions.heat_flux
    dim_name = "heat_flux"


class HeatSourceType(DimensionedType):
    """:class: HeatSourceType"""

    dim = u.dimensions.heat_source
    dim_name = "heat_source"


class HeatCapacityType(DimensionedType):
    """:class: HeatCapacityType"""

    dim = u.dimensions.heat_capacity
    dim_name = "heat_capacity"


class ThermalConductivityType(DimensionedType):
    """:class: ThermalConductivityType"""

    dim = u.dimensions.thermal_conductivity
    dim_name = "thermal_conductivity"


class InverseAreaType(DimensionedType):
    """:class: InverseAreaType"""

    dim = u.dimensions.inverse_area
    dim_name = "inverse_area"


class InverseLengthType(DimensionedType):
    """:class: InverseLengthType"""

    dim = u.dimensions.inverse_length
    dim_name = "inverse_length"


def _iterable(obj):
    try:
        len(obj)
    except TypeError:
        return False
    return True


class _Flow360BaseUnit(DimensionedType):
    dimension_type = None
    unit_name = None

    @classproperty
    def units(self):
        """
        Retrieve units of a flow360 unit system value
        """
        parent_self = self

        # pylint: disable=invalid-name
        class _units:
            dimensions = self.dimension_type.dim

            def __str__(self):
                return f"{parent_self.unit_name}"

        return _units()

    @property
    def value(self):
        """
        Retrieve value of a flow360 unit system value, use np.ndarray to keep interface consistant with unyt
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
    def factory(cls, value, unit_name):
        """Returns specialised class object based on unit name

        Parameters
        ----------
        value : Numeric or Collection
            Base value
        unit_name : str
            Unit name, e.g. flow360_length_unit

        Returns
        -------
        Specialised _Flow360BaseUnit
            Returns specialised _Flow360BaseUnit such as unit_name equals provided unit_name

        Raises
        ------
        ValueError
            If specialised class was not found based on provided unit_name
        """
        for sub_classes in _Flow360BaseUnit.__subclasses__():
            if sub_classes.unit_name == unit_name:
                return sub_classes(value)
        raise ValueError(f"No class found for unit_name: {unit_name}")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.val == other.val
        return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.val != other.val
        return True

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


class Flow360MassUnit(_Flow360BaseUnit):
    """:class: Flow360MassUnit"""

    dimension_type = MassType
    unit_name = "flow360_mass_unit"


class Flow360TimeUnit(_Flow360BaseUnit):
    """:class: Flow360TimeUnit"""

    dimension_type = TimeType
    unit_name = "flow360_time_unit"


class Flow360TemperatureUnit(_Flow360BaseUnit):
    """:class: Flow360TemperatureUnit"""

    dimension_type = TemperatureType
    unit_name = "flow360_temperature_unit"


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


class Flow360HeatCapacityUnit(_Flow360BaseUnit):
    """:class: Flow360HeatCapacityUnit"""

    dimension_type = HeatCapacityType
    unit_name = "flow360_heat_capacity_unit"


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


class BaseSystemType(Enum):
    """
    :class: Type of the base unit system to use for unit inference (all units need to be specified if not provided)
    """

    SI = "SI"
    CGS = "CGS"
    IMPERIAL = "Imperial"
    FLOW360 = "Flow360"
    NONE = None


class UnitSystem(pd.BaseModel):
    """
    :class: Customizable unit system containing definitions for most atomic and complex dimensions.
    """

    mass: MassType = pd.Field()
    length: LengthType = pd.Field()
    time: TimeType = pd.Field()
    temperature: TemperatureType = pd.Field()
    velocity: VelocityType = pd.Field()
    area: AreaType = pd.Field()
    force: ForceType = pd.Field()
    pressure: PressureType = pd.Field()
    density: DensityType = pd.Field()
    viscosity: ViscosityType = pd.Field()
    power: PowerType = pd.Field()
    moment: MomentType = pd.Field()
    angular_velocity: AngularVelocityType = pd.Field()
    heat_flux: HeatFluxType = pd.Field()
    heat_source: HeatSourceType = pd.Field()
    heat_capacity: HeatCapacityType = pd.Field()
    thermal_conductivity: ThermalConductivityType = pd.Field()
    inverse_area: InverseAreaType = pd.Field()
    inverse_length: InverseLengthType = pd.Field()

    name: Literal["Custom"] = pd.Field("Custom")

    _verbose: bool = pd.PrivateAttr(True)

    _dim_names = pd.PrivateAttr(
        [
            "mass",
            "length",
            "time",
            "temperature",
            "velocity",
            "area",
            "force",
            "pressure",
            "density",
            "viscosity",
            "power",
            "moment",
            "angular_velocity",
            "heat_flux",
            "heat_source",
            "heat_capacity",
            "thermal_conductivity",
            "inverse_area",
            "inverse_length",
        ]
    )

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

    def __post_init__(self, verbose: bool = True, **kwargs):
        base_system = kwargs.get("base_system")
        base_system = BaseSystemType(base_system)
        units = {}

        for dim in self._dim_names:
            unit = kwargs.get(dim)
            units[dim] = UnitSystem.__get_unit(base_system, dim, unit)

        missing = set(self._dim_names) - set(units.keys())

        super().__init__(**units, base_system=base_system)

        if len(missing) > 0:
            raise ValueError(
                f"Tried defining incomplete unit system, missing definitions for {','.join(missing)}"
            )

        self._verbose = verbose

    def __eq__(self, other):
        equal = [getattr(self, name) == getattr(other, name) for name in self._dim_names]
        return all(equal)

        # @classmethod
        # def from_dict(cls, **kwargs):
        #     """Construct a unit system from the provided dictionary"""

        #     class _TemporaryModel(pd.BaseModel):
        #         unit_system: UnitSystemType = pd.Field(discriminator="name")

        #     params = {"unit_system": kwargs}
        #     model = _TemporaryModel(**params)

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
        'viscosity': 'Pa*s', 'power': 'W', 'angular_velocity': 'rad/s', 'heat_flux': 'kg/s**3',
        'heat_capacity': 'kg/(s**2*m*K)', 'thermal_conductivity': 'kg*m/(s**3*K)',
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
            log.info(f"using: {self.system_repr()} unit system")
        unit_system_manager.set_current(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _lock.release()
        unit_system_manager.set_current(None)


_SI_system = u.unit_systems.mks_unit_system
_CGS_system = u.unit_systems.cgs_unit_system
_imperial_system = u.unit_systems.imperial_unit_system

flow360_length_unit = Flow360LengthUnit()
flow360_mass_unit = Flow360MassUnit()
flow360_time_unit = Flow360TimeUnit()
flow360_temperature_unit = Flow360TemperatureUnit()
flow360_velocity_unit = Flow360VelocityUnit()
flow360_area_unit = Flow360AreaUnit()
flow360_force_unit = Flow360ForceUnit()
flow360_pressure_unit = Flow360PressureUnit()
flow360_density_unit = Flow360DensityUnit()
flow360_viscosity_unit = Flow360ViscosityUnit()
flow360_power_unit = Flow360PowerUnit()
flow360_moment_unit = Flow360MomentUnit()
flow360_angular_velocity_unit = Flow360AngularVelocityUnit()
flow360_heat_flux_unit = Flow360HeatFluxUnit()
flow360_heat_source_unit = Flow360HeatSourceUnit()
flow360_heat_capacity_unit = Flow360HeatCapacityUnit()
flow360_thermal_conductivity_unit = Flow360ThermalConductivityUnit()
flow360_inverse_area_unit = Flow360InverseAreaUnit()
flow360_inverse_length_unit = Flow360InverseLengthUnit()

dimensions = [
    flow360_length_unit,
    flow360_mass_unit,
    flow360_time_unit,
    flow360_temperature_unit,
    flow360_velocity_unit,
    flow360_area_unit,
    flow360_force_unit,
    flow360_pressure_unit,
    flow360_density_unit,
    flow360_viscosity_unit,
    flow360_power_unit,
    flow360_moment_unit,
    flow360_angular_velocity_unit,
    flow360_heat_flux_unit,
    flow360_heat_capacity_unit,
    flow360_thermal_conductivity_unit,
    flow360_inverse_area_unit,
    flow360_inverse_length_unit,
    flow360_heat_source_unit,
]

_flow360_system = {u.dimension_type.dim_name: u for u in dimensions}


class _PredefinedUnitSystem(UnitSystem):
    mass: MassType = pd.Field(None, exclude=True)
    length: LengthType = pd.Field(None, exclude=True)
    time: TimeType = pd.Field(None, exclude=True)
    temperature: TemperatureType = pd.Field(None, exclude=True)
    velocity: VelocityType = pd.Field(None, exclude=True)
    area: AreaType = pd.Field(None, exclude=True)
    force: ForceType = pd.Field(None, exclude=True)
    pressure: PressureType = pd.Field(None, exclude=True)
    density: DensityType = pd.Field(None, exclude=True)
    viscosity: ViscosityType = pd.Field(None, exclude=True)
    power: PowerType = pd.Field(None, exclude=True)
    moment: MomentType = pd.Field(None, exclude=True)
    angular_velocity: AngularVelocityType = pd.Field(None, exclude=True)
    heat_flux: HeatFluxType = pd.Field(None, exclude=True)
    heat_source: HeatSourceType = pd.Field(None, exclude=True)
    heat_capacity: HeatCapacityType = pd.Field(None, exclude=True)
    thermal_conductivity: ThermalConductivityType = pd.Field(None, exclude=True)
    inverse_area: InverseAreaType = pd.Field(None, exclude=True)
    inverse_length: InverseLengthType = pd.Field(None, exclude=True)

    def system_repr(self):
        return self.name


class SIUnitSystem(_PredefinedUnitSystem):
    """:class: `SIUnitSystem` predefined SI system wrapper"""

    name: Literal["SI"] = pd.Field("SI", frozen=True)

    def __init__(self, verbose: bool = True):
        super().__init__(base_system=BaseSystemType.SI, verbose=verbose)

    @classmethod
    def validate(cls, _):
        return SIUnitSystem()
