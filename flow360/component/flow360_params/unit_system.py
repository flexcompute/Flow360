"""
Unit system definitions and utilities
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number
from operator import add, sub
from threading import Lock
from typing import Collection

import pydantic as pd
import unyt as u

from ...utils import classproperty

u.dimensions.viscosity = u.dimensions.pressure * u.dimensions.time
u.dimensions.angular_velocity = u.dimensions.angle / u.dimensions.time

# pylint: disable=no-member
u.unit_systems.mks_unit_system["viscosity"] = u.Pa * u.s
# pylint: disable=no-member
u.unit_systems.mks_unit_system["angular_velocity"] = u.rad / u.s

# pylint: disable=no-member
u.unit_systems.cgs_unit_system["viscosity"] = u.dyn * u.s / u.cm**2
# pylint: disable=no-member
u.unit_systems.cgs_unit_system["angular_velocity"] = u.rad / u.s

# pylint: disable=no-member
u.unit_systems.imperial_unit_system["viscosity"] = u.lbf * u.s / u.ft**2
# pylint: disable=no-member
u.unit_systems.imperial_unit_system["angular_velocity"] = u.rad / u.s


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
            return self._current.copy(deep=True)
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
def _has_dimensions(quant, dim):
    """
    Checks the argument has the right dimensionality.
    """

    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = u.dimensionless
    return arg_dim == dim


def _unit_object_parser(value, unyt_type: type):
    """
    Parses {'value': value, 'units': units}, into unyt_type object : unyt.unyt_quantity, unyt.unyt_array
    """
    if isinstance(value, dict) and "value" in value and "units" in value:
        value = unyt_type(value["value"], value["units"])
    return value


def _is_unit_validator(value):
    """
    Parses str (eg: "m", "cm"), into unyt.Unit object
    """
    if isinstance(value, str):
        value = u.Unit(value)
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


def _has_dimensions_validator(value, dim):
    """
    Checks if value has expected dimention and raises TypeError
    """
    if not _has_dimensions(value, dim):
        raise TypeError(f"arg '{value}' does not match {dim}")
    return value


# pylint: disable=too-few-public-methods
class ValidatedType(ABC):
    """
    :class: Abstract class for dimensioned types with custom validation
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    @abstractmethod
    def validate(cls, value):
        """validation"""


class DimensionedType(ValidatedType):
    """
    :class: Base class for dimensioned values
    """

    dim = None
    dim_name = None

    @classmethod
    def validate(cls, value):
        """
        Validator for value
        """

        value = _unit_object_parser(value, u.unyt_quantity)
        value = _is_unit_validator(value)
        value = _unit_inference_validator(value, cls.dim_name)
        value = _has_dimensions_validator(value, cls.dim)

        return value

    class _Constrained:
        """
        :class: _Constrained
        Note that these constrains work only for values, disregards units.
        We cannot constrain that mass > 2kg, we can only constrain that mass.value > 2
        """

        @classmethod
        def get_class_object(cls, dim_type, **kwargs):
            """Get a dynamically created metaclass representing the constraint"""

            class _ConType:
                type_ = pd.confloat(**kwargs)

            def validate(con_cls, value):
                """Additional validator for value"""
                dimensioned_value = dim_type.validate(value)
                pd.validators.number_size_validator(dimensioned_value.value, con_cls.con_type)
                pd.validators.float_finite_validator(
                    dimensioned_value.value, con_cls.con_type, None
                )
                return dimensioned_value

            cls_obj = type("_Constrained", (), {})
            setattr(cls_obj, "con_type", _ConType)
            setattr(cls_obj, "validate", lambda value: validate(cls_obj, value))
            setattr(cls_obj, "__get_validators__", lambda: (yield getattr(cls_obj, "validate")))
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
        def get_class_object(cls, dim_type, allow_zero_coord=True, allow_zero_norm=True):
            """Get a dynamically created metaclass representing the vector"""

            def validate(vec_cls, value):
                """additional validator for value"""
                value = _unit_object_parser(value, u.unyt_array)
                value = _is_unit_validator(value)

                if not isinstance(value, Collection) and len(value) != 3:
                    raise TypeError(f"arg '{value}' needs to be a collection of 3 values")
                if not vec_cls.allow_zero_coord and any(item == 0 for item in value):
                    raise ValueError(f"arg '{value}' cannot have zero coordinate values")
                if not vec_cls.allow_zero_norm and all(item == 0 for item in value):
                    raise ValueError(f"arg '{value}' cannot have zero norm")

                value = _unit_inference_validator(value, vec_cls.type.dim_name, is_array=True)
                value = _has_dimensions_validator(value, vec_cls.type.dim)

                return value

            cls_obj = type("_VectorType", (), {})
            setattr(cls_obj, "type", dim_type)
            setattr(cls_obj, "allow_zero_norm", allow_zero_norm)
            setattr(cls_obj, "allow_zero_coord", allow_zero_coord)
            setattr(cls_obj, "validate", lambda value: validate(cls_obj, value))
            setattr(cls_obj, "__get_validators__", lambda: (yield getattr(cls_obj, "validate")))
            return cls_obj

    # pylint: disable=invalid-name
    @classproperty
    def Point(self):
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


class AngularVelocityType(DimensionedType):
    """:class: AngularVelocityType"""

    dim = u.dimensions.angular_velocity
    dim_name = "angular_velocity"


class _Flow360BaseUnit(DimensionedType):
    dimension_type = None
    unit_name = None

    @classproperty
    def units(self):
        """
        Retrieve units of a flow360 unit system value
        """

        class _units:
            dimensions = self.dimension_type.dim

        return _units

    @property
    def value(self):
        """
        Retrieve value of a flow360 unit system value
        """
        return self.val

    def __init__(self, val=None) -> None:
        self.val = val

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

    def _unit_iter(self, iterable):
        for value in iter(iterable):
            dimensioned = self.__class__(value)
            yield dimensioned

    def __iter__(self):
        try:
            return self._unit_iter(self.val)
        except TypeError as exc:
            raise TypeError(f"{self} is not iterable") from exc

    def __repr__(self):
        if self.val:
            return f"({self.val}, {self.unit_name})"
        return f"({self.unit_name})"

    def __str__(self):
        if self.val:
            return f"{self.val} {self.unit_name}"
        return f"{self.unit_name}"

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
        if isinstance(other, Collection) and not self.val:
            return self.__class__(other)
        raise TypeError(f"Operation not defined on {self} and {other}")


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


class Flow360AngularVelocityUnit(_Flow360BaseUnit):
    """:class: Flow360AngularVelocityUnit"""

    dimension_type = AngularVelocityType
    unit_name = "flow360_angular_velocity_unit"


_lock = Lock()


class BaseSystemType(Enum):
    """
    :class: Type of the base unit system to use for unit inference (all units need to be specified if not provided)
    """

    SI = "SI"
    CGS = "CGS"
    IMPERIAL = "Imperial"
    FLOW360 = "Flow360"


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
    angular_velocity: AngularVelocityType = pd.Field()

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

    def __init__(self, **kwargs):
        base_system = kwargs.get("base_system")

        dim_names = [
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
            "angular_velocity",
        ]
        units = {}

        for dim in dim_names:
            unit = kwargs.get(dim)
            units[dim] = UnitSystem.__get_unit(base_system, dim, unit)

        missing = set(dim_names) - set(units.keys())

        super().__init__(**units)

        if len(missing) > 0:
            raise ValueError(
                f"Tried defining incomplete unit system, missing definitions for {','.join(missing)}"
            )

    def __getitem__(self, item):
        """to support [] access"""
        return getattr(self, item)

    def __enter__(self):
        _lock.acquire()
        print(f"using: ({self.mass}, {self.length}, {self.time}, {self.temperature}) unit system")
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
flow360_angular_velocity_unit = Flow360AngularVelocityUnit()


_flow360_system = {
    u.dimension_type.dim_name: u
    for u in [
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
        flow360_angular_velocity_unit,
    ]
}


# pylint: disable=too-many-arguments
def flow360_conversion_unit_system(
    base_length=1.0,
    base_mass=1.0,
    base_time=1.0,
    base_temperature=1.0,
    base_velocity=1.0,
    base_density=1.0,
    base_pressure=1.0,
    base_viscosity=1.0,
    base_angular_velocity=1.0,
):
    """
    Register an unyt unit system for flow360 units with defined conversion factors
    """
    _flow360_reg = u.UnitRegistry()
    _flow360_reg.add("grid", base_length, u.dimensions.length)
    _flow360_reg.add("mass", base_mass, u.dimensions.mass)
    _flow360_reg.add("time", base_time, u.dimensions.time)
    _flow360_reg.add("T_inf", base_temperature, u.dimensions.temperature)
    _flow360_reg.add("C_inf", base_velocity, u.dimensions.velocity)
    _flow360_reg.add("rho_inf", base_density, u.dimensions.density)
    _flow360_reg.add("p_inf", base_pressure, u.dimensions.pressure)
    _flow360_reg.add("mu", base_viscosity, u.dimensions.viscosity)
    _flow360_reg.add("omega", base_angular_velocity, u.dimensions.angular_velocity)

    _flow360_conv_system = u.UnitSystem(
        "flow360", "grid", "mass", "time", "T_inf", registry=_flow360_reg
    )

    _flow360_conv_system["velocity"] = "C_inf"
    _flow360_conv_system["density"] = "rho_inf"
    _flow360_conv_system["pressure"] = "p_inf"
    _flow360_conv_system["viscosity"] = "mu"
    _flow360_conv_system["angular_velocity"] = "omega"

    return _flow360_conv_system


SI_unit_system = UnitSystem(base_system=BaseSystemType.SI)
CGS_unit_system = UnitSystem(base_system=BaseSystemType.CGS)
imperial_unit_system = UnitSystem(base_system=BaseSystemType.IMPERIAL)
flow360_unit_system = UnitSystem(base_system=BaseSystemType.FLOW360)
