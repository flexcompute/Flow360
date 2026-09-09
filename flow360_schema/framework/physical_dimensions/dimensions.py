"""
Physical dimension type namespaces

Each class provides all type variants for that physical dimension.

The ``si_unit`` strings use the UnitsDSL grammar, which is compatible with
tinyExpr (github.com/codeplea/tinyexpr). Supported operators:

    ^   exponentiation          meter^2
    *   multiplication          kilogram*meter
    /   division                meter/second
    ()  grouping                kilogram/(meter*second^2)
    -   unary minus (in exp)    meter^-2

Precedence (high → low): unary ±, ^, * / , + -
Parentheses are required for compound exponents: X^(-a/b+c).
Without them, X^-a/b+c parses as ((X^-a)/b)+c.
"""

from typing import Any

from .constraint_kinds import ConstraintKind
from .dimension_base import PhysicalDimensionBase
from .dimension_meta import PhysicalDimensionMeta


def _delta_temperature_validation_hook(unyt_value: Any, *_: Any) -> Any:
    """Reject offset-based temperature units (e.g. degC, degF) for delta temperature fields.

    Delta temperatures represent differences, so only offset-free units are valid:
    K, R, delta_degC, delta_degF, etc.  Offset units like degC/degF have a non-zero
    base_offset in unyt, which makes them unsuitable for temperature differences.
    """
    if unyt_value.units.base_offset != 0:
        raise ValueError(
            f"Invalid delta temperature unit '{unyt_value.units}': "
            f"offset-based temperature units are not allowed for temperature differences. "
            f"Use delta units (e.g. delta_degC, delta_degF) or absolute units (K, R) instead."
        )
    return unyt_value.value


def _absolute_temperature_validation_hook(unyt_value: Any, *_: Any) -> Any:
    """Validate absolute temperature against the physical lower bound (0 K)."""
    kelvin_value = unyt_value.to("K").value

    try:
        below_absolute_zero = bool((kelvin_value < 0).any())
    except AttributeError:
        below_absolute_zero = kelvin_value < 0

    if below_absolute_zero:
        raise ValueError(f"Invalid absolute temperature {unyt_value}: value cannot be below absolute zero (0 K).")

    return unyt_value.value


class Length(PhysicalDimensionBase):
    """Length physical dimension (SI unit: meter)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="length", si_unit="meter")


class Area(PhysicalDimensionBase):
    """Area physical dimension (SI unit: meter^2)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="area", si_unit="meter^2")


class Density(PhysicalDimensionBase):
    """Density physical dimension (SI unit: kilogram/meter^3)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="density", si_unit="kilogram/meter^3")


class AbsoluteTemperature(PhysicalDimensionBase):
    """Absolute temperature physical dimension (SI unit: kelvin)"""

    physical_dimension_meta = PhysicalDimensionMeta(
        name="temperature",
        si_unit="kelvin",
        allow_zero=False,
        supported_constraint_kinds=(ConstraintKind.NO_RANGE,),
        validation_value_hook=_absolute_temperature_validation_hook,
    )


class DeltaTemperature(PhysicalDimensionBase):
    """Delta temperature physical dimension (SI unit: kelvin)"""

    physical_dimension_meta = PhysicalDimensionMeta(
        name="delta_temperature",
        si_unit="kelvin",
        validation_value_hook=_delta_temperature_validation_hook,
    )


class Angle(PhysicalDimensionBase):
    """Angle physical dimension (SI unit: radian)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="angle", si_unit="radian", unit_system_inference=False)


class Mass(PhysicalDimensionBase):
    """Mass physical dimension (SI unit: kilogram)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="mass", si_unit="kilogram")


class Time(PhysicalDimensionBase):
    """Time physical dimension (SI unit: second)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="time", si_unit="second")


class Velocity(PhysicalDimensionBase):
    """Velocity physical dimension (SI unit: meter/second)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="velocity", si_unit="meter/second")


class Acceleration(PhysicalDimensionBase):
    """Acceleration physical dimension (SI unit: meter/second^2)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="acceleration", si_unit="meter/second^2")


class Force(PhysicalDimensionBase):
    """Force physical dimension (SI unit: newton)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="force", si_unit="newton")


class Pressure(PhysicalDimensionBase):
    """Pressure physical dimension (SI unit: pascal)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="pressure", si_unit="pascal")


class Viscosity(PhysicalDimensionBase):
    """Dynamic viscosity physical dimension (SI unit: pascal*second)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="viscosity", si_unit="pascal*second")


class KinematicViscosity(PhysicalDimensionBase):
    """Kinematic viscosity physical dimension (SI unit: meter^2/second)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="kinematic_viscosity", si_unit="meter^2/second")


class Power(PhysicalDimensionBase):
    """Power physical dimension (SI unit: watt)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="power", si_unit="watt")


class Moment(PhysicalDimensionBase):
    """Moment/torque physical dimension (SI unit: newton*meter)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="moment", si_unit="newton*meter")


class AngularVelocity(PhysicalDimensionBase):
    """Angular velocity physical dimension (SI unit: radian/second)"""

    physical_dimension_meta = PhysicalDimensionMeta(
        name="angular_velocity", si_unit="radian/second", unit_system_inference=False
    )


class HeatFlux(PhysicalDimensionBase):
    """Heat flux physical dimension (SI unit: watt/meter^2)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="heat_flux", si_unit="watt/meter^2")


class HeatSource(PhysicalDimensionBase):
    """Heat source physical dimension (SI unit: watt/meter^3)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="heat_source", si_unit="watt/meter^3")


class SpecificHeatCapacity(PhysicalDimensionBase):
    """Specific heat capacity physical dimension (SI unit: joule/(kilogram*kelvin))"""

    physical_dimension_meta = PhysicalDimensionMeta(name="specific_heat_capacity", si_unit="joule/(kilogram*kelvin)")


class ThermalConductivity(PhysicalDimensionBase):
    """Thermal conductivity physical dimension (SI unit: watt/(meter*kelvin))"""

    physical_dimension_meta = PhysicalDimensionMeta(name="thermal_conductivity", si_unit="watt/(meter*kelvin)")


class InverseArea(PhysicalDimensionBase):
    """Inverse area physical dimension (SI unit: meter^-2)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="inverse_area", si_unit="meter^-2")


class InverseLength(PhysicalDimensionBase):
    """Inverse length physical dimension (SI unit: meter^-1)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="inverse_length", si_unit="meter^-1")


class MassFlowRate(PhysicalDimensionBase):
    """Mass flow rate physical dimension (SI unit: kilogram/second)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="mass_flow_rate", si_unit="kilogram/second")


class SpecificEnergy(PhysicalDimensionBase):
    """Specific energy physical dimension (SI unit: joule/kilogram)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="specific_energy", si_unit="joule/kilogram")


class MolarMass(PhysicalDimensionBase):
    """Molar mass (molecular weight) physical dimension (SI unit: kilogram/mol).

    The SI base for amount of substance is the *mole*, so the SI unit is ``kg/mol``
    (e.g. N2 ~= 0.028 kg/mol). Any per-mole equivalent (g/mol, kg/kmol) is also
    accepted; bare-mass values (``kg``, ``g``) without the mole component are
    rejected by ``check_dimension`` (#5730).

    ``unit_system_inference`` is disabled (as for ``Angle``) because molar mass
    has no natural unit-system default; users must supply explicit units,
    e.g. ``28.97 * u.g / u.mol``.
    """

    physical_dimension_meta = PhysicalDimensionMeta(
        name="molar_mass", si_unit="kilogram/mol", unit_system_inference=False
    )


class Frequency(PhysicalDimensionBase):
    """Frequency physical dimension (SI unit: hertz)"""

    physical_dimension_meta = PhysicalDimensionMeta(name="frequency", si_unit="hertz")


__all__ = [
    "Length",
    "Area",
    "Density",
    "AbsoluteTemperature",
    "DeltaTemperature",
    "Angle",
    "Mass",
    "Time",
    "Velocity",
    "Acceleration",
    "Force",
    "Pressure",
    "Viscosity",
    "KinematicViscosity",
    "Power",
    "Moment",
    "AngularVelocity",
    "HeatFlux",
    "HeatSource",
    "SpecificHeatCapacity",
    "ThermalConductivity",
    "InverseArea",
    "InverseLength",
    "MassFlowRate",
    "SpecificEnergy",
    "MolarMass",
    "Frequency",
]
