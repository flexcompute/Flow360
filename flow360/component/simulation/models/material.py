"""Material classes for the simulation framework."""

from typing import List, Literal, Optional, Union

import pydantic as pd
from numpy import sqrt

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import (
    AbsoluteTemperatureType,
    DensityType,
    PressureType,
    SpecificHeatCapacityType,
    ThermalConductivityType,
    VelocityType,
    ViscosityType,
)


class MaterialBase(Flow360BaseModel):
    """
    Basic properties required to define a material.
    For example: young's modulus, viscosity as an expression of temperature, etc.
    """

    type: str = pd.Field()
    name: str = pd.Field()


class NASA9CoefficientSet(Flow360BaseModel):
    """
    Represents a set of 9 NASA polynomial coefficients for a specific temperature range.

    The NASA 9-coefficient polynomial (McBride et al., 2002) computes thermodynamic
    properties as:

    cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4

    h/RT = -a0*T^-2 + a1*ln(T)/T + a2 + (a3/2)*T + (a4/3)*T^2 + (a5/4)*T^3 + (a6/5)*T^4 + a7/T

    s/R = -(a0/2)*T^-2 - a1*T^-1 + a2*ln(T) + a3*T + (a4/2)*T^2 + (a5/3)*T^3 + (a6/4)*T^4 + a8

    Coefficients:
    - a0-a6: cp polynomial coefficients
    - a7: enthalpy integration constant
    - a8: entropy integration constant

    Example
    -------

    >>> fl.NASA9CoefficientSet(
    ...     temperature_range_min=200.0 * fl.u.K,
    ...     temperature_range_max=1000.0 * fl.u.K,
    ...     coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ... )

    ====
    """

    temperature_range_min: AbsoluteTemperatureType = pd.Field(
        description="Minimum temperature for which this coefficient set is valid."
    )
    temperature_range_max: AbsoluteTemperatureType = pd.Field(
        description="Maximum temperature for which this coefficient set is valid."
    )
    coefficients: List[float] = pd.Field(
        description="Nine NASA polynomial coefficients [a0, a1, a2, a3, a4, a5, a6, a7, a8]. "
        "a0-a6 are cp/R polynomial coefficients, a7 is the enthalpy integration constant, "
        "and a8 is the entropy integration constant."
    )

    @pd.model_validator(mode="after")
    def validate_coefficients(self):
        """Validate that exactly 9 coefficients are provided."""
        if len(self.coefficients) != 9:
            raise ValueError(
                f"NASA 9-coefficient polynomial requires exactly 9 coefficients, "
                f"got {len(self.coefficients)}"
            )
        return self


class NASA9Coefficients(Flow360BaseModel):
    """
    NASA 9-coefficient polynomial coefficients for computing temperature-dependent thermodynamic properties.

    Supports 1-5 temperature ranges with continuous boundaries. Defaults to a single temperature range.

    Example
    -------

    Single temperature range (default):

    >>> fl.NASA9Coefficients(
    ...     temperature_ranges=[
    ...         fl.NASA9CoefficientSet(
    ...             temperature_range_min=200.0 * fl.u.K,
    ...             temperature_range_max=6000.0 * fl.u.K,
    ...             coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ...         )
    ...     ]
    ... )

    Multiple temperature ranges:

    >>> fl.NASA9Coefficients(
    ...     temperature_ranges=[
    ...         fl.NASA9CoefficientSet(
    ...             temperature_range_min=200.0 * fl.u.K,
    ...             temperature_range_max=1000.0 * fl.u.K,
    ...             coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ...         ),
    ...         fl.NASA9CoefficientSet(
    ...             temperature_range_min=1000.0 * fl.u.K,
    ...             temperature_range_max=6000.0 * fl.u.K,
    ...             coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ...         )
    ...     ]
    ... )

    ====
    """

    temperature_ranges: List[NASA9CoefficientSet] = pd.Field(
        min_length=1,
        max_length=5,
        description="List of NASA 9-coefficient sets for different temperature ranges. "
        "Must be ordered by increasing temperature and be continuous. Maximum 5 ranges supported."
    )

    @pd.model_validator(mode="after")
    def validate_temperature_continuity(self):
        """Validate that temperature ranges are continuous and non-overlapping."""
        for i in range(len(self.temperature_ranges) - 1):
            current_max = self.temperature_ranges[i].temperature_range_max
            next_min = self.temperature_ranges[i + 1].temperature_range_min
            if current_max != next_min:
                raise ValueError(
                    f"Temperature ranges must be continuous: range {i} max "
                    f"({current_max}) must equal range {i+1} min ({next_min})"
                )
        return self


# Legacy aliases for backward compatibility during transition
NASAPolynomialCoefficientSet = NASA9CoefficientSet
NASAPolynomialCoefficients = NASA9Coefficients


class Sutherland(Flow360BaseModel):
    """
    Represents Sutherland's law for calculating dynamic viscosity.
    This class implements Sutherland's formula to compute the dynamic viscosity of a gas
    as a function of temperature.

    Example
    -------

    >>> fl.Sutherland(
    ...     reference_viscosity=1.70138e-5 * fl.u.Pa * fl.u.s,
    ...     reference_temperature=300.0 * fl.u.K,
    ...     effective_temperature=110.4 * fl.u.K,
    ... )

    ====
    """

    # pylint: disable=no-member
    reference_viscosity: ViscosityType.NonNegative = pd.Field(
        description="The reference dynamic viscosity at the reference temperature."
    )
    reference_temperature: AbsoluteTemperatureType = pd.Field(
        description="The reference temperature associated with the reference viscosity."
    )
    effective_temperature: AbsoluteTemperatureType = pd.Field(
        description="The effective temperature constant used in Sutherland's formula."
    )

    @pd.validate_call
    def get_dynamic_viscosity(
        self, temperature: AbsoluteTemperatureType
    ) -> ViscosityType.NonNegative:
        """
        Calculates the dynamic viscosity at a given temperature using Sutherland's law.

        Parameters
        ----------
        temperature : AbsoluteTemperatureType
            The temperature at which to calculate the dynamic viscosity.

        Returns
        -------
        ViscosityType.NonNegative
            The calculated dynamic viscosity at the specified temperature.
        """
        return self.reference_viscosity * float(
            pow(temperature / self.reference_temperature, 1.5)
            * (self.reference_temperature + self.effective_temperature)
            / (temperature + self.effective_temperature)
        )


# pylint: disable=no-member, missing-function-docstring
class Air(MaterialBase):
    """
    Represents the material properties for air.
    This sets specific material properties for air,
    including dynamic viscosity, specific heat ratio, gas constant, and Prandtl number.

    The thermodynamic properties can be specified using NASA 9-coefficient polynomials
    for temperature-dependent specific heats. By default, coefficients are set to
    reproduce a constant gamma=1.4 (calorically perfect gas).

    Example
    -------

    >>> fl.Air(
    ...     dynamic_viscosity=1.063e-05 * fl.u.Pa * fl.u.s
    ... )

    With custom NASA 9-coefficient polynomial:

    >>> fl.Air(
    ...     nasa_9_coefficients=fl.NASA9Coefficients(
    ...         temperature_ranges=[
    ...             fl.NASA9CoefficientSet(
    ...                 temperature_range_min=200.0 * fl.u.K,
    ...                 temperature_range_max=6000.0 * fl.u.K,
    ...                 coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ...             )
    ...         ]
    ...     )
    ... )

    ====
    """

    type: Literal["air"] = pd.Field("air", frozen=True)
    name: str = pd.Field("air")
    dynamic_viscosity: Union[Sutherland, ViscosityType.NonNegative] = pd.Field(
        Sutherland(
            reference_viscosity=1.716e-5 * u.Pa * u.s,
            reference_temperature=273.15 * u.K,
            # pylint: disable=fixme
            # TODO: validation error for effective_temperature not equal 110.4 K
            effective_temperature=110.4 * u.K,
        ),
        description=(
            "The dynamic viscosity model or value for air. Defaults to a `Sutherland` "
            "model with standard atmospheric conditions."
        ),
    )
    nasa_9_coefficients: NASA9Coefficients = pd.Field(
        default_factory=lambda: NASA9Coefficients(
            temperature_ranges=[
                NASA9CoefficientSet(
                    temperature_range_min=200.0 * u.K,
                    temperature_range_max=6000.0 * u.K,
                    # For constant gamma=1.4: cp/R = gamma/(gamma-1) = 1.4/0.4 = 3.5
                    # In NASA9 format, constant cp/R is the a2 coefficient (index 2)
                    # All other coefficients (inverse T terms, positive T terms, integration constants) are zero
                    coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ]
        ),
        description=(
            "NASA 9-coefficient polynomial coefficients for computing temperature-dependent "
            "thermodynamic properties (cp, enthalpy, entropy). Defaults to a single temperature "
            "range with coefficients that reproduce constant gamma=1.4 (calorically perfect gas). "
            "For air with gamma=1.4: cp/R = 3.5 (stored in a2)."
        ),
    )

    @property
    def specific_heat_ratio(self) -> pd.PositiveFloat:
        """
        Returns the specific heat ratio (gamma) for air.

        Returns
        -------
        pd.PositiveFloat
            The specific heat ratio, typically 1.4 for air.
        """
        return 1.4

    @property
    def gas_constant(self) -> SpecificHeatCapacityType.Positive:
        """
        Returns the specific gas constant for air.

        Returns
        -------
        SpecificHeatCapacityType.Positive
            The specific gas constant for air.
        """

        return 287.0529 * u.m**2 / u.s**2 / u.K

    @property
    def prandtl_number(self) -> pd.PositiveFloat:
        """
        Returns the Prandtl number for air.

        Returns
        -------
        pd.PositiveFloat
            The Prandtl number, typically around 0.72 for air.
        """

        return 0.72

    @pd.validate_call
    def get_pressure(
        self, density: DensityType.Positive, temperature: AbsoluteTemperatureType
    ) -> PressureType.Positive:
        """
        Calculates the pressure of air using the ideal gas law.

        Parameters
        ----------
        density : DensityType.Positive
            The density of the air.
        temperature : AbsoluteTemperatureType
            The temperature of the air.

        Returns
        -------
        PressureType.Positive
            The calculated pressure.
        """
        temperature = temperature.to("K")
        return density * self.gas_constant * temperature

    @pd.validate_call
    def get_speed_of_sound(self, temperature: AbsoluteTemperatureType) -> VelocityType.Positive:
        """
        Calculates the speed of sound in air at a given temperature.

        Parameters
        ----------
        temperature : AbsoluteTemperatureType
            The temperature at which to calculate the speed of sound.

        Returns
        -------
        VelocityType.Positive
            The speed of sound at the specified temperature.
        """
        temperature = temperature.to("K")
        return sqrt(self.specific_heat_ratio * self.gas_constant * temperature)

    @pd.validate_call
    def get_dynamic_viscosity(
        self, temperature: AbsoluteTemperatureType
    ) -> ViscosityType.NonNegative:
        """
        Calculates the dynamic viscosity of air at a given temperature.

        Parameters
        ----------
        temperature : AbsoluteTemperatureType
            The temperature at which to calculate the dynamic viscosity.

        Returns
        -------
        ViscosityType.NonNegative
            The dynamic viscosity at the specified temperature.
        """
        if temperature.units is u.degC or temperature.units is u.degF:
            temperature = temperature.to("K")
        if isinstance(self.dynamic_viscosity, Sutherland):
            return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
        return self.dynamic_viscosity


class SolidMaterial(MaterialBase):
    """
    Represents the solid material properties for heat transfer volume.

    Example
    -------

    >>> fl.SolidMaterial(
    ...     name="aluminum",
    ...     thermal_conductivity=235 * fl.u.kg / fl.u.s**3 * fl.u.m / fl.u.K,
    ...     density=2710 * fl.u.kg / fl.u.m**3,
    ...     specific_heat_capacity=903 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
    ... )

    ====
    """

    type: Literal["solid"] = pd.Field("solid", frozen=True)
    name: str = pd.Field(frozen=True, description="Name of the solid material.")
    thermal_conductivity: ThermalConductivityType.Positive = pd.Field(
        frozen=True, description="Thermal conductivity of the material."
    )
    density: Optional[DensityType.Positive] = pd.Field(
        None, frozen=True, description="Density of the material."
    )
    specific_heat_capacity: Optional[SpecificHeatCapacityType.Positive] = pd.Field(
        None, frozen=True, description="Specific heat capacity of the material."
    )


aluminum = SolidMaterial(
    name="aluminum",
    thermal_conductivity=235 * u.kg / u.s**3 * u.m / u.K,
    density=2710 * u.kg / u.m**3,
    specific_heat_capacity=903 * u.m**2 / u.s**2 / u.K,
)


class Water(MaterialBase):
    """
    Water material used for :class:`LiquidOperatingCondition`

    Example
    -------

    >>> fl.Water(
    ...     name="Water",
    ...     density=1000 * fl.u.kg / fl.u.m**3,
    ...     dynamic_viscosity=0.001002 * fl.u.kg / fl.u.m / fl.u.s,
    ... )

    ====
    """

    type: Literal["water"] = pd.Field("water", frozen=True)
    name: str = pd.Field(frozen=True, description="Custom name of the water with given property.")
    density: Optional[DensityType.Positive] = pd.Field(
        1000 * u.kg / u.m**3, frozen=True, description="Density of the water."
    )
    dynamic_viscosity: ViscosityType.NonNegative = pd.Field(
        0.001002 * u.kg / u.m / u.s, frozen=True, description="Dynamic viscosity of the water."
    )


SolidMaterialTypes = SolidMaterial
FluidMaterialTypes = Union[Air, Water]
