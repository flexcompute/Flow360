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


class FrozenSpecies(Flow360BaseModel):
    """
    Represents a single gas species with NASA 9-coefficient thermodynamic properties.

    Used within :class:`ThermallyPerfectGas` to define multi-species gas mixtures
    where each species contributes to the mixture properties weighted by mass fraction.
    The term "frozen" indicates fixed mass fractions (non-reacting flow).

    Example
    -------

    >>> fl.FrozenSpecies(
    ...     name="N2",
    ...     nasa_9_coefficients=fl.NASA9Coefficients(
    ...         temperature_ranges=[
    ...             fl.NASA9CoefficientSet(
    ...                 temperature_range_min=200.0 * fl.u.K,
    ...                 temperature_range_max=6000.0 * fl.u.K,
    ...                 coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ...             )
    ...         ]
    ...     ),
    ...     mass_fraction=0.7555
    ... )

    ====
    """

    name: str = pd.Field(description="Species name (e.g., 'N2', 'O2', 'Ar')")
    nasa_9_coefficients: NASA9Coefficients = pd.Field(
        description="NASA 9-coefficient polynomial for this species"
    )
    mass_fraction: pd.PositiveFloat = pd.Field(
        description="Mass fraction of this species (must sum to 1 across all species in mixture)"
    )


class ThermallyPerfectGas(Flow360BaseModel):
    """
    Multi-species thermally perfect gas model.

    Combines NASA 9-coefficient polynomials from multiple species weighted by mass fraction.
    All species must use the same temperature range boundaries. The mixture properties
    are computed as mass-fraction-weighted averages of individual species properties.

    This model supports temperature-dependent specific heats (cp) while maintaining
    fixed mass fractions (non-reacting flow).

    Example
    -------

    >>> fl.ThermallyPerfectGas(
    ...     species=[
    ...         fl.FrozenSpecies(name="N2", nasa_9_coefficients=..., mass_fraction=0.7555),
    ...         fl.FrozenSpecies(name="O2", nasa_9_coefficients=..., mass_fraction=0.2316),
    ...         fl.FrozenSpecies(name="Ar", nasa_9_coefficients=..., mass_fraction=0.0129),
    ...     ]
    ... )

    ====
    """

    species: List[FrozenSpecies] = pd.Field(
        min_length=1,
        description="List of species with their NASA 9 coefficients and mass fractions. "
        "Mass fractions must sum to 1.0."
    )

    @pd.model_validator(mode="after")
    def validate_mass_fractions_sum_to_one(self):
        """Validate that mass fractions sum to 1."""
        total = sum(s.mass_fraction for s in self.species)
        if not (0.999 <= total <= 1.001):  # Allow small tolerance for floating point
            raise ValueError(f"Mass fractions must sum to 1.0, got {total}")
        return self

    @pd.model_validator(mode="after")
    def validate_temperature_ranges_match(self):
        """Validate all species have matching temperature range boundaries."""
        if len(self.species) < 2:
            return self
        ref_ranges = self.species[0].nasa_9_coefficients.temperature_ranges
        for species in self.species[1:]:
            ranges = species.nasa_9_coefficients.temperature_ranges
            if len(ranges) != len(ref_ranges):
                raise ValueError(
                    f"Species '{species.name}' has {len(ranges)} temperature ranges, "
                    f"but '{self.species[0].name}' has {len(ref_ranges)}. "
                    "All species must have the same number of temperature ranges."
                )
            for i, (r1, r2) in enumerate(zip(ref_ranges, ranges)):
                if r1.temperature_range_min != r2.temperature_range_min or \
                   r1.temperature_range_max != r2.temperature_range_max:
                    raise ValueError(
                        f"Temperature range {i} boundaries mismatch between species "
                        f"'{self.species[0].name}' and '{species.name}'. "
                        "All species must use the same temperature range boundaries."
                    )
        return self


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

    For multi-species gas mixtures, use the `thermally_perfect_gas` parameter which
    combines species properties weighted by mass fraction.

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

    With multi-species thermally perfect gas:

    >>> fl.Air(
    ...     thermally_perfect_gas=fl.ThermallyPerfectGas(
    ...         species=[
    ...             fl.FrozenSpecies(name="N2", nasa_9_coefficients=..., mass_fraction=0.7555),
    ...             fl.FrozenSpecies(name="O2", nasa_9_coefficients=..., mass_fraction=0.2316),
    ...             fl.FrozenSpecies(name="Ar", nasa_9_coefficients=..., mass_fraction=0.0129),
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
            "For air with gamma=1.4: cp/R = 3.5 (stored in a2). "
            "Note: If thermally_perfect_gas is specified, it takes precedence over this field."
        ),
    )
    thermally_perfect_gas: Optional[ThermallyPerfectGas] = pd.Field(
        default=None,
        description=(
            "Multi-species thermally perfect gas model. When specified, this takes precedence "
            "over nasa_9_coefficients. Use this to define gas mixtures with multiple species, "
            "each with their own NASA 9-coefficient polynomials and mass fractions. "
            "The mixture properties are computed as mass-fraction-weighted averages."
        ),
    )
    prandtl_number: pd.PositiveFloat = pd.Field(
        0.72,
        description="Laminar Prandtl number. Default is 0.72 for air.",
    )
    turbulent_prandtl_number: pd.PositiveFloat = pd.Field(
        0.9,
        description="Turbulent Prandtl number. Default is 0.9.",
    )

    # Default CPG coefficients for comparison (constant gamma=1.4)
    _CPG_COEFFICIENTS: list = [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @property
    def uses_thermally_perfect_gas(self) -> bool:
        """
        Determine if thermally perfect gas model should be used.

        Returns True if:
        - thermally_perfect_gas is explicitly set (multi-species), OR
        - nasa_9_coefficients has been customized (not the default CPG coefficients)

        For backward compatibility, default Air material uses constant gamma (CPG).
        """
        # If multi-species TPG is explicitly set, use TPG
        if self.thermally_perfect_gas is not None:
            return True

        # Check if nasa_9_coefficients has been customized
        # Default is single range with CPG coefficients [0, 0, 3.5, 0, 0, 0, 0, 0, 0]
        ranges = self.nasa_9_coefficients.temperature_ranges
        if len(ranges) != 1:
            return True  # Multiple ranges means customized

        coeffs = list(ranges[0].coefficients)
        return coeffs != self._CPG_COEFFICIENTS

    def get_specific_heat_ratio(self, temperature: AbsoluteTemperatureType) -> pd.PositiveFloat:
        """
        Computes the specific heat ratio (gamma) at a given temperature from NASA polynomial.

        For thermally perfect gas, gamma = cp/cv = (cp/R) / (cp/R - 1) varies with temperature.
        The cp/R is computed from the NASA 9-coefficient polynomial:
            cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4

        Parameters
        ----------
        temperature : AbsoluteTemperatureType
            The temperature at which to compute gamma.

        Returns
        -------
        pd.PositiveFloat
            The specific heat ratio at the given temperature.
        """
        T = temperature.to("K").v.item()

        # Get coefficients from the appropriate source
        if self.thermally_perfect_gas is not None:
            # For multi-species, combine coefficients by mass fraction
            coeffs = [0.0] * 9
            for species in self.thermally_perfect_gas.species:
                # Find the temperature range that contains T
                for coeff_set in species.nasa_9_coefficients.temperature_ranges:
                    t_min = coeff_set.temperature_range_min.to("K").v.item()
                    t_max = coeff_set.temperature_range_max.to("K").v.item()
                    if t_min <= T <= t_max:
                        for i in range(9):
                            coeffs[i] += species.mass_fraction * coeff_set.coefficients[i]
                        break
        else:
            # Single-species: find the temperature range that contains T
            coeffs = None
            for coeff_set in self.nasa_9_coefficients.temperature_ranges:
                t_min = coeff_set.temperature_range_min.to("K").v.item()
                t_max = coeff_set.temperature_range_max.to("K").v.item()
                if t_min <= T <= t_max:
                    coeffs = list(coeff_set.coefficients)
                    break
            if coeffs is None:
                # Fallback to first range if T is out of bounds
                coeffs = list(self.nasa_9_coefficients.temperature_ranges[0].coefficients)

        # Compute cp/R from the polynomial
        # cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4
        cp_over_R = (
            coeffs[0] * T ** (-2)
            + coeffs[1] * T ** (-1)
            + coeffs[2]
            + coeffs[3] * T
            + coeffs[4] * T**2
            + coeffs[5] * T**3
            + coeffs[6] * T**4
        )

        # cv/R = cp/R - 1 (for ideal gas: cp - cv = R)
        cv_over_R = cp_over_R - 1

        # gamma = cp/cv
        gamma = cp_over_R / cv_over_R

        return gamma

    @property
    def specific_heat_ratio(self) -> pd.PositiveFloat:
        """
        Returns the specific heat ratio (gamma) for air at a reference temperature (298.15 K).

        For temperature-dependent gamma, use `get_specific_heat_ratio(temperature)` instead.

        Returns
        -------
        pd.PositiveFloat
            The specific heat ratio at 298.15 K.
        """
        # Compute gamma at reference temperature (298.15 K)
        return self.get_specific_heat_ratio(298.15 * u.K)

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

        For thermally perfect gas, uses the temperature-dependent gamma from the NASA polynomial.

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
        gamma = self.get_specific_heat_ratio(temperature)
        return sqrt(gamma * self.gas_constant * temperature)

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
