"""Material classes for the simulation framework."""

from typing import Literal, Union

import pydantic as pd
import unyt as u
from numpy import sqrt

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.physical_dimensions import (
    AbsoluteTemperature,
    Density,
    Pressure,
    SpecificHeatCapacity,
    ThermalConductivity,
    Velocity,
    Viscosity,
)


def compute_cp_over_r(coeffs, temperature):
    """
    Compute cp/R from NASA 9-coefficient polynomial.

    cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4
    """
    temp = temperature
    return (
        coeffs[0] * temp ** (-2)
        + coeffs[1] * temp ** (-1)
        + coeffs[2]
        + coeffs[3] * temp
        + coeffs[4] * temp**2
        + coeffs[5] * temp**3
        + coeffs[6] * temp**4
    )


def compute_gamma_from_coefficients(coeffs, temperature):
    """
    Compute specific heat ratio (gamma) from NASA 9-coefficient polynomial.

    gamma = cp/cv = (cp/R) / (cp/R - 1)
    """
    cp_r = compute_cp_over_r(coeffs, temperature)
    cv_r = cp_r - 1
    return cp_r / cv_r


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
    """

    type_name: Literal["NASA9CoefficientSet"] = pd.Field("NASA9CoefficientSet", frozen=True)
    temperature_range_min: AbsoluteTemperature.Float64 = pd.Field(
        description="Minimum temperature for which this coefficient set is valid."
    )
    temperature_range_max: AbsoluteTemperature.Float64 = pd.Field(
        description="Maximum temperature for which this coefficient set is valid."
    )
    coefficients: list[float] = pd.Field(
        description="Nine NASA polynomial coefficients [a0, a1, a2, a3, a4, a5, a6, a7, a8]. "
        "a0-a6 are cp/R polynomial coefficients, a7 is the enthalpy integration constant, "
        "and a8 is the entropy integration constant."
    )

    @pd.field_validator("coefficients", mode="after")
    @classmethod
    def validate_coefficients(cls, v):
        """Validate that exactly 9 coefficients are provided."""
        if len(v) != 9:
            raise ValueError(f"NASA 9-coefficient polynomial requires exactly 9 coefficients, got {len(v)}")
        return v

    @pd.field_validator("temperature_range_max", mode="after")
    @classmethod
    def validate_temperature_range_order(cls, v, info):
        """Validate that temperature_range_min < temperature_range_max."""
        t_min = info.data.get("temperature_range_min")
        if t_min is not None:
            t_min_k = t_min.to("K").v.item()
            t_max_k = v.to("K").v.item()
            if t_min_k >= t_max_k:
                raise ValueError(f"temperature_range_min ({t_min}) must be less than " f"temperature_range_max ({v})")
        return v


class NASA9Coefficients(Flow360BaseModel):
    """
    NASA 9-coefficient polynomial coefficients for computing temperature-dependent thermodynamic properties.
    """

    type_name: Literal["NASA9Coefficients"] = pd.Field("NASA9Coefficients", frozen=True)
    temperature_ranges: list[NASA9CoefficientSet] = pd.Field(
        min_length=1,
        max_length=5,
        description="List of NASA 9-coefficient sets for different temperature ranges. "
        "Must be ordered by increasing temperature and be continuous. Maximum 5 ranges supported.",
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

    @pd.validate_call
    def get_coefficients_at_temperature(self, temp_k: float) -> list:
        """
        Get the NASA 9 coefficients for a given temperature.
        """
        for coeff_set in self.temperature_ranges:
            t_min = coeff_set.temperature_range_min.to("K").v.item()
            t_max = coeff_set.temperature_range_max.to("K").v.item()
            if t_min <= temp_k <= t_max:
                return list(coeff_set.coefficients)

        return list(self.temperature_ranges[0].coefficients)


class FrozenSpecies(Flow360BaseModel):
    """
    Represents a single gas species with NASA 9-coefficient thermodynamic properties.
    """

    type_name: Literal["FrozenSpecies"] = pd.Field("FrozenSpecies", frozen=True)
    name: str = pd.Field(description="Species name (e.g., 'N2', 'O2', 'Ar')")
    nasa_9_coefficients: NASA9Coefficients = pd.Field(description="NASA 9-coefficient polynomial for this species")
    mass_fraction: pd.PositiveFloat = pd.Field(
        description="Mass fraction of this species (must sum to 1 across all species in mixture)"
    )


class ThermallyPerfectGas(Flow360BaseModel):
    """
    Multi-species thermally perfect gas model.
    """

    type_name: Literal["ThermallyPerfectGas"] = pd.Field("ThermallyPerfectGas", frozen=True)
    species: list[FrozenSpecies] = pd.Field(
        min_length=1,
        description="List of species with their NASA 9 coefficients and mass fractions. "
        "Mass fractions must sum to 1.0.",
    )

    @pd.model_validator(mode="after")
    def validate_mass_fractions_sum_to_one(self):
        """Validate that mass fractions sum to 1 and re-normalize if within tolerance."""
        total = sum(s.mass_fraction for s in self.species)
        tolerance = 1.0e-3
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Mass fractions must sum to 1.0, got {total}")
        if total != 1.0:
            for species in self.species:
                species.mass_fraction = species.mass_fraction / total
        return self

    @pd.model_validator(mode="after")
    def validate_unique_species_names(self):
        """Validate that all species have unique names."""
        names = [s.name for s in self.species]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Species names must be unique. Duplicates found: {set(duplicates)}")
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
            for i, (r1, r2) in enumerate(zip(ref_ranges, ranges, strict=False)):
                if (
                    r1.temperature_range_min != r2.temperature_range_min
                    or r1.temperature_range_max != r2.temperature_range_max
                ):
                    raise ValueError(
                        f"Temperature range {i} boundaries mismatch between species "
                        f"'{self.species[0].name}' and '{species.name}'. "
                        "All species must use the same temperature range boundaries."
                    )
        return self


class Sutherland(Flow360BaseModel):
    """
    Represents Sutherland's law for calculating dynamic viscosity.
    """

    reference_viscosity: Viscosity.NonNegativeFloat64 = pd.Field(
        description="The reference dynamic viscosity at the reference temperature."
    )
    reference_temperature: AbsoluteTemperature.Float64 = pd.Field(
        description="The reference temperature associated with the reference viscosity."
    )
    effective_temperature: AbsoluteTemperature.Float64 = pd.Field(
        description="The effective temperature constant used in Sutherland's formula."
    )

    @pd.validate_call
    def get_dynamic_viscosity(self, temperature: AbsoluteTemperature.Float64) -> Viscosity.NonNegativeFloat64:
        """
        Calculates the dynamic viscosity at a given temperature using Sutherland's law.
        """
        return self.reference_viscosity * float(
            pow(temperature / self.reference_temperature, 1.5)
            * (self.reference_temperature + self.effective_temperature)
            / (temperature + self.effective_temperature)
        )


class Air(MaterialBase):
    """
    Represents the material properties for air.
    """

    type: Literal["air"] = pd.Field("air", frozen=True)
    name: str = pd.Field("air")
    dynamic_viscosity: Sutherland | Viscosity.NonNegativeFloat64 = pd.Field(
        Sutherland(
            reference_viscosity=1.716e-5 * u.Pa * u.s,
            reference_temperature=273.15 * u.K,
            effective_temperature=110.4 * u.K,
        ),
        description=(
            "The dynamic viscosity model or value for air. Defaults to a `Sutherland` "
            "model with standard atmospheric conditions."
        ),
    )
    thermally_perfect_gas: ThermallyPerfectGas = pd.Field(
        default_factory=lambda: ThermallyPerfectGas(
            species=[
                FrozenSpecies(
                    name="Air",
                    nasa_9_coefficients=NASA9Coefficients(
                        temperature_ranges=[
                            NASA9CoefficientSet(
                                temperature_range_min=200.0 * u.K,
                                temperature_range_max=6000.0 * u.K,
                                coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            ),
                        ]
                    ),
                    mass_fraction=1.0,
                )
            ]
        ),
        description=(
            "Thermally perfect gas model with NASA 9-coefficient polynomials for "
            "temperature-dependent thermodynamic properties. Defaults to a single-species "
            "'Air' with coefficients that reproduce constant gamma=1.4 (calorically perfect gas). "
            "For multi-species gas mixtures, specify multiple FrozenSpecies with their "
            "respective mass fractions."
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

    def get_specific_heat_ratio(self, temperature: AbsoluteTemperature.Float64) -> pd.PositiveFloat:
        """
        Computes the specific heat ratio (gamma) at a given temperature from NASA polynomial.
        """
        temp_k = temperature.to("K").v.item()
        coeffs = self._get_coefficients_at_temperature(temp_k)
        return compute_gamma_from_coefficients(coeffs, temp_k)

    def _get_coefficients_at_temperature(self, temp_k: float) -> list:
        """
        Get the NASA 9 coefficients at a given temperature.
        """
        coeffs = [0.0] * 9
        for species in self.thermally_perfect_gas.species:
            species_coeffs = species.nasa_9_coefficients.get_coefficients_at_temperature(temp_k)
            for i in range(9):
                coeffs[i] += species.mass_fraction * species_coeffs[i]
        return coeffs

    @property
    def gas_constant(self) -> SpecificHeatCapacity.PositiveFloat64:
        """
        Returns the specific gas constant for air.
        """
        return 287.0529 * u.m**2 / u.s**2 / u.K

    @pd.validate_call
    def get_pressure(
        self, density: Density.PositiveFloat64, temperature: AbsoluteTemperature.Float64
    ) -> Pressure.PositiveFloat64:
        """
        Calculates the pressure of air using the ideal gas law.
        """
        temperature = temperature.to("K")
        return density * self.gas_constant * temperature

    @pd.validate_call
    def get_speed_of_sound(self, temperature: AbsoluteTemperature.Float64) -> Velocity.PositiveFloat64:
        """
        Calculates the speed of sound in air at a given temperature.
        """
        temperature = temperature.to("K")
        gamma = self.get_specific_heat_ratio(temperature)
        return sqrt(gamma * self.gas_constant * temperature)

    @pd.validate_call
    def get_dynamic_viscosity(self, temperature: AbsoluteTemperature.Float64) -> Viscosity.NonNegativeFloat64:
        """
        Calculates the dynamic viscosity of air at a given temperature.
        """
        if temperature.units is u.degC or temperature.units is u.degF:
            temperature = temperature.to("K")
        if isinstance(self.dynamic_viscosity, Sutherland):
            return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
        return self.dynamic_viscosity


class SolidMaterial(MaterialBase):
    """
    Represents the solid material properties for heat transfer volume.
    """

    type: Literal["solid"] = pd.Field("solid", frozen=True)
    name: str = pd.Field(frozen=True, description="Name of the solid material.")
    thermal_conductivity: ThermalConductivity.PositiveFloat64 = pd.Field(
        frozen=True, description="Thermal conductivity of the material."
    )
    density: Density.PositiveFloat64 | None = pd.Field(None, frozen=True, description="Density of the material.")
    specific_heat_capacity: SpecificHeatCapacity.PositiveFloat64 | None = pd.Field(
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
    """

    type: Literal["water"] = pd.Field("water", frozen=True)
    name: str = pd.Field(frozen=True, description="Custom name of the water with given property.")
    density: Density.PositiveFloat64 | None = pd.Field(
        1000 * u.kg / u.m**3, frozen=True, description="Density of the water."
    )
    dynamic_viscosity: Viscosity.NonNegativeFloat64 = pd.Field(
        0.001002 * u.kg / u.m / u.s, frozen=True, description="Dynamic viscosity of the water."
    )


SolidMaterialTypes = SolidMaterial
FluidMaterialTypes = Union[Air, Water]
