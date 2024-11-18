"""Material classes for the simulation framework."""

from typing import Literal, Optional, Union

import pydantic as pd
from numpy import sqrt

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import (
    DensityType,
    PressureType,
    SpecificHeatCapacityType,
    TemperatureType,
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


class Sutherland(Flow360BaseModel):
    """
    Represents Sutherland's law for calculating dynamic viscosity.
    This class implements Sutherland's formula to compute the dynamic viscosity of a gas
    as a function of temperature.
    """

    # pylint: disable=no-member
    reference_viscosity: ViscosityType.NonNegative = pd.Field(
        description="The reference dynamic viscosity at the reference temperature."
    )
    reference_temperature: TemperatureType.Positive = pd.Field(
        description="The reference temperature associated with the reference viscosity."
    )
    effective_temperature: TemperatureType.Positive = pd.Field(
        description="The effective temperature constant used in Sutherland's formula."
    )

    @pd.validate_call
    def get_dynamic_viscosity(
        self, temperature: TemperatureType.Positive
    ) -> ViscosityType.NonNegative:
        """
        Calculates the dynamic viscosity at a given temperature using Sutherland's law.

        Parameters
        ----------
        temperature : TemperatureType.Positive
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
        self, density: DensityType.Positive, temperature: TemperatureType.Positive
    ) -> PressureType.Positive:
        """
        Calculates the pressure of air using the ideal gas law.

        Parameters
        ----------
        density : DensityType.Positive
            The density of the air.
        temperature : TemperatureType.Positive
            The temperature of the air.

        Returns
        -------
        PressureType.Positive
            The calculated pressure.
        """
        return density * self.gas_constant * temperature

    @pd.validate_call
    def get_speed_of_sound(self, temperature: TemperatureType.Positive) -> VelocityType.Positive:
        """
        Calculates the speed of sound in air at a given temperature.

        Parameters
        ----------
        temperature : TemperatureType.Positive
            The temperature at which to calculate the speed of sound.

        Returns
        -------
        VelocityType.Positive
            The speed of sound at the specified temperature.
        """
        return sqrt(self.specific_heat_ratio * self.gas_constant * temperature)

    @pd.validate_call
    def get_dynamic_viscosity(
        self, temperature: TemperatureType.Positive
    ) -> ViscosityType.NonNegative:
        """
        Calculates the dynamic viscosity of air at a given temperature.

        Parameters
        ----------
        temperature : TemperatureType.Positive
            The temperature at which to calculate the dynamic viscosity.

        Returns
        -------
        ViscosityType.NonNegative
            The dynamic viscosity at the specified temperature.
        """
        if isinstance(self.dynamic_viscosity, Sutherland):
            return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
        return self.dynamic_viscosity


class SolidMaterial(MaterialBase):
    """
    Represents the solid material properties for heat transfer volume.
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


SolidMaterialTypes = SolidMaterial
FluidMaterialTypes = Air
