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
    Sutherland's law
    """

    # pylint: disable=no-member
    reference_viscosity: ViscosityType.Positive = pd.Field()
    reference_temperature: TemperatureType.Positive = pd.Field()
    effective_temperature: TemperatureType.Positive = pd.Field()

    @pd.validate_call
    def get_dynamic_viscosity(
        self, temperature: TemperatureType.Positive
    ) -> ViscosityType.Positive:
        """dynamic viscosity"""
        return (
            self.reference_viscosity
            * pow(temperature / self.reference_temperature, 1.5)
            * (self.reference_temperature + self.effective_temperature)
            / (temperature + self.effective_temperature)
        )


# pylint: disable=no-member, missing-function-docstring
class Air(MaterialBase):
    """
    Material properties for Air
    """

    type: Literal["air"] = pd.Field("air", frozen=True)
    name: str = pd.Field("air")
    dynamic_viscosity: Union[Sutherland, ViscosityType.Positive] = pd.Field(
        Sutherland(
            reference_viscosity=1.716e-5 * u.Pa * u.s,
            reference_temperature=273.15 * u.K,
            # pylint: disable=fixme
            # TODO: validation error for effective_temperature not equal 110.4 K
            effective_temperature=110.4 * u.K,
        )
    )

    @property
    def specific_heat_ratio(self) -> pd.PositiveFloat:
        return 1.4

    @property
    def gas_constant(self) -> SpecificHeatCapacityType.Positive:
        return 287.0529 * u.m**2 / u.s**2 / u.K

    @property
    def prandtl_number(self) -> pd.PositiveFloat:
        return 0.72

    @pd.validate_call
    def get_pressure(
        self, density: DensityType.Positive, temperature: TemperatureType.Positive
    ) -> PressureType.Positive:
        return density * self.gas_constant * temperature

    @pd.validate_call
    def get_speed_of_sound(self, temperature: TemperatureType.Positive) -> VelocityType.Positive:
        return sqrt(self.specific_heat_ratio * self.gas_constant * temperature)

    @pd.validate_call
    def get_dynamic_viscosity(
        self, temperature: TemperatureType.Positive
    ) -> ViscosityType.Positive:
        if isinstance(self.dynamic_viscosity, Sutherland):
            return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
        return self.dynamic_viscosity


class SolidMaterial(MaterialBase):
    """
    Solid material base
    """

    type: Literal["solid"] = pd.Field("solid", frozen=True)
    name: str = pd.Field(frozen=True)
    thermal_conductivity: ThermalConductivityType.Positive = pd.Field(frozen=True)
    density: Optional[DensityType.Positive] = pd.Field(None, frozen=True)
    specific_heat_capacity: Optional[SpecificHeatCapacityType.Positive] = pd.Field(
        None, frozen=True
    )


aluminum = SolidMaterial(
    name="aluminum",
    thermal_conductivity=235 * u.kg / u.s**3 * u.m / u.K,
    density=2710 * u.kg / u.m**3,
    specific_heat_capacity=903 * u.m**2 / u.s**2 / u.K,
)


SolidMaterialTypes = SolidMaterial
FluidMaterialTypes = Air
