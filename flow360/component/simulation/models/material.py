from typing import Literal, Optional, Union

import pydantic as pd
import unyt as u

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import (
    DensityType,
    HeatCapacityType,
    SI_unit_system,
    TemperatureType,
    ThermalConductivityType,
    ViscosityType,
)


class MaterialBase(Flow360BaseModel):
    # Basic properties required to define a material.
    # For example: young's modulus, viscosity as an expression of temperature, etc.
    ...
    name: str = pd.Field()


class Sutherland(Flow360BaseModel):
    reference_viscosity: ViscosityType.Positive = pd.Field()
    reference_temperature: TemperatureType.Positive = pd.Field()
    effective_temperature: TemperatureType.Positive = pd.Field()


class Air(Flow360BaseModel):
    name: Literal["air"] = pd.Field("air", frozen=True)
    dynamic_viscosity: Union[ViscosityType.Positive, Sutherland] = pd.Field(
        Sutherland(
            reference_viscosity=1.716e-5 * u.Pa * u.s,
            reference_temperature=273 * u.K,
            effective_temperature=111 * u.K,
        )
    )

    @property
    def specific_heat_ratio(self) -> pd.PositiveFloat:
        # TODO: serialize
        return 1.4

    @property
    def gas_constant(self) -> HeatCapacityType.Positive:
        return 287.0529 * u.m**2 / u.s**2 / u.K

    @property
    def prandtl_number(self) -> pd.PositiveFloat:
        return 0.72


class SolidMaterial(MaterialBase):
    name: Literal["solid"] = pd.Field("solid", frozen=True)
    thermal_conductivity: ThermalConductivityType.Positive = pd.Field(frozen=True)
    density: Optional[DensityType.Positive] = pd.Field(None, frozen=True)
    specific_heat_capacity: Optional[HeatCapacityType.Positive] = pd.Field(None, frozen=True)


with SI_unit_system:
    aluminum = SolidMaterial(thermal_conductivity=235, density=2710, specific_heat_capacity=903)


SolidMaterialTypes = SolidMaterial
FluidMaterialTypes = Air
