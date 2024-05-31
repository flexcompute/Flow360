from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


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
            reference_viscosity=1.716e-5, reference_temperature=273, effective_temperature=111
        )
    )

    @property
    def specific_heat_ratio(self) -> PositiveFloat:
        # TODO: serialize
        return 1.4

    @property
    def gas_constant(self) -> SpecificHeatType.Positive:
        return 287.0529

    @property
    def prandtl_number(self) -> PositiveFloat:
        return 0.72


class SolidMaterial(MaterialBase):
    name: Literal["solid"] = pd.Field("solid", frozen=True)
    thermal_conductivity: ThermalConductivityType.Positive = pd.Field()
    density: Optional[DensityType.Positive] = pd.Field(None)
    specific_heat_capacity: Optional[SpecificHeatCapacityType.Positive] = pd.Field(None)


class Aluminum(MaterialBase):
    name: Literal["Aluminum"] = pd.Field("Aluminum", frozen=True)

    @property
    def thermal_conductivity(self) -> ThermalConductivityType.Positive:
        return 235

    @property
    def density(self) -> DensityType.Positive:
        return 2710

    @property
    def specific_heat_capacity(self) -> SpecificHeatCapacityType.Positive:
        return 903


SolidMaterialTypes = Union[SolidMaterial, Aluminum]
FluidMaterialTypes = Air
