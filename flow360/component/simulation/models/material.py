from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class MaterialBase(Flow360BaseModel):
    # Basic properties required to define a material.
    # For example: young's modulus, viscosity as an expression of temperature, etc.
    ...
    name: str = pd.Field()


class Gas(MaterialBase):
    name: Literal["gas"] = pd.Field("gas", frozen=True)
    dynamic_viscosity: Union[ViscosityType, Sutherland] = pd.Field()
    specific_heat_ratio: float = pd.Field()
    gas_constant: GasConstantType = pd.Field()
    Prandtl_number: float = pd.Field()


class Sutherland(Flow360BaseModel):
    reference_viscosity: ViscosityType = pd.Field()
    reference_temperature: TemperatureType = pd.Field()
    effective_temperature: TemperatureType = pd.Field()


class Air(Gas):
    name: Literal["air"] = pd.Field("air", frozen=True)
    dynamic_viscosity: Union[ViscosityType, Sutherland] = pd.Field(
        Sutherland(
            reference_viscosity=1.716e-5, reference_temperature=273, effective_temperature=111
        )
    )
    specific_heat_ratio: Literal["1.4"] = pd.Field("1.4", frozen=True)
    gas_constant: Literal["287.0529"] = pd.Field("287.0529", frozen=True)
    Prandtl_number: Literal["0.72"] = pd.Field("0.72", frozen=True)


class Solid(MaterialBase):
    name: Literal["solid"] = pd.Field("solid", frozen=True)
    thermal_conductivity: ThermalConductivityType = pd.Field()
    density: Optional[DensityType] = pd.Field(None)
    specific_heat_capacity: Optional[SpecificHeatCapacityType] = pd.Field(None)


class Aluminum(Solid):
    name: Literal["Aluminum"] = pd.Field("Aluminum", frozen=True)
    thermal_conductivity: Literal["235"] = pd.Field("235", frozen=True)
    density: Literal["2710"] = pd.Field("2710", frozen=True)
    specific_heat_capacity: Literal["903"] = pd.Field("903", frozen=True)


MaterialTypes = Union[Air, Solid, Aluminum]
