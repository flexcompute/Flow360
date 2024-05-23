from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class MaterialBase(Flow360BaseModel):
    # Basic properties required to define a material.
    # For example: young's modulus, viscosity as an expression of temperature, etc.
    ...
    name: str = pd.Field()


class Gas(MaterialBase):
    # contains models of getting properites, for example US standard atmosphere model
    name: Literal["gas"] = pd.Field(frozen=True)
    viscosity_type: Literal["Sutherland", "constant"] = pd.field("Sutherland")
    constant_dynamic_viscosity: Optional[ViscosityType] = pd.Field(None)
    sutherland_constants: Optional[SutherlandConstants] = pd.Field(None)
    specific_heat_ratio: float = pd.Field()
    gas_constant: GasConstantType = pd.Field()
    Prandtl_number: float = pd.Field()

    def viscosity_from_temperature(self, temperature: TemperatureType) -> ViscosityType:
        """Sutherland's Law"""
        return constant_dynamic_viscosity

    def speed_of_sound(self, temperature: TemperatureType) -> VelocityType:
        """Calculates the speed of sound in the air based on the temperature. Returns dimensioned value"""
        return np.sqrt(self.specific_heat_ratio * self.gas_constant * temperature.to("K")).to("m/s")


class SutherlandConstants(Flow360BaseModel):
    reference_viscosity: ViscosityType = pd.Field()
    reference_temperature: TemperatureType = pd.Field()
    effective_temperature: TemperatureType = pd.Field()


class Air(Gas):
    name: Literal["air"] = pd.Field(frozen=True)
    constant_dynamic_viscosity: ViscosityType = pd.Field(1.825e-5)
    sutherland_constants: SutherlandConstants = pd.Field(
        SutherlandConstants(
            reference_viscosity=1.716e-5, reference_temperature=273, effective_temperature=111
        )
    )
    specific_heat_ratio: float = pd.Field(1.4, frozen=True)
    gas_constant: GasConstantType = pd.Field(287.0529, frozen=True)
    Prandtl_number: float = pd.Field(0.72, frozen=True)


class Solid(MaterialBase):
    name: Literal["solid"] = pd.Field(frozen=True)
    thermal_conductivity: ThermalConductivityType = pd.Field()
    density: Optional[ThermalConductivityType] = pd.Field(None)
    specific_heat_capacity: Optional[SpecificHeatCapacityType] = pd.Field(None)


class Aluminum(Solid):
    name: Literal["Aluminum"] = pd.Field(frozen=True)
    thermal_conductivity: ThermalConductivityType = pd.Field(235)
    density: DensityType = pd.Field(2710)
    specific_heat_capacity: SpecificHeatCapacityType = pd.Field(903)


MaterialTypes = Union[Air, Solid, Aluminum]
