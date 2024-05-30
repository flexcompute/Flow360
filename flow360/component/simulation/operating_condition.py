from typing import Optional, Union

import numpy as np
import pydantic as pd
from pydantic import validate_arguments

from flow360.component.simulation.framework.base_model import Flow360BaseModel

VelocityVectorType = Unioin[VelocityType.Vector, Tuple[pd.StrictStr, pd.StrictStr, pd.StrictStr]]


class ThermalCondition(Flow360BaseModel):
    temperature: TemperatureType = 288.15
    density: DensityType = 1.225
    material: materialTypes = Air()
    # TODO: special serializer
    _altitude: Optional[LengthType] = None
    _temperature_offset: Optional[TemperatureType] = None

    @validate_arguments
    @classmethod
    def from_standard_atmosphere(
        cls, altitude: LengthType = 0, temperature_offset: TemperatureType = 0
    ):
        # TODO: add standard atmosphere implementation
        density = 1.225
        temperature = 288.15

        return cls(
            density=density,
            temperature=temperature,
            material=Air(),
        )

    @property
    def altitude(self) -> LengthType:
        return self._altitude

    @property
    def temperature_offset(self) -> TemperatureType:
        return self._temperature_offset

    @property
    def speed_of_sound(self) -> VelocityType:
        return np.sqrt(
            self.material.specific_heat_ratio * self.material.gas_constant * self.temperature
        )

    @property
    def pressure(self) -> PressureType:
        # TODO: implement
        return 1.013e5

    @property
    def dynamic_viscosity(self) -> ViscosityType:
        # TODO: implement
        return 1.825e-5


class GenericReferenceCondition(Flow360BaseModel):
    """
    Operating condition defines the physical (non-geometrical) reference values for the problem.
    """

    velocity_magnitude: VelocityType.Positive
    thermal_condition: ThermalCondition = ThermalCondition()

    @validate_arguments
    @classmethod
    def from_Mach(
        cls,
        mach: PositiveFloat,
        thermal_condition: ThermalCondition = ThermalCondition(),
    ):
        velocity_magnitude = mach * self.thermal_condition.speed_of_sound
        return cls(velocity_magnitude=velocity_magnitude, thermal_condition=thermal_condition)


class AerospaceCondition(Flow360BaseModel):
    alpha: float = 0
    beta: float = 0
    velocity_magnitude: VelocityType.NonNegative
    atmosphere: ThermalCondition = ThermalCondition()
    reference_velocity_magnitude: Optional[VelocityType.Positive] = None

    @validate_arguments
    @classmethod
    def from_Mach(
        cls,
        mach: PositiveFloat,
        alpha: float = 0,
        beta: float = 0,
        atmosphere: ThermalCondition = ThermalCondition(),
    ):
        pass

    @validate_arguments
    @classmethod
    def from_stationary(
        cls,
        reference_velocity_magnitude: VelocityType.Positive,
        atmosphere: ThermalCondition = ThermalCondition(),
    ):
        pass


# TODO: AutomotiveCondition
OperatingConditionType = Union[GenericReferenceCondition, AerospaceCondition]
