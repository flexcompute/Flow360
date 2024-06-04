from typing import Optional, Tuple, Union

import numpy as np
import pydantic as pd
from pydantic import validate_call

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.models.material import Air, FluidMaterialTypes
from flow360.component.simulation.unit_system import (
    DensityType,
    LengthType,
    PressureType,
    TemperatureType,
    VelocityType,
    ViscosityType,
)

VelocityVectorType = Union[VelocityType.Vector, Tuple[pd.StrictStr, pd.StrictStr, pd.StrictStr]]


class ThermalState(Flow360BaseModel):
    temperature: TemperatureType.Positive = 288.15 * u.K
    density: DensityType.Positive = 1.225 * u.kg / u.m**3
    material: FluidMaterialTypes = Air()
    # TODO: special serializer
    _altitude: Optional[LengthType.Positive] = None
    _temperature_offset: Optional[TemperatureType.Positive] = None

    @classmethod
    @validate_call
    def from_standard_atmosphere(
        cls, altitude: LengthType.Positive = 0, temperature_offset: TemperatureType = 0
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
    def altitude(self) -> LengthType.Positive:
        return self._altitude

    @property
    def temperature_offset(self) -> TemperatureType:
        return self._temperature_offset

    @property
    def speed_of_sound(self) -> VelocityType.Positive:
        return np.sqrt(
            self.material.specific_heat_ratio * self.material.gas_constant * self.temperature
        )

    @property
    def pressure(self) -> PressureType.Positive:
        # TODO: implement
        return 1.013e5

    @property
    def dynamic_viscosity(self) -> ViscosityType.Positive:
        # TODO: implement
        return 1.825e-5


class GenericReferenceCondition(Flow360BaseModel):
    """
    Operating condition defines the physical (non-geometrical) reference values for the problem.
    """

    velocity_magnitude: VelocityType.Positive
    thermal_state: ThermalState = ThermalState()

    @classmethod
    @validate_call
    def from_mach(
        cls,
        mach: pd.PositiveFloat,
        thermal_state: ThermalState = ThermalState(),
    ):
        velocity_magnitude = mach * thermal_state.speed_of_sound
        return cls(velocity_magnitude=velocity_magnitude, thermal_state=thermal_state)

    @property
    def mach(self) -> pd.PositiveFloat:
        return self.velocity_magnitude / self.thermal_state.speed_of_sound


class AerospaceCondition(Flow360BaseModel):
    alpha: float = 0
    beta: float = 0
    velocity_magnitude: VelocityType.NonNegative
    atmosphere: ThermalState = ThermalState()
    reference_velocity_magnitude: Optional[VelocityType.Positive] = None

    @classmethod
    @validate_call
    def from_mach(
        cls,
        mach: pd.PositiveFloat,
        alpha: float = 0,
        beta: float = 0,
        atmosphere: ThermalState = ThermalState(),
        reference_mach: Optional[pd.PositiveFloat] = None,
    ):
        pass

    @classmethod
    @validate_call
    def from_stationary(
        cls,
        reference_velocity_magnitude: VelocityType.Positive,
        atmosphere: ThermalState = ThermalState(),
    ):
        pass

    @property
    def mach(self) -> pd.PositiveFloat:
        return self.velocity_magnitude / self.atmosphere.speed_of_sound


# TODO: AutomotiveCondition
OperatingConditionTypes = Union[GenericReferenceCondition, AerospaceCondition]
