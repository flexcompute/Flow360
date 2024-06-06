from typing import Optional, Tuple, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.cached_model_base import CachedModelBase
from flow360.component.simulation.models.material import Air, FluidMaterialTypes
from flow360.component.simulation.unit_system import (
    DensityType,
    LengthType,
    PressureType,
    TemperatureType,
    VelocityType,
    ViscosityType,
)
from flow360.log import log

VelocityVectorType = Union[VelocityType.Vector, Tuple[pd.StrictStr, pd.StrictStr, pd.StrictStr]]


class ThermalStateCache(Flow360BaseModel):
    altitude: Optional[LengthType.Positive] = None
    temperature_offset: Optional[TemperatureType] = None


class ThermalState(CachedModelBase):
    # TODO: romove frozen and throw warning if temperature/density is modified after construction from atmospheric model
    temperature: TemperatureType.Positive = pd.Field(288.15 * u.K, frozen=True)
    density: DensityType.Positive = pd.Field(1.225 * u.kg / u.m**3, frozen=True)
    material: FluidMaterialTypes = pd.Field(Air(), frozen=True)
    _cached: ThermalStateCache = ThermalStateCache()

    @classmethod
    @pd.validate_call
    def from_standard_atmosphere(
        cls, altitude: LengthType.Positive = 0 * u.m, temperature_offset: TemperatureType = 0 * u.K
    ):
        # TODO: add standard atmosphere implementation
        density = 1.225 * u.kg / u.m**3
        temperature = 288.15 * u.K

        state = cls(
            density=density,
            temperature=temperature,
            material=Air(),
        )
        state._cached = ThermalStateCache(altitude=altitude, temperature_offset=temperature_offset)

        return state

    @property
    def altitude(self) -> Optional[LengthType.Positive]:
        if not self._cached.altitude:
            log.warning("Altitude not provided from input")
            return self._cached.altitude

    @property
    def temperature_offset(self) -> Optional[TemperatureType]:
        if not self._cached.altitude:
            log.warning("Temperature offset not provided from input")
        return self._cached.temperature_offset

    @property
    def speed_of_sound(self) -> VelocityType.Positive:
        # TODO: implement
        # return self.material.speed_of_sound(self.temperature)
        return 343 * u.m / u.s

    @property
    def pressure(self) -> PressureType.Positive:
        # TODO: implement
        return 1.013e5 * u.Pa

    @property
    def dynamic_viscosity(self) -> ViscosityType.Positive:
        # TODO: implement
        # return self.material.speed_of_sound(self.temperature)
        return 1.825e-5 * u.Pa * u.s


class GenericReferenceCondition(Flow360BaseModel):
    """
    Operating condition defines the physical (non-geometrical) reference values for the problem.
    """

    velocity_magnitude: VelocityType.Positive
    thermal_state: ThermalState = ThermalState()

    @classmethod
    @pd.validate_call
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
    # TODO: add units for angles
    alpha: float = 0
    beta: float = 0
    velocity_magnitude: VelocityType.NonNegative
    atmosphere: ThermalState = ThermalState()
    reference_velocity_magnitude: Optional[VelocityType.Positive] = None

    @classmethod
    @pd.validate_call
    def from_mach(
        cls,
        mach: pd.PositiveFloat,
        alpha: float = 0,
        beta: float = 0,
        atmosphere: ThermalState = ThermalState(),
        reference_mach: Optional[pd.PositiveFloat] = None,
    ):
        velocity_magnitude = mach * atmosphere.speed_of_sound
        reference_velocity_magnitude = reference_mach * atmosphere.speed_of_sound
        return cls(
            velocity_magnitude=velocity_magnitude,
            alpha=alpha,
            beta=beta,
            atmosphere=atmosphere,
            reference_velocity_magnitude=reference_velocity_magnitude,
        )

    @classmethod
    @pd.validate_call
    def from_stationary(
        cls,
        reference_velocity_magnitude: VelocityType.Positive,
        atmosphere: ThermalState = ThermalState(),
    ):
        return cls(
            velocity_magnitude=0 * u.m / u.s,
            atmosphere=atmosphere,
            reference_velocity_magnitude=reference_velocity_magnitude,
        )

    @property
    def mach(self) -> pd.PositiveFloat:
        return self.velocity_magnitude / self.atmosphere.speed_of_sound


# TODO: AutomotiveCondition
OperatingConditionTypes = Union[GenericReferenceCondition, AerospaceCondition]
